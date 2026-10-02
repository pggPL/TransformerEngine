# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Bulk quantized activation ownership and CPU round trips."""

import contextlib
import copy

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpu_offload import (
    NVTE_CPU_OFFLOAD_V1,
    OffloadableLayerState,
    start_offload,
)
from transformer_engine.pytorch.quantized_tensor import prepare_for_saving, restore_from_saved
from transformer_engine.pytorch.module._split_quantization import _split_quantize
from transformer_engine.pytorch.tensor._bulk import BulkTensorState, restore_tensor_buffers
from transformer_engine.common import recipe
from hybrid_quantization_utils import hybrid_fp8_mxfp8_qfactory


def make_quantizer(kind, rowwise=True, columnwise=True):
    if kind == "fp8_block":
        supported, reason = te.is_fp8_block_scaling_available(return_reason=True)
        if not supported:
            pytest.skip(reason)
        quantizer = te.Float8BlockQuantizer(
            te.DType.kFloat8E4M3, rowwise=rowwise, columnwise=columnwise, block_scaling_dim=1
        )
    elif kind == "mxfp8":
        supported, reason = te.is_mxfp8_available(return_reason=True)
        if not supported:
            pytest.skip(reason)
        quantizer = te.MXFP8Quantizer(te.DType.kFloat8E4M3, rowwise=rowwise, columnwise=columnwise)
    else:
        supported, reason = te.is_nvfp4_available(return_reason=True)
        if not supported:
            pytest.skip(reason)
        quantizer = te.NVFP4Quantizer(
            rowwise=rowwise,
            columnwise=columnwise,
            row_scaled_nvfp4=kind == "nvfp4_row_scaled",
            nvfp4_use_4over6=kind == "nvfp4_4over6",
        )
    quantizer.internal = True
    return quantizer


@pytest.mark.parametrize(
    "kind", ["fp8_block", "mxfp8", "nvfp4", "nvfp4_row_scaled", "nvfp4_4over6"]
)
@pytest.mark.parametrize("usage", ["rowwise", "columnwise", "both"])
def test_bulk_quantized_round_trip(kind, usage):
    splits = [128, 0, 256]
    quantizers = [
        make_quantizer(
            kind,
            rowwise=usage != "columnwise" or kind == "nvfp4_row_scaled",
            columnwise=usage != "rowwise",
        )
        for _ in splits
    ]
    x = torch.randn((sum(splits), 128), device="cuda", dtype=torch.bfloat16)
    outputs, buffers, descriptions = tex.split_quantize(x, splits, quantizers)
    original, objects = prepare_for_saving(*outputs)
    rebuilt = tex.get_tensors(buffers, descriptions)
    assert len(rebuilt) == len(original)
    for actual, expected in zip(rebuilt, original):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    restore_from_saved(objects, original)
    del original, rebuilt
    for output in outputs:
        output.update_usage(
            rowwise_usage=usage != "columnwise", columnwise_usage=usage != "rowwise"
        )
    tensors, objects = prepare_for_saving(*outputs)
    reference = [None if tensor is None else tensor.cpu() for tensor in tensors]
    restore_from_saved(objects, tensors)
    del tensors
    packed, layout = tex.save_tensor_buffers(outputs, buffers, descriptions)
    assert len(packed) == (2 if usage == "both" else 1)
    tex.restore_tensor_buffers(outputs, [tensor.cpu().cuda() for tensor in packed], layout)
    restored, objects = prepare_for_saving(*outputs)
    for actual, expected in zip(restored, reference):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    rebuilt = restore_from_saved(objects, restored)
    assert [tuple(output.size()) for output in rebuilt] == [(rows, 128) for rows in splits]


@pytest.mark.parametrize(
    "kind", ["fp8_block", "mxfp8", "nvfp4", "nvfp4_row_scaled", "nvfp4_4over6"]
)
def test_bulk_offload_releases_allocation(kind):
    splits = [256] * 8
    quantizers = [make_quantizer(kind) for _ in splits]
    x = torch.randn((sum(splits), 512), device="cuda", dtype=torch.bfloat16)
    tex.split_quantize(x, splits, quantizers)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    outputs, buffers, descriptions = tex.split_quantize(x, splits, quantizers)
    for output in outputs:
        output.update_usage(rowwise_usage=False, columnwise_usage=True)
    tensors, objects = prepare_for_saving(*outputs)
    reference = [None if tensor is None else tensor.cpu() for tensor in tensors]
    restore_from_saved(objects, tensors)
    del tensors
    packed, layout = tex.save_tensor_buffers(outputs, buffers, descriptions)
    state = OffloadableLayerState(offload_stream=torch.cuda.Stream())
    start_offload(*packed)
    handles = [state.push_tensor(tensor) for tensor in packed]
    assert len(state.fwd_gpu_tensor_group.tensor_list) == 1
    del packed, buffers
    state.start_offload()
    state.release_activation_forward_gpu_memory()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    state.start_reload()
    restored_buffers = [state.pop_tensor(handle) for handle in handles]
    tex.restore_tensor_buffers(outputs, restored_buffers, layout)
    restored, objects = prepare_for_saving(*outputs)
    for actual, expected in zip(restored, reference):
        if expected is not None:
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    restore_from_saved(objects, restored)
    state.release_all_memory()


@pytest.mark.parametrize(
    "kind", ["fp8_block", "mxfp8", "nvfp4", "nvfp4_row_scaled", "nvfp4_4over6"]
)
def test_get_tensors_buffer_validation_and_lifetime(kind):
    x = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    outputs, buffers, descriptions = tex.split_quantize(
        x, [128, 0, 128], [make_quantizer(kind)] * 3
    )
    expected, _ = prepare_for_saving(*outputs)
    restored_buffers = [buffer.clone() for buffer in buffers]
    rebuilt = tex.get_tensors(restored_buffers, descriptions)
    assert len(rebuilt) == len(expected)
    del buffers, restored_buffers, outputs
    for actual, reference in zip(rebuilt, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)
    with pytest.raises(RuntimeError, match="Invalid buffer index"):
        tex.get_tensors([], descriptions)
    with pytest.raises(RuntimeError, match="backing buffer"):
        tex.get_tensors([torch.empty(1, dtype=torch.uint8, device="cuda")] * 2, descriptions)
    with pytest.raises(RuntimeError, match="uint8 buffer"):
        tex.get_tensors([x] * 2, descriptions)


def test_split_quantize_without_bulk_buffers():
    supported, reason = te.is_fp8_available(return_reason=True)
    if not supported:
        pytest.skip(reason)
    quantizer = te.Float8CurrentScalingQuantizer(te.DType.kFloat8E4M3, device="cuda")
    x = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    outputs, buffers, _ = tex.split_quantize(x, [128, 128], [quantizer, quantizer])
    assert len(outputs) == 2
    assert not buffers
    outputs, buffers, descriptions = tex.split_quantize(x[:0], [], [])
    assert not outputs and not buffers
    assert not tex.get_tensors(buffers, descriptions)


@pytest.mark.parametrize("row_bulk", [False, True])
@pytest.mark.parametrize("column_bulk", [False, True])
@pytest.mark.parametrize("source", ["original", "rowwise_dequantized"])
@pytest.mark.parametrize("usage", ["rowwise", "columnwise", "both"])
def test_hybrid_bulk_layout(row_bulk, column_bulk, source, usage):
    make_quantizer("mxfp8")
    splits = [128, 0, 256]
    quantizers = [
        te.HybridQuantizer(
            rowwise_quantizer=(
                make_quantizer("mxfp8")
                if row_bulk
                else te.Float8CurrentScalingQuantizer(te.DType.kFloat8E4M3, device="cuda")
            ),
            columnwise_quantizer=(
                make_quantizer("mxfp8")
                if column_bulk
                else te.Float8CurrentScalingQuantizer(te.DType.kFloat8E4M3, device="cuda")
            ),
            columnwise_source=source,
        )
        for _ in splits
    ]
    for quantizer in quantizers:
        quantizer.internal = True
    x = torch.randn((sum(splits), 128), device="cuda", dtype=torch.bfloat16)
    state = BulkTensorState()
    outputs, _ = _split_quantize(x, splits, quantizers, x.dtype, bulk_state=state)
    for output in outputs:
        output.update_usage(
            rowwise_usage=usage != "columnwise", columnwise_usage=usage != "rowwise"
        )
    flat, objects = prepare_for_saving(*outputs)
    reference = [None if tensor is None else tensor.cpu() for tensor in flat]
    restore_from_saved(objects, flat)
    del flat
    offload_tensors = state.offload_tensors(outputs)
    if not row_bulk and usage != "columnwise":
        assert (
            any(tensor is outputs[0]._rowwise_storage for tensor in offload_tensors)
            or not state.groups
        )
    if not column_bulk and usage != "rowwise":
        assert (
            any(tensor is outputs[0]._columnwise_storage for tensor in offload_tensors)
            or not state.groups
        )
    saved_buffers, metadata = state.prepare_for_saving(outputs)
    assert len(saved_buffers) == int(row_bulk and usage != "columnwise") + int(
        column_bulk and usage != "rowwise"
    )
    flat, objects = prepare_for_saving(*outputs)
    rebuilt = restore_from_saved(objects, [None if t is None else t.cpu().cuda() for t in flat])
    restore_tensor_buffers(rebuilt, [t.cpu().cuda() for t in saved_buffers], metadata)
    actual, _ = prepare_for_saving(*rebuilt)
    assert len(actual) == len(reference)
    for actual_tensor, expected in zip(actual, reference):
        if expected is None:
            assert actual_tensor is None
        else:
            torch.testing.assert_close(
                actual_tensor.cpu(), expected, rtol=0, atol=0, equal_nan=True
            )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bulk_allocate_layout(device):
    shapes = [(2, 3), (), (0, 4), (1, 2, 3, 4, 5)]
    dtypes = [torch.float32, torch.uint8, torch.float16, torch.float64]
    alignments = [256, 16, 256, 16]
    tensors = tex.bulk_allocate(shapes, dtypes, torch.device(device), alignments)
    for index, (tensor, shape, dtype, alignment) in enumerate(
        zip(tensors, shapes, dtypes, alignments)
    ):
        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.is_contiguous()
        assert tensor.data_ptr() % alignment == 0
        tensor.fill_(index + 1)
    assert tensors[1].data_ptr() - tensors[0].data_ptr() == 32
    assert tensors[2].data_ptr() == 0
    assert tensors[3].data_ptr() - tensors[0].data_ptr() == 256
    for index, tensor in enumerate(tensors):
        assert torch.all(tensor == index + 1)
    retained = tensors[-1]
    del tensors, tensor
    assert retained.sum().item() == 480


@pytest.mark.parametrize("kind", ["fp8_block", "mxfp8", "nvfp4", "hybrid"])
@pytest.mark.parametrize("saving", ["quantized", "original", "dequantized", "frozen"])
def test_grouped_linear_bulk_offload(kind, saving, monkeypatch):
    make_quantizer("mxfp8" if kind == "hybrid" else kind)
    recipes = {
        "fp8_block": recipe.Float8BlockScaling,
        "mxfp8": recipe.MXFP8BlockScaling,
        "nvfp4": lambda: recipe.NVFP4BlockScaling(disable_stochastic_rounding=True),
        "hybrid": lambda: recipe.CustomRecipe(qfactory=hybrid_fp8_mxfp8_qfactory),
    }
    quantization = recipes[kind]()
    if saving == "dequantized":
        quantization.backward_override = "dequantized"
    splits = [256, 0, 256, 512]
    layer = te.GroupedLinear(
        len(splits),
        512,
        512,
        params_dtype=torch.bfloat16,
        save_original_input=saving == "original",
        use_grouped_tensor=False,
    )
    if saving == "frozen":
        for name, parameter in layer.named_parameters():
            if "weight" in name:
                parameter.requires_grad_(False)
    offloaded = copy.deepcopy(layer)
    x = torch.randn((sum(splits), 512), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    offload_context, commit = te.get_cpu_offload_context(enabled=True, num_layers=1, model_layers=2)
    transfers = []
    original_start = OffloadableLayerState.start_offload

    def record_transfers(state):
        transfers.append([tensor.dtype for tensor in state.fwd_gpu_tensor_group.tensor_list])
        return original_start(state)

    monkeypatch.setattr(OffloadableLayerState, "start_offload", record_transfers)

    def run(module, offload):
        module.zero_grad(set_to_none=True)
        x.grad = None
        with offload_context if offload else contextlib.nullcontext():
            with te.autocast(enabled=True, recipe=quantization):
                output = module(x, splits)
        if offload:
            output = commit(output)
            with offload_context:
                output = output + 0
            output = commit(output)
        output.backward(dy)
        return [output.detach().clone(), x.grad.clone()] + [
            parameter.grad.clone() for parameter in module.parameters() if parameter.requires_grad
        ]

    for _ in range(3):
        expected = run(layer, False)
        actual = run(offloaded, True)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
    if not NVTE_CPU_OFFLOAD_V1 and saving in ("quantized", "dequantized"):
        assert transfers == [[torch.uint8]] * 3
