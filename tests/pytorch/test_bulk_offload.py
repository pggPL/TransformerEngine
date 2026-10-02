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
from transformer_engine.pytorch.tensor._bulk import pack_tensor_buffers, unpack_tensor_buffers
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
        quantizer = te.NVFP4Quantizer(rowwise=rowwise, columnwise=columnwise)
    quantizer.internal = True
    return quantizer


@pytest.mark.parametrize("kind", ["fp8_block", "mxfp8", "nvfp4"])
@pytest.mark.parametrize("usage", ["rowwise", "columnwise", "both"])
def test_bulk_quantized_round_trip(kind, usage):
    splits = [128, 0, 256]
    quantizers = [make_quantizer(kind) for _ in splits]
    x = torch.randn((sum(splits), 128), device="cuda", dtype=torch.bfloat16)
    outputs, buffers = tex.split_quantize(x, splits, quantizers)
    for output in outputs:
        output.update_usage(
            rowwise_usage=usage != "columnwise", columnwise_usage=usage != "rowwise"
        )
    tensors, objects = prepare_for_saving(*outputs)
    reference = [None if tensor is None else tensor.cpu() for tensor in tensors]
    packed, metadata = pack_tensor_buffers(tensors, buffers)
    owners = [tensor for tensor in packed if any(tensor is buffer for buffer in buffers)]
    assert len(owners) == (2 if usage == "both" else 1)
    restored = unpack_tensor_buffers(
        [None if tensor is None else tensor.cpu().cuda() for tensor in packed], metadata
    )
    for actual, expected in zip(restored, reference):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    rebuilt = restore_from_saved(objects, restored)
    assert len(rebuilt) == len(splits)
    assert [tuple(output.size()) for output in rebuilt] == [(rows, 128) for rows in splits]


@pytest.mark.parametrize("kind", ["fp8_block", "mxfp8", "nvfp4"])
def test_bulk_offload_releases_allocation(kind):
    splits = [256] * 8
    quantizers = [make_quantizer(kind) for _ in splits]
    x = torch.randn((sum(splits), 512), device="cuda", dtype=torch.bfloat16)
    # Warm up lazy quantization state before measuring live allocations.
    tex.split_quantize(x, splits, quantizers)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    outputs, buffers = tex.split_quantize(x, splits, quantizers)
    for output in outputs:
        output.update_usage(rowwise_usage=False, columnwise_usage=True)
    tensors, objects = prepare_for_saving(*outputs)
    reference = [None if tensor is None else tensor.cpu() for tensor in tensors]
    packed, metadata = pack_tensor_buffers(tensors, buffers)
    state = OffloadableLayerState(offload_stream=torch.cuda.Stream())
    start_offload(*packed)
    handles = [state.push_tensor(tensor) if tensor is not None else None for tensor in packed]
    assert len(state.fwd_gpu_tensor_group.tensor_list) == 1
    del tensors, packed, buffers, outputs, output
    state.start_offload()
    state.release_activation_forward_gpu_memory()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    state.start_reload()
    restored = unpack_tensor_buffers(
        [state.pop_tensor(handle) if handle is not None else None for handle in handles], metadata
    )
    for actual, expected in zip(restored, reference):
        if expected is not None:
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    restore_from_saved(objects, restored)
    state.release_all_memory()


def test_bulk_buffer_typed_strided_views():
    buffer = torch.arange(256, dtype=torch.uint8, device="cuda")
    data = buffer[16:80].view(torch.float32).as_strided((2, 3), (4, 1))
    other = torch.randn(3, device="cuda")
    packed, metadata = pack_tensor_buffers([data, other, None, data[:, :0]], [buffer])
    restored = unpack_tensor_buffers(
        [tensor.clone() if tensor is not None else None for tensor in packed], metadata
    )
    for actual, expected in zip(restored, [data, other, None, data[:, :0]]):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored[0].stride() == data.stride()
    assert restored[-1].untyped_storage().data_ptr() == restored[0].untyped_storage().data_ptr()


def test_split_quantize_without_bulk_buffers():
    supported, reason = te.is_fp8_available(return_reason=True)
    if not supported:
        pytest.skip(reason)
    quantizer = te.Float8CurrentScalingQuantizer(te.DType.kFloat8E4M3, device="cuda")
    x = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    outputs, buffers = tex.split_quantize(x, [128, 128], [quantizer, quantizer])
    assert len(outputs) == 2
    assert not buffers
    outputs, buffers = tex.split_quantize(x[:0], [], [])
    assert not outputs and not buffers


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
