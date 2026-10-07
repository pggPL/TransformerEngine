# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Alignment guarantees used to select grouped MLP fusion."""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.ep import EpConfig
from transformer_engine.pytorch.ops.fuser import OperationFuser
from transformer_engine.pytorch.ops.fused.grouped_mlp import (
    GroupedMLP_CuTeGEMMGLU,
    GroupedMLP_CuTeGEMMUnary,
    fuse_glu_ops,
    fuse_unary_activation_ops,
)
from transformer_engine.pytorch.tensor import GroupedTensor, MXFP8Quantizer


@pytest.fixture(autouse=True)
def enable_grouped_fusions(monkeypatch):
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    for cls in (GroupedMLP_CuTeGEMMGLU, GroupedMLP_CuTeGEMMUnary):
        cls.is_supported.cache_clear()
    if not GroupedMLP_CuTeGEMMGLU.is_supported():
        pytest.skip("CuTeDSL grouped MLP is unavailable")
    monkeypatch.setattr(
        OperationFuser,
        "forward_backward_fusion_functions",
        [fuse_glu_ops, fuse_unary_activation_ops],
    )
    yield
    for cls in (GroupedMLP_CuTeGEMMGLU, GroupedMLP_CuTeGEMMUnary):
        cls.is_supported.cache_clear()


def make_ops(unary=False, single_grouped_weight=False):
    width = 256 if unary else 512
    kwargs = dict(
        bias=False, device="cuda", dtype=torch.bfloat16, single_grouped_weight=single_grouped_weight
    )
    return [
        te.ops.GroupedLinear(4, 256, width, **kwargs),
        te.ops.ScaledSReLU() if unary else te.ops.ScaledSwiGLU(glu_interleave_size=32),
        te.ops.GroupedLinear(4, 256, 256, **kwargs),
    ]


def is_fused(fuser):
    return any(
        isinstance(op, (GroupedMLP_CuTeGEMMGLU, GroupedMLP_CuTeGEMMUnary))
        for op, _ in fuser._forward_ops
    )


@pytest.mark.parametrize("alignment", [1, 128, 256, 384, 512])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("unary", [False, True])
def test_grouped_mlp_alignment_guarantee(alignment, quantized, unary):
    if unary and not GroupedMLP_CuTeGEMMUnary.is_supported():
        pytest.skip("CuTeDSL SReLU is unavailable")
    # Every declared divisor is truthful; 384 still does not prove 256 alignment.
    splits = torch.tensor([0, 1536, 1536, 1536], dtype=torch.int64, device="cuda")
    x = torch.randn(4608, 256, dtype=torch.bfloat16, device="cuda")
    if quantized:
        quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
        grouped = tex.group_quantize(x, quantizer, 4, splits)
        grouped.row_alignment = alignment
    else:
        grouped = GroupedTensor.from_tensor(x, splits, row_alignment=alignment)
    fuser = OperationFuser(make_ops(unary))
    fuser.maybe_fuse_ops(True, MXFP8BlockScaling(), grouped, [(splits,), (None,), (splits,)])
    assert is_fused(fuser) == (alignment in (256, 512))


def test_grouped_mlp_alignment_cache_and_split_aliases():
    splits = torch.tensor([0, 512, 512, 512], dtype=torch.int64, device="cuda")
    x = torch.randn(1536, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    fuser = OperationFuser(make_ops())
    plans = {}
    for alignment in (256, 128, 512, 1, 256, None):
        inp = (
            x
            if alignment is None
            else GroupedTensor.from_tensor(x, splits, row_alignment=alignment)
        )
        fuser.maybe_fuse_ops(True, MXFP8BlockScaling(), inp, [(splits,), (None,), (splits,)])
        expected = alignment in (256, 512)
        assert is_fused(fuser) == expected
        if expected in plans:
            assert fuser._forward_ops is plans[expected]
        plans[expected] = fuser._forward_ops
    assert len(fuser._fused_ops_cache) == 2

    grouped = GroupedTensor.from_tensor(x, splits, row_alignment=256)
    for fc2_splits, expected in (
        (splits.detach(), True),
        (splits.clone(), False),
        (splits.view(torch.float64), False),
        (splits.as_strided(splits.shape, (0,)), False),
        (splits, True),
    ):
        fuser.maybe_fuse_ops(
            True, MXFP8BlockScaling(), grouped, [(splits,), (None,), (fc2_splits,)]
        )
        assert is_fused(fuser) == expected
        assert fuser._forward_ops is plans[expected]


@pytest.mark.parametrize("alignment", [0, 128, 256, 512])
@pytest.mark.parametrize("bound_splits", [0, 1, 2])
def test_grouped_mlp_dispatch_alignment(alignment, bound_splits):
    config = EpConfig(
        top_k=1,
        hidden_dim=256,
        num_local_experts=4,
        max_tokens_per_rank=1024,
        recv_capacity_per_rank=4096,
        ep_group=None,
        alignment=alignment,
    )
    dispatch = te.ops.MoeDispatch(config)
    dispatch.set_extra_output_channel(0, "splits", output_to_caller=False)
    dispatch.set_extra_output_channel(1, "probs", output_to_caller=False)
    fc1, activation, fc2 = make_ops()
    for linear in (fc1, fc2)[:bound_splits]:
        linear.set_extra_input_channel(0, "splits")
    activation.set_extra_input_channel(0, "probs")
    fuser = OperationFuser([dispatch, fc1, activation, fc2])
    x = torch.empty(1024, 256, device="cuda", dtype=torch.bfloat16)
    external_splits = torch.zeros(4, device="cuda", dtype=torch.int64)
    extra_inputs = [(None, None), (external_splits,), (None,), (external_splits,)]
    fuser.maybe_fuse_ops(True, MXFP8BlockScaling(), x, extra_inputs)
    assert is_fused(fuser) == (alignment in (256, 512) and bound_splits == 2)


@pytest.mark.parametrize("wrapped", [False, True])
def test_unaligned_grouped_mlp_matches_unfused(wrapped, monkeypatch):
    torch.manual_seed(1234)
    splits = torch.tensor([0, 128, 384, 256], device="cuda", dtype=torch.int64)
    x = torch.randn(768, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(768, device="cuda", dtype=x.dtype, requires_grad=True)
    grad = torch.randn_like(x)
    mlp = te.ops.Sequential(*make_ops())
    reference = te.ops.Sequential(*make_ops())
    reference.load_state_dict(mlp.state_dict())

    def run(module, inp):
        x.grad = None
        probs.grad = None
        with te.autocast(recipe=MXFP8BlockScaling()):
            out = module(inp, splits, probs, splits)
        out.backward(grad)
        return [out.detach(), x.grad.clone(), probs.grad.clone()] + [
            param.grad.clone() for param in module.parameters()
        ]

    with monkeypatch.context() as patch:
        patch.setattr(OperationFuser, "forward_backward_fusion_functions", [])
        expected = run(reference, x)
    inp = GroupedTensor.from_tensor(x, splits, row_alignment=128) if wrapped else x
    actual = run(mlp, inp)
    assert not is_fused(mlp._module_groups[0])
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_unaligned_grouped_mlp_cuda_graph():
    torch.manual_seed(1234)
    splits = torch.tensor([0, 128, 384, 256], device="cuda", dtype=torch.int64)
    x = torch.randn(768, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(768, device="cuda", dtype=x.dtype, requires_grad=True)
    grad = torch.randn_like(x)
    mlp = te.ops.Sequential(*make_ops())
    reference = te.ops.Sequential(*make_ops())
    reference.load_state_dict(mlp.state_dict())

    def forward_backward():
        inp = GroupedTensor.from_tensor(x, splits, row_alignment=128)
        with te.autocast(recipe=MXFP8BlockScaling()):
            out = mlp(inp, splits, probs, splits)
        out.backward(grad)
        return out

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            mlp.zero_grad(set_to_none=True)
            x.grad = probs.grad = None
            forward_backward()
    torch.cuda.current_stream().wait_stream(stream)
    mlp.zero_grad(set_to_none=True)
    x.grad = probs.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = forward_backward()
    assert not is_fused(mlp._module_groups[0])

    for rows in ([128, 384, 0, 256], [256, 0, 128, 384]):
        splits.copy_(torch.tensor(rows, device="cuda"))
        with torch.no_grad():
            x.copy_(torch.randn_like(x))
            probs.copy_(torch.rand_like(probs))
        graph.replay()
        reference.zero_grad(set_to_none=True)
        x_ref = x.detach().clone().requires_grad_()
        probs_ref = probs.detach().clone().requires_grad_()
        with te.autocast(recipe=MXFP8BlockScaling()):
            expected = reference(x_ref, splits, probs_ref, splits)
        expected.backward(grad)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)
        torch.testing.assert_close(probs.grad, probs_ref.grad, rtol=0, atol=0)
        for param, ref_param in zip(mlp.parameters(), reference.parameters()):
            torch.testing.assert_close(param.grad, ref_param.grad, rtol=0, atol=0)


@pytest.mark.parametrize("single_grouped_weight", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_grouped_mlp_interleaved_alignment_backward(single_grouped_weight, reverse):
    """Each outstanding forward must retain its own fused/unfused backward plan."""
    torch.manual_seed(1234)
    mlp = te.ops.Sequential(*make_ops(single_grouped_weight=single_grouped_weight))
    reference = te.ops.Sequential(*make_ops(single_grouped_weight=single_grouped_weight))
    reference.load_state_dict(mlp.state_dict())
    inputs = []
    for rows, alignment in (([0, 256, 512, 256], 256), ([128, 128, 384, 384], 128)):
        splits = torch.tensor(rows, device="cuda", dtype=torch.int64)
        x = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        probs = torch.rand(1024, device="cuda", dtype=x.dtype, requires_grad=True)
        inputs.append((x, splits, probs, alignment, torch.randn_like(x)))

    def forward(module, values):
        x, splits, probs, alignment, _ = values
        inp = GroupedTensor.from_tensor(x, splits, row_alignment=alignment)
        with te.autocast(recipe=MXFP8BlockScaling()):
            out = module(inp, splits, probs, splits)
        assert is_fused(module._module_groups[0]) == (alignment == 256)
        return out

    def gradients(module, out, values):
        x, _, probs, _, grad = values
        grads = torch.autograd.grad(out, (x, probs, *module.parameters()), grad)
        return [
            (g.rowwise_data.view(g.shape) if isinstance(g, GroupedTensor) else g).clone()
            for g in grads
        ]

    expected = []
    for values in inputs:
        out = forward(reference, values)
        expected.append((out.detach().clone(), gradients(reference, out, values)))

    order = [1, 0] if reverse else [0, 1]
    outputs = {idx: forward(mlp, inputs[idx]) for idx in order}
    for idx in order:
        out = outputs[idx]
        actual_grads = gradients(mlp, out, inputs[idx])
        expected_out, expected_grads = expected[idx]
        torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
        for actual, ref in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
