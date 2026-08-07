# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Proof of concept: a pipeline-level ``autograd.Function`` traced as a HOP.

This is the load-bearing assumption behind compiling ``ops.Sequential``, checked
end to end before the fuser is touched:

1. Dynamo traces a ``torch.autograd.Function`` as the ``autograd_function_apply``
   higher-order op, so **both** its forward and its backward end up in the graph.
2. That lets the forward and the backward walk **different op groupings**, which
   is what ``OperationFuser`` does -- forward fuses linear+bias while backward
   fuses bias+activation, and the two partitions overlap without nesting. An op
   whose autograd is registered per op could not do this: autograd would compose
   the backward out of the forward's nodes.
3. ``OperationContext`` objects are created inside the forward and read in the
   backward. Dynamo permits mutating objects created *within* the HOP scope, so
   they never have to cross an op schema.
4. A quantized (FP8) tensor is produced by one op and consumed by the next
   *inside* the traced region.

The ops themselves are the real registered custom ops; only the pipeline driver
is written out longhand here, standing in for the fuser.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch.ops.basic.activation import ActivationBwdArgs
from transformer_engine.pytorch.ops.basic.bias import BiasBwdArgs
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_device = "cuda"


class _OpCtx:
    """Stand-in for ``OperationContext``: created in forward, read in backward."""

    def __init__(self) -> None:
        self.saved: tuple = ()
        self.attrs: dict = {}


def _fp8_quantizer() -> Any:
    quantizer = Float8CurrentScalingQuantizer(DType.kFloat8E4M3, torch.device(_device))
    quantizer.set_usage(rowwise=True, columnwise=False)
    return quantizer


class _Pipeline(torch.autograd.Function):
    """Runs a fixed 3-op pipeline: Bias -> GELU -> Bias.

    The forward walks the ops one by one; the backward walks a *coarser*
    grouping, handling (bias1, gelu) as a single step. That asymmetry is the
    whole point -- it is only expressible because autograd is owned here, at the
    pipeline level, rather than by each op.
    """

    @staticmethod
    def forward(func_ctx, x, bias1, bias2, ops, quantize_middle):
        bias_fwd, _ = ops["bias"]
        act_fwd, _ = ops["act"]

        ctxs = [_OpCtx() for _ in range(3)]

        # op 0: bias
        args0 = ops["bias_op1"].resolve_fwd_args(
            x, requires_grad=True, prev_op_grad_output_quantizer=None
        )
        y0, saved0, attrs0 = bias_fwd(args0)
        ctxs[0].saved, ctxs[0].attrs = saved0, attrs0

        # op 1: activation, quantizing its output with the next op's quantizer
        args1 = ops["act_op"].resolve_fwd_args(
            y0,
            requires_grad=True,
            prev_op_grad_output_quantizer=None,
            next_op_input_quantizer=_fp8_quantizer() if quantize_middle else None,
        )
        y1, saved1, attrs1 = act_fwd(args1)
        ctxs[1].saved = ops["act_op"].saved_for_backward(saved1, y0)
        ctxs[1].attrs = attrs1

        # op 2: bias, consuming what op 1 produced (FP8 when quantize_middle)
        args2 = ops["bias_op2"].resolve_fwd_args(
            y1, requires_grad=True, prev_op_grad_output_quantizer=None
        )
        y2, saved2, attrs2 = bias_fwd(args2)
        ctxs[2].saved, ctxs[2].attrs = saved2, attrs2

        func_ctx.ctxs = ctxs
        func_ctx.ops = ops
        func_ctx.save_for_backward(x, bias1, bias2)
        return y2

    @staticmethod
    def backward(func_ctx, grad_output):
        _, bias_bwd = func_ctx.ops["bias"]
        _, act_bwd = func_ctx.ops["act"]
        ctxs = func_ctx.ctxs

        # Backward group A: op 2 alone.
        dy2, db2 = bias_bwd(
            BiasBwdArgs(
                grad_output=grad_output,
                grad_input_quantizer=ctxs[2].attrs["grad_input_quantizer"],
            )
        )
        if dy2 is None:
            dy2 = grad_output

        # Backward group B: ops 1 and 0 handled together -- a coarser grouping
        # than the forward used. A real fused backward op would replace these
        # two calls; what matters here is that the grouping may differ at all.
        (dy1,) = act_bwd(
            ActivationBwdArgs(
                grad_output=dy2,
                saved_input=ctxs[1].saved[0],
                dtype=ctxs[1].attrs["dtype"],
                grad_input_quantizer=ctxs[1].attrs["prev_op_grad_output_quantizer"],
            )
        )
        dy0, db1 = bias_bwd(
            BiasBwdArgs(
                grad_output=dy1,
                grad_input_quantizer=ctxs[0].attrs["grad_input_quantizer"],
            )
        )
        if dy0 is None:
            dy0 = dy1

        return dy0, db1, db2, None, None


def _build(dtype: torch.dtype):
    bias_op1 = te.ops.Bias(64, device=_device, dtype=dtype)
    bias_op2 = te.ops.Bias(64, device=_device, dtype=dtype)
    act_op = te.ops.GELU()
    with torch.no_grad():
        bias_op1.bias.copy_(torch.randn_like(bias_op1.bias))
        bias_op2.bias.copy_(torch.randn_like(bias_op2.bias))
    ops = {
        "bias": bias_op1.compile_ops,
        "act": act_op.compile_ops,
        "bias_op1": bias_op1,
        "bias_op2": bias_op2,
        "act_op": act_op,
    }
    return ops, bias_op1, bias_op2


def _run(x, ops, bias_op1, bias_op2, quantize_middle):
    return _Pipeline.apply(x, bias_op1.bias, bias_op2.bias, ops, quantize_middle)


@_cuda
@pytest.mark.parametrize("quantize_middle", [False, True])
def test_hop_pipeline_compiles_fullgraph(quantize_middle) -> None:
    """The pipeline must trace as a HOP and match eager, forward and backward."""
    dtype = torch.bfloat16
    ops, bias_op1, bias_op2 = _build(dtype)
    x = torch.randn(16, 64, device=_device, dtype=dtype, requires_grad=True)
    dy = torch.randn(16, 64, device=_device, dtype=dtype)

    def step(x_):
        return _run(x_, ops, bias_op1, bias_op2, quantize_middle)

    # Eager reference.
    out_ref = step(x)
    out_ref.backward(dy)
    grads_ref = [x.grad.clone(), bias_op1.bias.grad.clone(), bias_op2.bias.grad.clone()]

    for t in (x, bias_op1.bias, bias_op2.bias):
        t.grad = None

    compiled = torch.compile(step, fullgraph=True)
    out = compiled(x)
    out.backward(dy)
    grads = [x.grad, bias_op1.bias.grad, bias_op2.bias.grad]

    torch.testing.assert_close(out, out_ref)
    for got, expected in zip(grads, grads_ref):
        torch.testing.assert_close(got, expected)


@_cuda
def test_hop_pipeline_carries_fp8_between_ops() -> None:
    """The middle tensor really is FP8, and it is consumed by the next op."""
    dtype = torch.bfloat16
    ops, bias_op1, bias_op2 = _build(dtype)
    x = torch.randn(16, 64, device=_device, dtype=dtype, requires_grad=True)

    bias_fwd, _ = ops["bias"]
    act_fwd, _ = ops["act"]
    args0 = ops["bias_op1"].resolve_fwd_args(
        x.detach(), requires_grad=False, prev_op_grad_output_quantizer=None
    )
    y0, _, _ = bias_fwd(args0)
    args1 = ops["act_op"].resolve_fwd_args(
        y0,
        requires_grad=False,
        prev_op_grad_output_quantizer=None,
        next_op_input_quantizer=_fp8_quantizer(),
    )
    y1, _, _ = act_fwd(args1)
    assert isinstance(y1, QuantizedTensorStorage), f"expected FP8, got {type(y1)}"

    args2 = ops["bias_op2"].resolve_fwd_args(
        y1, requires_grad=False, prev_op_grad_output_quantizer=None
    )
    y2, _, _ = bias_fwd(args2)
    assert y2 is not None
