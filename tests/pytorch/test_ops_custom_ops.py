# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Per-op custom ops for ``transformer_engine.pytorch.ops``.

Each fusible operation registers its forward and backward as independent custom
ops (``register_op_halves``) so a compiled pipeline can call them directly. The
tests here check each half in isolation -- numerics against the eager op, and
that the data-free fake agrees with the real impl slot for slot.

The fake is what the compiler believes; if it disagrees with the real impl the
result is a silently misassembled tensor rather than an error, so the
conformance check runs for every op.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.dynamo import TensorSpec
from transformer_engine.pytorch.ops.basic.bias import (
    BiasBwdArgs,
    BiasFwdArgs,
    _bias_backward_impl,
    _bias_backward_impl_fake,
    _bias_forward_impl,
    _bias_forward_impl_fake,
    _bias_ops,
)
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage

# Test setup
_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_device = "cuda"


# --------------------------------------------------------------------------- #
# Conformance: the fake must describe what the real impl produces
# --------------------------------------------------------------------------- #


def _describe(value: Any) -> Optional[Tuple]:
    """Structural fingerprint of a real tensor or of the spec describing it."""
    if value is None:
        return None
    if isinstance(value, TensorSpec):
        quantizer = value.quantizer
        return (tuple(value.shape), value.dtype, type(quantizer) if quantizer else None)
    quantizer = getattr(value, "_quantizer", None)
    if not isinstance(value, QuantizedTensorStorage):
        quantizer = None
    return (tuple(value.shape), value.dtype, type(quantizer) if quantizer else None)


def _describe_all(values: Any) -> List[Optional[Tuple]]:
    if values is None:
        return []
    if not isinstance(values, (tuple, list)):
        values = (values,)
    return [_describe(v) for v in values]


def assert_fwd_fake_matches_real(args: Any, impl, fake_impl) -> None:
    """Run a forward impl and its fake on the same args; require agreement.

    Compares user outputs, saved tensors (count included -- the compiled path
    slices a flat payload by the fake's saved-tensor list) and ``ctx_attrs``
    keys.
    """
    real_out, real_saved, real_attrs = impl(args)
    fake_out, fake_saved, fake_attrs = fake_impl(args)

    assert _describe_all(real_out) == _describe_all(fake_out), "forward outputs disagree"
    assert _describe_all(real_saved) == _describe_all(fake_saved), "saved tensors disagree"
    assert set(real_attrs) == set(fake_attrs), "ctx_attrs keys disagree"


def assert_bwd_fake_matches_real(args: Any, impl, fake_impl) -> None:
    """Run a backward impl and its fake on the same args; require agreement."""
    real = impl(args)
    fake = fake_impl(args)
    assert _describe_all(real) == _describe_all(fake), "gradients disagree"


# --------------------------------------------------------------------------- #
# Bias
# --------------------------------------------------------------------------- #


def _bias_op(size: int, dtype: torch.dtype) -> te.ops.Bias:
    op = te.ops.Bias(size, device=_device, dtype=dtype)
    with torch.no_grad():
        op.bias.copy_(torch.randn_like(op.bias))
    return op


def _bias_fwd_args(shape, size: int, dtype: torch.dtype) -> BiasFwdArgs:
    op = _bias_op(size, dtype)
    x = torch.randn(*shape, device=_device, dtype=dtype)
    return op.resolve_fwd_args(x, requires_grad=True, prev_op_grad_output_quantizer=None)


@_cuda
@pytest.mark.parametrize("shape", [(16, 32), (2, 8, 32), (32,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bias_fake_conformance(shape, dtype) -> None:
    args = _bias_fwd_args(shape, shape[-1], dtype)
    assert_fwd_fake_matches_real(args, _bias_forward_impl, _bias_forward_impl_fake)

    dy = torch.randn(*shape, device=_device, dtype=dtype)
    bwd_args = BiasBwdArgs(grad_output=dy, grad_input_quantizer=None)
    assert_bwd_fake_matches_real(bwd_args, _bias_backward_impl, _bias_backward_impl_fake)


@_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bias_custom_op_matches_eager(dtype) -> None:
    """The registered op must reproduce the eager operation, forward and backward."""
    assert _bias_ops is not None, "bias custom ops failed to register"
    forward_fn, backward_fn = _bias_ops

    shape, size = (16, 32), 32
    op = _bias_op(size, dtype)
    x = torch.randn(*shape, device=_device, dtype=dtype, requires_grad=True)
    dy = torch.randn(*shape, device=_device, dtype=dtype)

    # Reference: the op as used today.
    y_ref = op(x)
    y_ref.backward(dy)
    dx_ref, db_ref = x.grad.clone(), op.bias.grad.clone()

    # Same computation through the custom ops.
    args = op.resolve_fwd_args(x.detach(), requires_grad=True, prev_op_grad_output_quantizer=None)
    y, saved, ctx_attrs = forward_fn(args)
    assert saved == (), "bias saves no tensors"
    dx, db = backward_fn(
        BiasBwdArgs(grad_output=dy, grad_input_quantizer=ctx_attrs["grad_input_quantizer"])
    )
    # A None grad input means "grad_output unchanged" -- a custom op may not
    # return one of its own inputs.
    if dx is None:
        dx = dy

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(dx, dx_ref)
    torch.testing.assert_close(db, db_ref)


@_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bias_custom_op_compiles_fullgraph(dtype) -> None:
    """Both halves must trace without a graph break.

    Run under ``no_grad``: the halves carry no autograd of their own (that is
    the point of ``register_op_halves``), so a caller is expected to wire them
    into its own ``autograd.Function``.
    """
    assert _bias_ops is not None, "bias custom ops failed to register"
    forward_fn, backward_fn = _bias_ops

    shape, size = (16, 32), 32
    op = _bias_op(size, dtype)
    x = torch.randn(*shape, device=_device, dtype=dtype)
    dy = torch.randn(*shape, device=_device, dtype=dtype)

    def fwd(x_):
        args = op.resolve_fwd_args(x_, requires_grad=True, prev_op_grad_output_quantizer=None)
        out, _saved, _attrs = forward_fn(args)
        return out

    def bwd(dy_):
        return backward_fn(BiasBwdArgs(grad_output=dy_, grad_input_quantizer=None))

    with torch.no_grad():
        torch.testing.assert_close(torch.compile(fwd, fullgraph=True)(x), fwd(x))
        compiled_grads = torch.compile(bwd, fullgraph=True)(dy)
        expected_grads = bwd(dy)
    for got, expected in zip(compiled_grads, expected_grads):
        if got is None or expected is None:
            assert got is expected is None
            continue
        torch.testing.assert_close(got, expected)
