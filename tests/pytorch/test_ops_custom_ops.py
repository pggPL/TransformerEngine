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
from transformer_engine.pytorch.dynamo.custom_op import _spec_slot_count, _value_to_flat_tensors
from transformer_engine.pytorch.ops.basic.activation import ActivationBwdArgs
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


def _geometry(value: Any) -> Optional[Tuple]:
    """Shape / dtype / quantizer type of a real value, or ``None`` for a sentinel."""
    if value is None:
        return None
    quantizer = getattr(value, "_quantizer", None)
    if not isinstance(value, QuantizedTensorStorage):
        quantizer = None
    return (tuple(value.shape), value.dtype, type(quantizer) if quantizer else None)


def _spec_geometry(spec: Optional[TensorSpec]) -> Optional[Tuple]:
    if spec is None:
        return None
    return (tuple(spec.shape), spec.dtype, type(spec.quantizer) if spec.quantizer else None)


def _as_sequence(values: Any) -> Tuple:
    if values is None:
        return ()
    if not isinstance(values, (tuple, list)):
        return (values,)
    return tuple(values)


def assert_values_match_specs(real: Any, specs: Any, what: str) -> None:
    """Require that the fake describes what the real impl produced.

    The invariant the compiled path actually relies on is the flat ``Tensor[]``
    slot layout, so that is what gets checked, using the framework's own
    helpers. Geometry is compared on top, but only where the real impl produced
    a value: ``None`` is a legal payload meaning "an input, unchanged" (a custom
    op may not return one of its own inputs), and it occupies the same single
    slot as an unquantized tensor.
    """
    real_seq, spec_seq = _as_sequence(real), _as_sequence(specs)
    assert len(real_seq) == len(spec_seq), f"{what}: count differs"
    for i, (value, spec) in enumerate(zip(real_seq, spec_seq)):
        real_slots = len(_value_to_flat_tensors(value))
        fake_slots = _spec_slot_count(spec)
        assert real_slots == fake_slots, f"{what}[{i}]: {real_slots} slots vs fake {fake_slots}"
        if value is not None:
            assert _geometry(value) == _spec_geometry(spec), f"{what}[{i}]: geometry differs"


def assert_fwd_fake_matches_real(args: Any, impl, fake_impl) -> None:
    """Run a forward impl and its fake on the same args; require agreement."""
    real_out, real_saved, real_attrs = impl(args)
    fake_out, fake_saved, fake_attrs = fake_impl(args)

    assert_values_match_specs(real_out, fake_out, "forward output")
    assert_values_match_specs(real_saved, fake_saved, "saved tensor")
    assert set(real_attrs) == set(fake_attrs), "ctx_attrs keys disagree"


def assert_bwd_fake_matches_real(args: Any, impl, fake_impl) -> None:
    """Run a backward impl and its fake on the same args; require agreement.

    Gradients are not slot-counted: the backward payload holds exactly one slot
    per gradient regardless of quantization, so only geometry is compared.
    """
    real_seq, spec_seq = _as_sequence(impl(args)), _as_sequence(fake_impl(args))
    assert len(real_seq) == len(spec_seq), "gradient count differs"
    for i, (value, spec) in enumerate(zip(real_seq, spec_seq)):
        assert _geometry(value) == _spec_geometry(spec), f"gradient[{i}]: geometry differs"


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


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #


def _fp8_quantizer(dtype: torch.dtype) -> Any:
    """A current-scaling FP8 quantizer, as a next op would hand down."""
    from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer
    from transformer_engine.pytorch.constants import DType

    del dtype
    quantizer = Float8CurrentScalingQuantizer(DType.kFloat8E4M3, torch.device(_device))
    quantizer.set_usage(rowwise=True, columnwise=False)
    return quantizer


_ACTIVATIONS = [te.ops.GELU, te.ops.ReLU, te.ops.SiLU, te.ops.GEGLU, te.ops.ReGLU]


@_cuda
@pytest.mark.parametrize("cls", _ACTIVATIONS)
@pytest.mark.parametrize("quantize_output", [False, True])
def test_activation_fake_conformance(cls, quantize_output) -> None:
    """The fake must predict output geometry, including whether it is quantized."""
    dtype = torch.bfloat16
    op = cls()
    x = torch.randn(16, 64, device=_device, dtype=dtype)
    args = op.resolve_fwd_args(
        x,
        requires_grad=True,
        prev_op_grad_output_quantizer=None,
        next_op_input_quantizer=_fp8_quantizer(dtype) if quantize_output else None,
    )
    assert_fwd_fake_matches_real(args, op._impls.forward, op._impls.forward_fake)

    out, saved, ctx_attrs = op._impls.forward(args)
    dy = torch.randn_like(out if not quantize_output else out.dequantize())
    bwd_args = ActivationBwdArgs(
        grad_output=dy,
        saved_input=x if saved[0] is None else saved[0],
        dtype=ctx_attrs["dtype"],
        grad_input_quantizer=None,
    )
    assert_bwd_fake_matches_real(bwd_args, op._impls.backward, op._impls.backward_fake)


@_cuda
@pytest.mark.parametrize("cls", _ACTIVATIONS)
def test_activation_custom_op_returns_fp8(cls) -> None:
    """With a next-op quantizer the registered op must hand back an FP8 tensor.

    This is the case that matters for a pipeline: an operation quantizes its
    output with the *next* operation's input quantizer, so a quantized tensor
    crosses the custom-op boundary rather than being dequantized at it.
    """
    assert cls._impls.ops is not None, f"{cls.__name__} custom ops failed to register"
    forward_fn, _ = cls._impls.ops

    dtype = torch.bfloat16
    op = cls()
    x = torch.randn(16, 64, device=_device, dtype=dtype)
    args = op.resolve_fwd_args(
        x,
        requires_grad=True,
        prev_op_grad_output_quantizer=None,
        next_op_input_quantizer=_fp8_quantizer(dtype),
    )
    y, saved, _ctx_attrs = forward_fn(args)

    assert isinstance(y, QuantizedTensorStorage), f"expected a quantized output, got {type(y)}"
    y_ref, _saved_ref, _ = op._impls.forward(args)
    torch.testing.assert_close(y.dequantize(), y_ref.dequantize())
    assert len(saved) == 1


@_cuda
@pytest.mark.parametrize("cls", _ACTIVATIONS)
def test_activation_custom_op_matches_eager(cls) -> None:
    dtype = torch.bfloat16
    op = cls()
    forward_fn, backward_fn = cls._impls.ops

    x = torch.randn(16, 64, device=_device, dtype=dtype, requires_grad=True)
    y_ref = op(x)
    dy = torch.randn_like(y_ref)
    y_ref.backward(dy)
    dx_ref = x.grad.clone()

    args = op.resolve_fwd_args(
        x.detach(),
        requires_grad=True,
        prev_op_grad_output_quantizer=None,
        next_op_input_quantizer=None,
    )
    y, saved, ctx_attrs = forward_fn(args)
    # None means "the input, unchanged" -- a custom op may not return its own input.
    saved_input = x.detach() if saved[0] is None else saved[0]
    (dx,) = backward_fn(
        ActivationBwdArgs(
            grad_output=dy,
            saved_input=saved_input,
            dtype=ctx_attrs["dtype"],
            grad_input_quantizer=ctx_attrs["prev_op_grad_output_quantizer"],
        )
    )

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(dx, dx_ref)


@_cuda
@pytest.mark.parametrize("cls", [te.ops.GELU, te.ops.GEGLU])
def test_activation_custom_op_compiles_fullgraph(cls) -> None:
    dtype = torch.bfloat16
    op = cls()
    forward_fn, _ = cls._impls.ops
    x = torch.randn(16, 64, device=_device, dtype=dtype)

    def fwd(x_):
        args = op.resolve_fwd_args(
            x_,
            requires_grad=False,
            prev_op_grad_output_quantizer=None,
            next_op_input_quantizer=None,
        )
        out, _saved, _attrs = forward_fn(args)
        return out

    with torch.no_grad():
        torch.testing.assert_close(torch.compile(fwd, fullgraph=True)(x), fwd(x))
