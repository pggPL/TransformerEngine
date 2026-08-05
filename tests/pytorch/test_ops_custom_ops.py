# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Per-op custom ops for ``transformer_engine.pytorch.ops``.

An operation opts into ``torch.compile`` by declaring its two argument
containers and implementing four compute halves; ``BasicOperation`` registers
the custom ops and drives them. The checks here are the same for every such
operation and are driven by ``_OP_CASES`` -- adding an operation means adding
one entry.

Each operation is checked three ways:

* the data-free fake agrees with the real impl, slot for slot -- the compiled
  path slices a flat ``Tensor[]`` payload by what the fake said, so a
  disagreement is a silently misassembled tensor rather than an error;
* the registered op reproduces the eager operation;
* both halves trace under ``fullgraph=True``, with and without an FP8 output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch.dynamo import TensorSpec
from transformer_engine.pytorch.dynamo.custom_op import _spec_slot_count, _value_to_flat_tensors
from transformer_engine.pytorch.ops.op import OperationContext
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_device = "cuda"
_dtype = torch.bfloat16
_HIDDEN = 64


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    """Compile each case from scratch.

    The compiled helpers below are closures over one operation, so every case
    recompiles the same code object; without a reset the parametrized runs walk
    into Dynamo's recompilation limit and fall back to eager, which is an error
    under ``fullgraph=True``.
    """
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _fp8_quantizer() -> Any:
    """An FP8 quantizer, standing in for the next operation's input quantizer."""
    quantizer = Float8CurrentScalingQuantizer(DType.kFloat8E4M3, torch.device(_device))
    quantizer.set_usage(rowwise=True, columnwise=False)
    return quantizer


# --------------------------------------------------------------------------- #
# The operations under test
# --------------------------------------------------------------------------- #


def _build_bias():
    op = te.ops.Bias(_HIDDEN, device=_device, dtype=_dtype)
    with torch.no_grad():
        op.bias.copy_(torch.randn_like(op.bias))
    return op


@dataclass
class OpCase:
    """One operation and how to build it."""

    name: str
    build: Callable[[], Any]
    quantizes_output: bool = True
    num_grads: int = 1
    in_shape: Tuple[int, ...] = (16, _HIDDEN)


_OP_CASES = [
    OpCase(name="Bias", build=_build_bias, quantizes_output=False, num_grads=2),
    *(
        OpCase(name=cls.__name__, build=cls)
        for cls in (te.ops.GELU, te.ops.ReLU, te.ops.SiLU, te.ops.GEGLU, te.ops.ReGLU)
    ),
]
_CASE_IDS = [case.name for case in _OP_CASES]
_QUANTIZING = [case for case in _OP_CASES if case.quantizes_output]


def _make_input(case: OpCase, requires_grad: bool = False) -> torch.Tensor:
    return torch.randn(*case.in_shape, device=_device, dtype=_dtype, requires_grad=requires_grad)


def _resolve(op, x, *, fp8_output: bool):
    return op.resolve_fwd_args(
        x,
        requires_grad=True,
        prev_op_grad_output_quantizer=None,
        next_op_input_quantizer=_fp8_quantizer() if fp8_output else None,
    )


def _forward_through_ctx(op, x, *, fp8_output: bool):
    """Run the eager forward and hand back a context the backward can read."""
    ctx = OperationContext()
    ctx.requires_grad = True
    y = op.op_forward(
        ctx,
        x,
        prev_op_grad_output_quantizer=None,
        next_op_input_quantizer=_fp8_quantizer() if fp8_output else None,
    )
    ctx.saved_tensors = ctx.to_save
    return y, ctx


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

    The invariant the compiled path relies on is the flat ``Tensor[]`` slot
    layout, so that is what gets checked, using the framework's own helpers.
    Geometry is compared on top.
    """
    real_seq, spec_seq = _as_sequence(real), _as_sequence(specs)
    assert len(real_seq) == len(spec_seq), f"{what}: count differs"
    for i, (value, spec) in enumerate(zip(real_seq, spec_seq)):
        real_slots = len(_value_to_flat_tensors(value))
        fake_slots = _spec_slot_count(spec)
        assert real_slots == fake_slots, f"{what}[{i}]: {real_slots} slots vs fake {fake_slots}"
        assert _geometry(value) == _spec_geometry(spec), f"{what}[{i}]: geometry differs"


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@_cuda
@pytest.mark.parametrize("case", _OP_CASES, ids=_CASE_IDS)
@pytest.mark.parametrize("fp8_output", [False, True])
def test_op_fake_matches_real(case: OpCase, fp8_output: bool) -> None:
    """The fake must predict output geometry, including whether it is quantized."""
    op = case.build()
    cls = type(op)
    x = _make_input(case)
    args = _resolve(op, x, fp8_output=fp8_output)

    real_out, real_saved, real_attrs = cls.forward_compute(args)
    fake_out, fake_saved, fake_attrs = cls.forward_fake(args)
    assert_values_match_specs(real_out, fake_out, "forward output")
    assert_values_match_specs(real_saved, fake_saved, "saved tensor")
    assert set(real_attrs) == set(fake_attrs), "ctx_attrs keys disagree"

    _y, ctx = _forward_through_ctx(op, x, fp8_output=fp8_output)
    dy = torch.randn(*real_out.shape, device=_device, dtype=_dtype)
    bwd_args = op.resolve_bwd_args(ctx, dy)
    real_grads = _as_sequence(cls.backward_compute(bwd_args))
    fake_grads = _as_sequence(cls.backward_fake(bwd_args))
    assert len(real_grads) == len(fake_grads) == case.num_grads
    for i, (value, spec) in enumerate(zip(real_grads, fake_grads)):
        # One slot per gradient regardless of quantization, so only geometry matters.
        assert _geometry(value) == _spec_geometry(spec), f"gradient[{i}]: geometry differs"


@_cuda
@pytest.mark.parametrize("case", _OP_CASES, ids=_CASE_IDS)
def test_op_matches_eager(case: OpCase) -> None:
    """The registered op must reproduce the eager operation."""
    op = case.build()
    assert op.compile_ops is not None, f"{case.name}: custom ops failed to register"
    forward_fn, backward_fn = op.compile_ops

    x = _make_input(case, requires_grad=True)
    y_ref = op(x)
    dy = torch.randn_like(y_ref)
    y_ref.backward(dy)
    dx_ref = x.grad.clone()

    y, _saved, _attrs = forward_fn(_resolve(op, x.detach(), fp8_output=False))
    _y_eager, ctx = _forward_through_ctx(op, x.detach(), fp8_output=False)
    grads = backward_fn(op.resolve_bwd_args(ctx, dy))
    # A None gradient means "the incoming gradient, unchanged" -- a custom op may
    # not return one of its own inputs.
    dx = grads[0] if grads[0] is not None else dy

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(dx, dx_ref)


@_cuda
@pytest.mark.parametrize("case", _OP_CASES, ids=_CASE_IDS)
@pytest.mark.parametrize("fp8_output", [False, True])
def test_op_compiles_fullgraph(case: OpCase, fp8_output: bool) -> None:
    """Every operation's halves must trace under ``fullgraph=True``.

    Run under ``no_grad``: the halves carry no autograd of their own (that is the
    point of ``register_op_halves``), so a caller wires them into its own
    ``autograd.Function`` -- see ``test_ops_hop_poc.py``.
    """
    op = case.build()
    assert op.compile_ops is not None, f"{case.name}: custom ops failed to register"
    forward_fn, backward_fn = op.compile_ops
    x = _make_input(case)

    def fwd(x_):
        out, saved, _attrs = forward_fn(_resolve(op, x_, fp8_output=fp8_output))
        # An FP8 output crosses the boundary as its inner buffers and is rebuilt
        # on the far side; compare the buffers, since dequantize is not traceable.
        if isinstance(out, QuantizedTensorStorage):
            return (out._data, out._scale_inv, *saved)
        return (out, *saved)

    with torch.no_grad():
        expected = fwd(x)
        got = torch.compile(fwd, fullgraph=True)(x)
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b)

    y, ctx = _forward_through_ctx(op, x, fp8_output=fp8_output)
    dy = torch.randn(*y.shape, device=_device, dtype=_dtype)
    bwd_args = op.resolve_bwd_args(ctx, dy)

    def bwd(args):
        return backward_fn(args)

    with torch.no_grad():
        expected_grads = bwd(bwd_args)
        got_grads = torch.compile(bwd, fullgraph=True)(bwd_args)
    for a, b in zip(got_grads, expected_grads):
        if a is None or b is None:
            assert a is b is None
            continue
        torch.testing.assert_close(a, b)


@_cuda
@pytest.mark.parametrize("case", _QUANTIZING, ids=[c.name for c in _QUANTIZING])
def test_op_returns_fp8(case: OpCase) -> None:
    """With a next-operation quantizer the op must hand back an FP8 tensor.

    This is what matters for a pipeline: an operation quantizes its output with
    the *next* operation's input quantizer, so a quantized tensor crosses the
    custom-op boundary -- as its inner buffers, rebuilt on the far side -- rather
    than being dequantized at it.
    """
    op = case.build()
    forward_fn, _ = op.compile_ops
    y, _saved, _attrs = forward_fn(_resolve(op, _make_input(case), fp8_output=True))
    assert isinstance(y, QuantizedTensorStorage), f"expected a quantized output, got {type(y)}"
