# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused ``x * sigmoid(gate)`` with its backward."""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

__all__ = ["HAVE_TRITON", "sigmoid_gate"]

_BLOCK = 2048


if HAVE_TRITON:

    @triton.jit
    def _sigmoid(x):
        return 1.0 / (1.0 + tl.exp(-x))

    @triton.jit
    def sigmoid_gate_fwd_kernel(X, G, Y, numel, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + offs, (x * _sigmoid(g)).to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def sigmoid_gate_bwd_kernel(X, G, dY, dX, dG, numel, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dY + offs, mask=mask, other=0.0).to(tl.float32)
        sg = _sigmoid(g)
        tl.store(dX + offs, (dy * sg).to(dX.dtype.element_ty), mask=mask)
        tl.store(dG + offs, (dy * x * sg * (1.0 - sg)).to(dG.dtype.element_ty), mask=mask)

    class _SigmoidGate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, gate):
            x = x.contiguous()
            gate = gate.contiguous()
            y = torch.empty_like(x)
            grid = (triton.cdiv(x.numel(), _BLOCK),)
            sigmoid_gate_fwd_kernel[grid](x, gate, y, x.numel(), BLOCK=_BLOCK)
            ctx.save_for_backward(x, gate)
            return y

        @staticmethod
        def backward(ctx, dy):
            x, gate = ctx.saved_tensors
            dy = dy.contiguous()
            dx = torch.empty_like(x)
            dg = torch.empty_like(gate)
            grid = (triton.cdiv(x.numel(), _BLOCK),)
            sigmoid_gate_bwd_kernel[grid](x, gate, dy, dx, dg, x.numel(), BLOCK=_BLOCK)
            return dx, dg


def sigmoid_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``x * sigmoid(gate)`` computed in float32; ``x`` and ``gate`` have the same shape."""
    if (
        HAVE_TRITON
        and x.is_cuda
        and x.shape == gate.shape
        and x.dtype == gate.dtype
        and not torch.compiler.is_compiling()
    ):
        return _SigmoidGate.apply(x, gate)
    return (x.float() * torch.sigmoid(gate.float())).to(x.dtype)
