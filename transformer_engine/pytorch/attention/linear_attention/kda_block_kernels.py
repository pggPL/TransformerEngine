# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused elementwise kernels around the KDA recurrence.

Two Triton kernels replace the PyTorch glue of a Kimi Delta Attention block:

* ``causal_conv_silu_split``: SiLU(causal depthwise short conv) over the fused
  qkv projection, written directly as three contiguous ``bshd`` tensors.
* ``gated_rmsnorm``: per-head RMSNorm of the recurrence output times
  ``sigmoid(gate)``, reading ``bshd`` and writing the layer's layout.

Both compute in float32 and are exact replacements of the PyTorch fallbacks in
:mod:`transformer_engine.pytorch.models.kimi_k3.kimi_linear_attention`.
"""

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

__all__ = ["HAVE_TRITON", "causal_conv_silu_split", "gated_rmsnorm"]


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


if HAVE_TRITON:

    @triton.jit
    def _sigmoid(x):
        return 1.0 / (1.0 + tl.exp(-x))

    @triton.jit
    def _load_rows(X, rows, cols, stride_s, S, mask_c):
        mask = (rows[:, None] >= 0) & (rows[:, None] < S) & mask_c[None, :]
        off = rows[:, None].to(tl.int64) * stride_s + cols[None, :]
        return tl.load(X + off, mask=mask, other=0.0).to(tl.float32)

    @triton.jit
    def _conv_pre(X, W, rows, cols, stride_s, S, mask_c, KSIZE: tl.constexpr):
        """Causal depthwise conv pre-activation at ``rows`` (float32)."""
        w = tl.load(W + cols * KSIZE, mask=mask_c, other=0.0)
        acc = w[None, :] * _load_rows(X, rows - (KSIZE - 1), cols, stride_s, S, mask_c)
        for k in tl.static_range(1, KSIZE):
            w = tl.load(W + cols * KSIZE + k, mask=mask_c, other=0.0)
            acc += w[None, :] * _load_rows(X, rows - (KSIZE - 1) + k, cols, stride_s, S, mask_c)
        return acc

    @triton.jit
    def conv_silu_split_fwd_kernel(
        X,
        W,
        Q,
        K,
        V,
        S,
        C: tl.constexpr,
        stride_x_s,
        stride_x_b,
        KSIZE: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_c = tl.program_id(2)
        part = pid_c // (C // BLOCK_C)
        rows = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = cols < 3 * C
        x_ptr = X + pid_b.to(tl.int64) * stride_x_b
        pre = _conv_pre(x_ptr, W, rows, cols, stride_x_s, S, mask_c, KSIZE)
        y = pre * _sigmoid(pre)
        out_off = (pid_b * S + rows)[:, None].to(tl.int64) * C + (cols - part * C)[None, :]
        mask = (rows[:, None] < S) & mask_c[None, :]
        if part == 0:
            tl.store(Q + out_off, y.to(Q.dtype.element_ty), mask=mask)
        elif part == 1:
            tl.store(K + out_off, y.to(K.dtype.element_ty), mask=mask)
        else:
            tl.store(V + out_off, y.to(V.dtype.element_ty), mask=mask)

    @triton.jit
    def _load_dy(dQ, dK, dV, part, off, mask):
        if part == 0:
            dy = tl.load(dQ + off, mask=mask, other=0.0)
        elif part == 1:
            dy = tl.load(dK + off, mask=mask, other=0.0)
        else:
            dy = tl.load(dV + off, mask=mask, other=0.0)
        return dy.to(tl.float32)

    @triton.jit
    def conv_silu_split_bwd_kernel(
        X,
        W,
        dQ,
        dK,
        dV,
        dX,
        dW_partial,
        S,
        C: tl.constexpr,
        stride_x_s,
        stride_x_b,
        KSIZE: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_c = tl.program_id(2)
        part = pid_c // (C // BLOCK_C)
        rows = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = cols < 3 * C
        x_ptr = X + pid_b.to(tl.int64) * stride_x_b
        local_cols = cols - part * C

        dx = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
        dpre0 = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(KSIZE):
            rows_j = rows + j
            pre = _conv_pre(x_ptr, W, rows_j, cols, stride_x_s, S, mask_c, KSIZE)
            sig = _sigmoid(pre)
            mask_j = (rows_j[:, None] < S) & mask_c[None, :]
            dy_off = (pid_b * S + rows_j)[:, None].to(tl.int64) * C + local_cols[None, :]
            dy = _load_dy(dQ, dK, dV, part, dy_off, mask_j)
            dpre = dy * sig * (1.0 + pre * (1.0 - sig))
            w = tl.load(W + cols * KSIZE + (KSIZE - 1 - j), mask=mask_c, other=0.0)
            dx += w[None, :] * dpre
            if j == 0:
                dpre0 = dpre

        mask = (rows[:, None] < S) & mask_c[None, :]
        dx_off = pid_b.to(tl.int64) * stride_x_b + rows[:, None].to(tl.int64) * stride_x_s
        tl.store(dX + dx_off + cols[None, :], dx.to(dX.dtype.element_ty), mask=mask)
        n_blocks_s = tl.num_programs(0)
        partial_row = pid_b * n_blocks_s + pid_s
        for k in tl.static_range(KSIZE):
            xk = _load_rows(x_ptr, rows - (KSIZE - 1) + k, cols, stride_x_s, S, mask_c)
            dw = tl.sum(dpre0 * xk, axis=0)
            dw_off = (partial_row.to(tl.int64) * (3 * C) + cols) * KSIZE + k
            tl.store(dW_partial + dw_off, dw, mask=mask_c)

    @triton.jit
    def gated_rmsnorm_fwd_kernel(
        O,
        G,
        W,
        Y,
        S,
        B,
        H: tl.constexpr,
        eps,
        stride_g_s,
        stride_g_b,
        stride_y_s,
        stride_y_b,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        h = tl.program_id(1)
        t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = t < S * B
        b = t // S
        s = t % S
        d = tl.arange(0, D)
        mask = mask_t[:, None] & (d[None, :] < D)
        o_off = (t[:, None].to(tl.int64) * H + h) * D + d[None, :]
        o = tl.load(O + o_off, mask=mask, other=0.0).to(tl.float32)
        g_off = s[:, None].to(tl.int64) * stride_g_s + b[:, None].to(tl.int64) * stride_g_b
        g = tl.load(G + g_off + h * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + d).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(o * o, axis=1) / D + eps)
        y = o * rstd[:, None] * w[None, :] * _sigmoid(g)
        y_off = s[:, None].to(tl.int64) * stride_y_s + b[:, None].to(tl.int64) * stride_y_b
        y_off = y_off + h * D + d[None, :]
        tl.store(Y + y_off, y.to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def gated_rmsnorm_bwd_kernel(
        O,
        G,
        W,
        dY,
        dO,
        dG,
        dW_partial,
        S,
        B,
        H: tl.constexpr,
        eps,
        stride_g_s,
        stride_g_b,
        stride_y_s,
        stride_y_b,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        h = tl.program_id(1)
        t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = t < S * B
        b = t // S
        s = t % S
        d = tl.arange(0, D)
        mask = mask_t[:, None] & (d[None, :] < D)
        o_off = (t[:, None].to(tl.int64) * H + h) * D + d[None, :]
        o = tl.load(O + o_off, mask=mask, other=0.0).to(tl.float32)
        g_off = s[:, None].to(tl.int64) * stride_g_s + b[:, None].to(tl.int64) * stride_g_b
        g_off = g_off + h * D + d[None, :]
        g = tl.load(G + g_off, mask=mask, other=0.0).to(tl.float32)
        y_off = s[:, None].to(tl.int64) * stride_y_s + b[:, None].to(tl.int64) * stride_y_b
        y_off = y_off + h * D + d[None, :]
        dy = tl.load(dY + y_off, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + d).to(tl.float32)

        rstd = 1.0 / tl.sqrt(tl.sum(o * o, axis=1) / D + eps)
        n = o * rstd[:, None]
        sg = _sigmoid(g)
        dg = dy * n * w[None, :] * sg * (1.0 - sg)
        dn = dy * w[None, :] * sg
        dw = tl.sum(dy * sg * n, axis=0)
        do = rstd[:, None] * (dn - n * (tl.sum(dn * n, axis=1) / D)[:, None])

        tl.store(dO + o_off, do.to(dO.dtype.element_ty), mask=mask)
        tl.store(dG + y_off, dg.to(dG.dtype.element_ty), mask=mask)
        tl.store(dW_partial + (pid_t.to(tl.int64) * H + h) * D + d, dw)

    class _CausalConvSiLUSplit(torch.autograd.Function):
        """x [s, b, 3C] or [b, s, 3C] -> SiLU(conv) as q, k, v, each [b, s, C]."""

        @staticmethod
        def forward(ctx, x, weight, seq_dim):
            S = x.shape[seq_dim]
            B = x.shape[1 - seq_dim]
            C = x.shape[-1] // 3
            ksize = weight.shape[-1]
            weight = weight.contiguous()
            q, k, v = (torch.empty(B, S, C, dtype=x.dtype, device=x.device) for _ in range(3))
            block_c = _conv_block_c(C)
            grid = (triton.cdiv(S, _CONV_BLOCK_S), B, 3 * C // block_c)
            conv_silu_split_fwd_kernel[grid](
                x,
                weight,
                q,
                k,
                v,
                S,
                C,
                x.stride(seq_dim),
                x.stride(1 - seq_dim),
                KSIZE=ksize,
                BLOCK_S=_CONV_BLOCK_S,
                BLOCK_C=block_c,
            )
            ctx.save_for_backward(x, weight)
            ctx.seq_dim = seq_dim
            return q, k, v

        @staticmethod
        def backward(ctx, dq, dk, dv):
            x, weight = ctx.saved_tensors
            seq_dim = ctx.seq_dim
            S = x.shape[seq_dim]
            B = x.shape[1 - seq_dim]
            C = x.shape[-1] // 3
            ksize = weight.shape[-1]
            dq, dk, dv = (t.contiguous() for t in (dq, dk, dv))
            dx = torch.empty_like(x)
            block_c = _conv_block_c(C)
            n_blocks_s = triton.cdiv(S, _CONV_BLOCK_S)
            dw_partial = torch.empty(
                B * n_blocks_s, 3 * C, ksize, dtype=torch.float32, device=x.device
            )
            grid = (n_blocks_s, B, 3 * C // block_c)
            conv_silu_split_bwd_kernel[grid](
                x,
                weight,
                dq,
                dk,
                dv,
                dx,
                dw_partial,
                S,
                C,
                x.stride(seq_dim),
                x.stride(1 - seq_dim),
                KSIZE=ksize,
                BLOCK_S=_CONV_BLOCK_S,
                BLOCK_C=block_c,
            )
            return dx, dw_partial.sum(dim=0), None

    class _GatedRMSNorm(torch.autograd.Function):
        """o [b, s, H*D] (bshd) -> RMSNorm_D(o) * w * sigmoid(g), in g's layout."""

        @staticmethod
        def forward(ctx, o, g, weight, eps, seq_dim, num_heads):
            o = o.contiguous()
            weight = weight.contiguous()
            S, B = o.shape[1], o.shape[0]
            D = o.shape[-1] // num_heads
            y = torch.empty_like(g)
            grid = (triton.cdiv(S * B, _NORM_BLOCK_T), num_heads)
            gated_rmsnorm_fwd_kernel[grid](
                o,
                g,
                weight,
                y,
                S,
                B,
                num_heads,
                eps,
                g.stride(seq_dim),
                g.stride(1 - seq_dim),
                y.stride(seq_dim),
                y.stride(1 - seq_dim),
                D=D,
                BLOCK_T=_NORM_BLOCK_T,
            )
            ctx.save_for_backward(o, g, weight)
            ctx.args = (eps, seq_dim, num_heads)
            return y

        @staticmethod
        def backward(ctx, dy):
            o, g, weight = ctx.saved_tensors
            eps, seq_dim, num_heads = ctx.args
            S, B = o.shape[1], o.shape[0]
            D = o.shape[-1] // num_heads
            dy = dy.contiguous()
            do = torch.empty_like(o)
            dg = torch.empty_like(g, memory_format=torch.contiguous_format)
            n_blocks_t = triton.cdiv(S * B, _NORM_BLOCK_T)
            dw_partial = torch.empty(n_blocks_t * num_heads, D, dtype=torch.float32, device=o.device)
            gated_rmsnorm_bwd_kernel[(n_blocks_t, num_heads)](
                o,
                g,
                weight,
                dy,
                do,
                dg,
                dw_partial,
                S,
                B,
                num_heads,
                eps,
                g.stride(seq_dim),
                g.stride(1 - seq_dim),
                dy.stride(seq_dim),
                dy.stride(1 - seq_dim),
                D=D,
                BLOCK_T=_NORM_BLOCK_T,
            )
            return do, dg, dw_partial.sum(dim=0), None, None, None


_CONV_BLOCK_S = 64
_NORM_BLOCK_T = 32


def _conv_block_c(channels: int) -> int:
    for block in (128, 64, 32, 16):
        if channels % block == 0:
            return block
    return 0


def conv_silu_split_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the Triton conv kernel handles these tensors."""
    return (
        HAVE_TRITON
        and x.is_cuda
        and x.dim() == 3
        and x.is_contiguous()
        and x.shape[-1] % 3 == 0
        and _conv_block_c(x.shape[-1] // 3) > 0
        and weight.dim() == 2
        and weight.shape[0] == x.shape[-1]
        and not torch.compiler.is_compiling()
    )


def gated_rmsnorm_supported(o: torch.Tensor, g: torch.Tensor, num_heads: int) -> bool:
    """Whether the Triton gated RMSNorm kernel handles these tensors."""
    return (
        HAVE_TRITON
        and o.is_cuda
        and o.dim() == 3
        and g.dim() == 3
        and g.is_contiguous()
        and o.shape[-1] % num_heads == 0
        and _is_power_of_two(o.shape[-1] // num_heads)
        and not torch.compiler.is_compiling()
    )


def causal_conv_silu_split(
    x: torch.Tensor, weight: torch.Tensor, seq_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SiLU(causal depthwise conv) of ``x`` split into three ``[b, s, C]`` tensors.

    ``x`` is ``[s, b, 3C]`` (``seq_dim=0``) or ``[b, s, 3C]`` (``seq_dim=1``) with a
    contiguous last dimension; ``weight`` is float32 ``[3C, kernel_size]``.
    """
    return _CausalConvSiLUSplit.apply(x, weight, seq_dim)


def gated_rmsnorm(
    o: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float, seq_dim: int, num_heads: int
) -> torch.Tensor:
    """``RMSNorm(o) * weight * sigmoid(g)`` per head, in float32.

    ``o`` is ``[b, s, H*D]``; ``g`` is ``[s, b, H*D]`` (``seq_dim=0``) or ``[b, s, H*D]``
    (``seq_dim=1``) and sets the output layout; ``weight`` is float32 ``[D]``.
    """
    return _GatedRMSNorm.apply(o, g, weight, eps, seq_dim, num_heads)
