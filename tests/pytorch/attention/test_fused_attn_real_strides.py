# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for NVTE_FUSED_ATTN_REAL_STRIDES (nvte_fused_attn_fwd/bwd_v2).

With the flag on, the PyTorch extension passes the real torch strides of
Q/K/V (and dO) to the cuDNN fused-attention graph instead of strides
reconstructed from the NVTE_QKV_Layout enum. Strided views into a packed
QKV buffer then compute correctly even when declared with the plain
*separate* layout enum -- the enum no longer needs to encode memory
geometry.
"""

import os

import pytest
import torch

import transformer_engine.pytorch  # noqa: F401  (loads libtransformer_engine.so)
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_bwd,
    fused_attn_fwd,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")

_BACKEND = tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen
_B, _S, _H, _D = 2, 128, 8, 64
_DTYPE = torch.bfloat16


@pytest.fixture
def real_strides_flag(monkeypatch):
    def set_flag(on: bool):
        if on:
            monkeypatch.setenv("NVTE_FUSED_ATTN_REAL_STRIDES", "1")
        else:
            monkeypatch.delenv("NVTE_FUSED_ATTN_REAL_STRIDES", raising=False)

    yield set_flag
    monkeypatch.delenv("NVTE_FUSED_ATTN_REAL_STRIDES", raising=False)


def _cu_seqlens():
    return torch.arange(0, (_B + 1) * _S, _S, dtype=torch.int32, device="cuda")


def _run(q, k, v, d_o, qkv_layout, fmt):
    """One fwd+bwd through the F16 arbitrary-seqlen backend; returns (out, dq, dk, dv)."""
    cu = _cu_seqlens()
    out, aux = fused_attn_fwd(
        True, _S, _S, cu, cu, q, k, v, _DTYPE, _BACKEND,
        dropout=0.0, qkv_layout=qkv_layout, o_format=fmt,
        attn_bias_type="no_bias", attn_mask_type="no_mask",
    )
    dq, dk, dv, _, _ = fused_attn_bwd(
        _S, _S, cu, cu, q, k, v, out, d_o, _DTYPE, aux, _BACKEND,
        dropout=0.0, qkv_layout=qkv_layout, o_format=fmt, do_format=fmt,
        dqkv_layout=qkv_layout, attn_bias_type="no_bias", attn_mask_type="no_mask",
        deterministic=True,
    )
    return out, dq, dk, dv


def _assert_bit_exact(result, reference):
    for name, x, y in zip(("out", "dq", "dk", "dv"), result, reference):
        assert torch.equal(x.contiguous(), y.contiguous()), f"{name} differs"


def _backend_supported():
    try:
        q = torch.randn(_B, _S, _H, _D, dtype=_DTYPE, device="cuda")
        _run(q, q.clone(), q.clone(), q.clone(), "bshd_bshd_bshd", "bshd")
        return True
    except Exception:
        return False


requires_backend = pytest.mark.skipif(
    not (torch.cuda.is_available() and _backend_supported()),
    reason="F16_arbitrary_seqlen fused attention backend is not supported on this device",
)


@requires_backend
@pytest.mark.parametrize("fmt", ["bshd", "sbhd"])
def test_contiguous_separate_matches_enum_path(real_strides_flag, fmt):
    """Contiguous separate q/k/v: real strides == enum strides, so flag on == flag off."""
    torch.manual_seed(0)
    shape = (_B, _S, _H, _D) if fmt == "bshd" else (_S, _B, _H, _D)
    q = torch.randn(*shape, dtype=_DTYPE, device="cuda")
    k, v, d_o = q.clone(), q.clone(), torch.randn(*shape, dtype=_DTYPE, device="cuda")
    layout = f"{fmt}_{fmt}_{fmt}"

    real_strides_flag(False)
    reference = _run(q, k, v, d_o, layout, fmt)
    real_strides_flag(True)
    result = _run(q, k, v, d_o, layout, fmt)
    _assert_bit_exact(result, reference)


@requires_backend
def test_packed_bs3hd_views_declared_separate(real_strides_flag):
    """Headline: strided views into a packed [b,s,3,h,d] buffer, declared with the
    plain separate layout enum, are bit-exact vs the contiguous baseline when real
    strides are passed."""
    torch.manual_seed(0)
    qkv = torch.randn(_B, _S, 3, _H, _D, dtype=_DTYPE, device="cuda")
    d_o = torch.randn(_B, _S, _H, _D, dtype=_DTYPE, device="cuda")

    real_strides_flag(False)
    reference = _run(
        qkv[:, :, 0].contiguous(), qkv[:, :, 1].contiguous(), qkv[:, :, 2].contiguous(),
        d_o, "bshd_bshd_bshd", "bshd",
    )

    real_strides_flag(True)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    assert not q.is_contiguous()
    result = _run(q, k, v, d_o, "bshd_bshd_bshd", "bshd")
    _assert_bit_exact(result, reference)


@requires_backend
def test_packed_sbh3d_views_declared_separate(real_strides_flag):
    """Same as above for the sbh3d interleave (packing at dim -2, sbhd format)."""
    torch.manual_seed(0)
    qkv = torch.randn(_S, _B, _H, 3, _D, dtype=_DTYPE, device="cuda")
    d_o = torch.randn(_S, _B, _H, _D, dtype=_DTYPE, device="cuda")

    real_strides_flag(False)
    reference = _run(
        qkv[:, :, :, 0].contiguous(), qkv[:, :, :, 1].contiguous(),
        qkv[:, :, :, 2].contiguous(), d_o, "sbhd_sbhd_sbhd", "sbhd",
    )

    real_strides_flag(True)
    q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
    assert not q.is_contiguous()
    result = _run(q, k, v, d_o, "sbhd_sbhd_sbhd", "sbhd")
    _assert_bit_exact(result, reference)


@requires_backend
def test_packed_views_with_packed_enum_unchanged(real_strides_flag):
    """Regression: the historical path (packed views with the matching packed enum)
    is unchanged with the flag on."""
    torch.manual_seed(0)
    qkv = torch.randn(_B, _S, 3, _H, _D, dtype=_DTYPE, device="cuda")
    d_o = torch.randn(_B, _S, _H, _D, dtype=_DTYPE, device="cuda")
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    real_strides_flag(False)
    reference = _run(q, k, v, d_o, "bs3hd", "bshd")
    real_strides_flag(True)
    result = _run(q, k, v, d_o, "bs3hd", "bshd")
    _assert_bit_exact(result, reference)
