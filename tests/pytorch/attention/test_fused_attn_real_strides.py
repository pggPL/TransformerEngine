# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for real-stride plumbing (nvte_fused_attn_fwd/bwd_v2).

For dense (non-THD, non-paged) f16 layouts, the PyTorch extension passes
the real torch strides of Q/K/V (and dO) to the cuDNN fused-attention
graph instead of strides reconstructed from the NVTE_QKV_Layout enum.
Strided views into a packed QKV buffer then compute correctly even when
declared with the plain *separate* layout enum -- the enum no longer needs
to encode memory geometry -- and DotProductAttention no longer needs the
pointer-based (data_ptr/storage_offset) layout detection that graph-breaks
under torch.compile.
"""

import pytest
import torch

import transformer_engine.pytorch  # noqa: F401  (loads libtransformer_engine.so)
import transformer_engine_torch as tex
from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import (
    dot_product_attention as dpa_module,
)
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_bwd,
    fused_attn_fwd,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")

_BACKEND = tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen
_B, _S, _H, _D = 2, 128, 8, 64
_DTYPE = torch.bfloat16


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
def test_packed_bs3hd_views_declared_separate():
    """Headline: strided views into a packed [b,s,3,h,d] buffer, declared with the
    plain separate layout enum, are bit-exact vs the contiguous baseline because the
    real strides are passed to the cuDNN graph."""
    torch.manual_seed(0)
    qkv = torch.randn(_B, _S, 3, _H, _D, dtype=_DTYPE, device="cuda")
    d_o = torch.randn(_B, _S, _H, _D, dtype=_DTYPE, device="cuda")

    reference = _run(
        qkv[:, :, 0].contiguous(), qkv[:, :, 1].contiguous(), qkv[:, :, 2].contiguous(),
        d_o, "bshd_bshd_bshd", "bshd",
    )

    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    assert not q.is_contiguous()
    result = _run(q, k, v, d_o, "bshd_bshd_bshd", "bshd")
    _assert_bit_exact(result, reference)


@requires_backend
def test_packed_sbh3d_views_declared_separate():
    """Same as above for the sbh3d interleave (packing at dim -2, sbhd format)."""
    torch.manual_seed(0)
    qkv = torch.randn(_S, _B, _H, 3, _D, dtype=_DTYPE, device="cuda")
    d_o = torch.randn(_S, _B, _H, _D, dtype=_DTYPE, device="cuda")

    reference = _run(
        qkv[:, :, :, 0].contiguous(), qkv[:, :, :, 1].contiguous(),
        qkv[:, :, :, 2].contiguous(), d_o, "sbhd_sbhd_sbhd", "sbhd",
    )

    q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
    assert not q.is_contiguous()
    result = _run(q, k, v, d_o, "sbhd_sbhd_sbhd", "sbhd")
    _assert_bit_exact(result, reference)


@requires_backend
def test_packed_views_with_packed_enum_unchanged():
    """Regression: the historical path (packed views with the matching packed enum)
    still computes the same values as the separate-contiguous baseline. For packed
    views the real strides coincide with the enum-derived ones, so passing them is
    a no-op numerically."""
    torch.manual_seed(0)
    qkv = torch.randn(_B, _S, 3, _H, _D, dtype=_DTYPE, device="cuda")
    d_o = torch.randn(_B, _S, _H, _D, dtype=_DTYPE, device="cuda")
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    reference = _run(
        qkv[:, :, 0].contiguous(), qkv[:, :, 1].contiguous(), qkv[:, :, 2].contiguous(),
        d_o, "bshd_bshd_bshd", "bshd",
    )
    result = _run(q, k, v, d_o, "bs3hd", "bshd")
    _assert_bit_exact(result, reference)


# ---------------------------------------------------------------------------
# DotProductAttention end-to-end: DPA skips pointer-based qkv layout detection
# (data_ptr/storage_offset games) for dense f16 layouts and declares the
# format-derived separate layout instead.
# ---------------------------------------------------------------------------


def _force_backend(monkeypatch, backend):
    """Force a single attention backend via env and invalidate the selection cache."""
    flash, fused = {"flash": ("1", "0"), "fused": ("0", "1")}[backend]
    monkeypatch.setenv("NVTE_FLASH_ATTN", flash)
    monkeypatch.setenv("NVTE_FUSED_ATTN", fused)
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    if backend == "flash":
        # flash-attn bwd uses atomics unless deterministic; the fused (cuDNN)
        # backend gets disabled outright by the deterministic flag on some
        # devices, and its bwd is run-to-run deterministic anyway.
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    dpa_module._attention_backends["backend_selection_requires_update"] = True


def _make_dpa(qkv_format):
    return DotProductAttention(
        _H, _D, attention_dropout=0.0, qkv_format=qkv_format, attn_mask_type="no_mask"
    )


def _dpa_separate(qkv_format):
    """Fwd+bwd on three contiguous leaves; returns (out, dq, dk, dv)."""
    torch.manual_seed(0)
    shape = (_B, _S, _H, _D) if qkv_format == "bshd" else (_S, _B, _H, _D)
    q, k, v = [
        torch.randn(*shape, dtype=_DTYPE, device="cuda", requires_grad=True) for _ in range(3)
    ]
    dpa_module._attention_backends["backend_selection_requires_update"] = True
    out = _make_dpa(qkv_format)(q, k, v)
    out.backward(torch.ones_like(out))
    return out, q.grad, k.grad, v.grad


def _dpa_packed_views(qkv_format):
    """Fwd+bwd on strided views into one packed leaf; returns (out, dq, dk, dv)."""
    torch.manual_seed(0)
    shape = (_B, _S, _H, _D) if qkv_format == "bshd" else (_S, _B, _H, _D)
    parts = [torch.randn(*shape, dtype=_DTYPE, device="cuda") for _ in range(3)]
    qkv = torch.stack(parts, dim=2).requires_grad_()  # bs3hd / sb3hd packing
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    assert not q.is_contiguous()
    dpa_module._attention_backends["backend_selection_requires_update"] = True
    out = _make_dpa(qkv_format)(q, k, v)
    out.backward(torch.ones_like(out))
    return out, qkv.grad[:, :, 0], qkv.grad[:, :, 1], qkv.grad[:, :, 2]


@requires_backend
@pytest.mark.parametrize("fmt", ["bshd", "sbhd"])
def test_dpa_fused_packed_views(monkeypatch, fmt):
    """Fused backend: packed stack views (declared separate, real strides) are
    bit-exact vs contiguous separate q/k/v, fwd and grads."""
    _force_backend(monkeypatch, "fused")
    reference = _dpa_separate(fmt)
    result = _dpa_packed_views(fmt)
    _assert_bit_exact(result, reference)


def test_dpa_flash_packed_views(monkeypatch):
    """Flash backend smoke: flash consumes real strides natively, so packed views
    declared separate are bit-exact vs separate."""
    _force_backend(monkeypatch, "flash")
    try:
        reference = _dpa_separate("bshd")
    except Exception as exc:
        pytest.skip(f"flash attention backend not available: {exc}")
    result = _dpa_packed_views("bshd")
    _assert_bit_exact(result, reference)


@requires_backend
def test_dpa_layout_detection_skipped_dense_kept_thd(monkeypatch):
    """get_qkv_layout is not called for dense bshd, but still runs for thd
    (ragged layouts ignore strides in C++, so detection stays)."""
    _force_backend(monkeypatch, "fused")

    calls = []
    orig = dpa_utils.get_qkv_layout

    def counting(*args, **kwargs):
        calls.append(kwargs.get("qkv_format"))
        return orig(*args, **kwargs)

    monkeypatch.setattr(dpa_utils, "get_qkv_layout", counting)

    _dpa_separate("bshd")
    assert not calls, "get_qkv_layout should be skipped for dense bshd"

    # thd: full sequences, padding mask
    torch.manual_seed(0)
    t = _B * _S
    q, k, v = [torch.randn(t, _H, _D, dtype=_DTYPE, device="cuda") for _ in range(3)]
    cu = _cu_seqlens()
    dpa_module._attention_backends["backend_selection_requires_update"] = True
    dpa = DotProductAttention(
        _H, _D, attention_dropout=0.0, qkv_format="thd", attn_mask_type="padding"
    )
    try:
        dpa(q, k, v, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=_S, max_seqlen_kv=_S)
    except ValueError:
        # No thd-capable backend on this device; the layout step (the subject
        # of this test) runs before backend dispatch, so the assertion below
        # still holds.
        pass
    assert calls == ["thd"], "get_qkv_layout must still run for thd"


@requires_backend
def test_dpa_noncontiguous_head_dim_normalized(monkeypatch):
    """Inputs with stride(-1) != 1 are normalized with .contiguous() in the bypass
    path (mirrors the old detection's contiguous-retry) and stay bit-exact."""
    _force_backend(monkeypatch, "fused")
    reference = _dpa_separate("bshd")

    torch.manual_seed(0)
    q, k, v = [
        torch.randn(_B, _S, _H, _D, dtype=_DTYPE, device="cuda", requires_grad=True)
        for _ in range(3)
    ]
    # transpose(-1, -2) of a transposed copy: same values, stride(-1) != 1
    q_t = q.detach().transpose(2, 3).contiguous().transpose(2, 3).requires_grad_()
    k_t = k.detach().transpose(2, 3).contiguous().transpose(2, 3).requires_grad_()
    v_t = v.detach().transpose(2, 3).contiguous().transpose(2, 3).requires_grad_()
    assert q_t.stride(-1) != 1
    dpa_module._attention_backends["backend_selection_requires_update"] = True
    out = _make_dpa("bshd")(q_t, k_t, v_t)
    out.backward(torch.ones_like(out))
    _assert_bit_exact((out, q_t.grad, k_t.grad, v_t.grad), reference)


def _compiled_graph_breaks():
    """Compile DPA (after an eager warm-up that caches backend selection) and
    return dynamo's graph_break counters for one fwd+bwd."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch.manual_seed(0)
    q, k, v = [
        torch.randn(_B, _S, _H, _D, dtype=_DTYPE, device="cuda", requires_grad=True)
        for _ in range(3)
    ]
    dpa = _make_dpa("bshd")
    dpa_module._attention_backends["backend_selection_requires_update"] = True
    dpa(q, k, v)  # eager warm-up: backend selection happens outside dynamo
    out = torch.compile(dpa)(q, k, v)
    out.backward(torch.ones_like(out))
    breaks = dict(torch._dynamo.utils.counters["graph_break"])
    torch._dynamo.reset()
    return breaks


def _pointer_breaks(breaks):
    return {
        reason: count
        for reason, count in breaks.items()
        if "data_ptr" in reason or "UntypedStorage" in reason
    }


@requires_backend
def test_dpa_torch_compile_no_data_ptr_graph_breaks(monkeypatch):
    """Compiling DPA no longer graph-breaks on data_ptr/UntypedStorage: the
    pointer-based layout detection is bypassed for dense layouts. Negative
    control: forcing the detection back on reintroduces those breaks."""
    _force_backend(monkeypatch, "fused")

    # negative control: force pointer-based detection, expect data_ptr breaks
    monkeypatch.setattr(dpa_module, "_skip_pointer_layout_detection", lambda *a: False)
    baseline = _compiled_graph_breaks()
    assert _pointer_breaks(baseline), (
        f"expected data_ptr/UntypedStorage breaks with detection forced on: {baseline}"
    )
    monkeypatch.undo()

    _force_backend(monkeypatch, "fused")
    breaks = _compiled_graph_breaks()
    assert not _pointer_breaks(breaks), (
        f"data_ptr/UntypedStorage graph breaks should be gone: {_pointer_breaks(breaks)}"
    )
