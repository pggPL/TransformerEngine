# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile coverage for DotProductAttention (FlashAttention backend).

Goal: ``DotProductAttention`` must trace cleanly through
``torch.compile(fullgraph=True)`` on the FlashAttention backend. ``fullgraph=True``
hard-errors on *any* graph break, so this suite is the target we drive to green
while removing the constructs Dynamo cannot trace.

We restrict ourselves to the FlashAttention backend, whose underlying
``flash_attn`` kernels are already registered as custom ops that Dynamo can
capture. The suite is parametrized over three axes -- all of which still route to
FlashAttention:

* model config: plain self-attention, causal, GQA/MQA, cross-attention,
  sliding window, larger head dim, and padding / padding-causal masks;
* qkv memory format: ``bshd`` and ``sbhd`` for every config, plus ``thd``
  (packed, variable-length) for the padding configs;
* dtype: ``bfloat16`` and ``float16``.

``thd`` and padding masks only combine with self-attention configs here
(``sq == skv``): a padded cross-attention case would need
``attention_type="cross"`` and a separate KV mask, which is a different code
path. The backend is *forced* to FlashAttention through the ``NVTE_*_ATTN`` env
vars, so every combination routes to the same backend.

Historically ``DotProductAttention.forward`` was wrapped with
``@no_torch_dynamo`` (i.e. ``torch._dynamo.disable``), so ``torch.compile``
never traced *into* attention -- it ran the whole module as an eager island.
That decorator has been removed on this branch, so Dynamo now traces the real
forward.
"""

import os

import pytest
import torch

from transformer_engine.pytorch.attention.dot_product_attention import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention import (
    _attention_backends,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    FlashAttentionUtils as fa_utils,
)

# ----------------------------------------------------------------------------
# Skips
# ----------------------------------------------------------------------------
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
    pytest.mark.skipif(not fa_utils.is_installed, reason="flash-attn is not installed"),
]


# ----------------------------------------------------------------------------
# Configurations (all expected to route to the FlashAttention backend)
# ----------------------------------------------------------------------------
# Each entry: kwargs for DotProductAttention plus the problem shape.
#   b   = batch size
#   sq  = query sequence length
#   skv = key/value sequence length
#   h   = number of attention heads
#   hg  = number of GQA groups (== h for MHA)
#   d   = head dim (qk == v)
_CONFIGS = {
    #                          b   sq   skv   h   hg    d   mask        window
    "self_no_mask": dict(b=2, sq=128, skv=128, h=8, hg=8, d=64, mask="no_mask", window=None),
    "self_causal": dict(b=2, sq=128, skv=128, h=8, hg=8, d=64, mask="causal", window=None),
    "gqa_causal": dict(b=2, sq=256, skv=256, h=16, hg=4, d=64, mask="causal", window=None),
    "mqa_causal": dict(b=2, sq=256, skv=256, h=16, hg=1, d=64, mask="causal", window=None),
    "cross_no_mask": dict(b=2, sq=128, skv=256, h=8, hg=8, d=64, mask="no_mask", window=None),
    "hdim128_causal": dict(b=2, sq=256, skv=256, h=8, hg=8, d=128, mask="causal", window=None),
    "sliding_window": dict(b=2, sq=512, skv=512, h=8, hg=8, d=64, mask="causal", window=(64, 0)),
    # Padding / variable-length (self-attention, sq == skv). These also carry
    # the thd (packed) format below.
    "self_padding": dict(b=4, sq=128, skv=128, h=8, hg=8, d=64, mask="padding", window=None),
    "self_padding_causal": dict(
        b=4, sq=128, skv=128, h=8, hg=8, d=64, mask="padding_causal", window=None
    ),
    "gqa_padding_causal": dict(
        b=4, sq=256, skv=256, h=16, hg=4, d=64, mask="padding_causal", window=None
    ),
}

# dtypes FlashAttention supports.
_DTYPES = [torch.bfloat16, torch.float16]


def _is_padding(cfg):
    return "padding" in cfg["mask"]


def _valid_formats(cfg):
    """Formats a config is exercised in: thd only makes sense with padding."""
    if _is_padding(cfg):
        return ["bshd", "sbhd", "thd"]
    return ["bshd", "sbhd"]


# (config_name, qkv_format) pairs -- only valid combinations.
_CASES = [
    (name, fmt) for name, cfg in _CONFIGS.items() for fmt in _valid_formats(cfg)
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _force_flash_backend():
    """Pin DotProductAttention to the FlashAttention backend via env vars."""
    os.environ["NVTE_FLASH_ATTN"] = "1"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    _attention_backends["backend_selection_requires_update"] = True


def _seqlens(b, max_s):
    """Deterministic, varied per-sequence lengths in ``[1, max_s]``."""
    fractions = [1.0, 0.5, 0.75, 0.25]
    return [max(1, int(max_s * fractions[i % len(fractions)])) for i in range(b)]


def _cu_seqlens(seqlens):
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.cumsum(torch.tensor(seqlens, dtype=torch.int32, device="cuda"), dim=0)
    return cu


def _padding_mask(seqlens, max_s):
    """Boolean self-attention mask ``[b, 1, 1, max_s]`` (True == masked out)."""
    mask = torch.ones(len(seqlens), 1, 1, max_s, dtype=torch.bool, device="cuda")
    for i, s in enumerate(seqlens):
        mask[i, :, :, :s] = False
    return mask


def _make_inputs(cfg, dtype, qkv_format):
    """Build (q, k, v, forward_kwargs) for a config in the given layout.

    * non-padding bshd/sbhd: dense tensors, no extra kwargs;
    * padding bshd/sbhd: dense tensors + a boolean ``attention_mask`` (so the
      mask -> cu_seqlens conversion path is exercised);
    * thd: packed variable-length tensors + ``cu_seqlens_*`` / ``max_seqlen_*``.
    """
    b, sq, skv, h, hg, d = (cfg["b"], cfg["sq"], cfg["skv"], cfg["h"], cfg["hg"], cfg["d"])

    if qkv_format == "thd":
        # Packed, variable length. Self-attention only (sq == skv), so q and kv
        # share the same per-sequence lengths / cu_seqlens.
        seqlens = _seqlens(b, sq)
        cu = _cu_seqlens(seqlens)
        total = int(cu[-1].item())
        q = torch.randn(total, h, d, dtype=dtype, device="cuda")
        k = torch.randn(total, hg, d, dtype=dtype, device="cuda")
        v = torch.randn(total, hg, d, dtype=dtype, device="cuda")
        kwargs = dict(
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=sq,
            max_seqlen_kv=skv,
        )
        return q, k, v, kwargs

    if qkv_format == "bshd":
        q_shape, kv_shape = (b, sq, h, d), (b, skv, hg, d)
    elif qkv_format == "sbhd":
        q_shape, kv_shape = (sq, b, h, d), (skv, b, hg, d)
    else:
        raise ValueError(f"unsupported qkv_format {qkv_format}")
    q = torch.randn(q_shape, dtype=dtype, device="cuda")
    k = torch.randn(kv_shape, dtype=dtype, device="cuda")
    v = torch.randn(kv_shape, dtype=dtype, device="cuda")

    kwargs = {}
    if _is_padding(cfg):
        # Self-attention padding mask (sq == skv) over the query seqlen.
        kwargs["attention_mask"] = _padding_mask(_seqlens(b, sq), sq)
    return q, k, v, kwargs


def _build_dpa(cfg, qkv_format):
    return DotProductAttention(
        num_attention_heads=cfg["h"],
        kv_channels=cfg["d"],
        num_gqa_groups=cfg["hg"],
        attention_dropout=0.0,
        qkv_format=qkv_format,
        attn_mask_type=cfg["mask"],
        window_size=cfg["window"],
    ).cuda()


# ----------------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).rsplit(".", 1)[-1])
@pytest.mark.parametrize(
    "config_name,qkv_format", _CASES, ids=[f"{n}-{f}" for n, f in _CASES]
)
def test_flash_backend_fullgraph(config_name, qkv_format, dtype):
    """DotProductAttention (FlashAttention) must trace with fullgraph=True.

    ``fullgraph=True`` raises on any graph break, so a passing run means the
    whole forward was captured into a single graph. We also assert the compiled
    output matches eager, for every (config, qkv_format, dtype) combination.
    """
    cfg = _CONFIGS[config_name]

    _force_flash_backend()
    dpa = _build_dpa(cfg, qkv_format)
    q, k, v, fwd_kwargs = _make_inputs(cfg, dtype, qkv_format)

    out_eager = dpa(q, k, v, **fwd_kwargs)

    torch._dynamo.reset()
    compiled = torch.compile(dpa, fullgraph=True, dynamic=False)
    out_compiled = compiled(q, k, v, **fwd_kwargs)

    torch.testing.assert_close(out_compiled, out_eager, atol=0.0, rtol=0.0)
