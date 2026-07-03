# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Deprecation-warning coverage for internal max_seqlen derivation.

On the ``thd`` (packed, variable-length) path, ``DotProductAttention`` needs
host-side ``max_seqlen_q`` / ``max_seqlen_kv`` integers. When the caller does
*not* pass them, TE derives them from ``cu_seqlens`` via a ``.item()`` call,
which forces a device-to-host CUDA synchronization and breaks
``torch.compile``. This suite asserts that:

* calling ``forward`` WITHOUT ``max_seqlen_q`` / ``max_seqlen_kv`` emits a
  ``DeprecationWarning`` (the derivation / sync path);
* calling WITH them emits no such warning;
* both produce a bit-identical result.

The backend is forced to FlashAttention (the ``thd`` derivation lives on that
path in ``backends.py``, and the same derivation also runs in
``dot_product_attention.py``).
"""

import os
import warnings

import pytest
import torch

from transformer_engine.pytorch.attention.dot_product_attention import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention import (
    _attention_backends,
)
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    FlashAttentionUtils as fa_utils,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
    pytest.mark.skipif(not fa_utils.is_installed, reason="flash-attn is not installed"),
]


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


def _make_thd_inputs(b=4, s=128, h=8, d=64, dtype=torch.bfloat16):
    """Packed thd self-attention q/k/v + cu_seqlens (variable seqlens)."""
    seqlens = _seqlens(b, s)
    cu = _cu_seqlens(seqlens)
    total = int(cu[-1].item())
    q = torch.randn(total, h, d, dtype=dtype, device="cuda")
    k = torch.randn(total, h, d, dtype=dtype, device="cuda")
    v = torch.randn(total, h, d, dtype=dtype, device="cuda")
    return q, k, v, cu, s


def _build_dpa(h=8, d=64):
    return DotProductAttention(
        num_attention_heads=h,
        kv_channels=d,
        num_gqa_groups=h,
        attention_dropout=0.0,
        qkv_format="thd",
        attn_mask_type="padding",
    ).cuda()


def test_max_seqlen_derivation_deprecation():
    """WITHOUT max_seqlen warns; WITH max_seqlen does not; outputs are identical."""
    _force_flash_backend()
    dpa = _build_dpa()
    q, k, v, cu, max_s = _make_thd_inputs()

    common = dict(cu_seqlens_q=cu, cu_seqlens_kv=cu)

    # WITH max_seqlen passed -> no derivation, no DeprecationWarning.
    dpa_utils._warned_max_seqlen_derivation = False
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out_with = dpa(q, k, v, max_seqlen_q=max_s, max_seqlen_kv=max_s, **common)
    assert not any(
        issubclass(w.category, DeprecationWarning) for w in record
    ), "passing max_seqlen_q/max_seqlen_kv must not emit a DeprecationWarning"

    # WITHOUT max_seqlen passed -> internal derivation (.item) -> DeprecationWarning.
    dpa_utils._warned_max_seqlen_derivation = False
    with pytest.warns(DeprecationWarning):
        out_without = dpa(q, k, v, **common)

    # Bit-exact equivalence of the two paths.
    torch.testing.assert_close(out_without, out_with, atol=0.0, rtol=0.0)
