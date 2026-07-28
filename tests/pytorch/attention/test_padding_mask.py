# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

from transformer_engine.pytorch.attention.dot_product_attention import utils as dpa_utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")


def _reference_mask(batch_size, cu_seqlens, max_seqlen):
    """Straightforward host-side construction of the padding mask."""
    seqlens = (cu_seqlens[1 : batch_size + 1] - cu_seqlens[:batch_size]).tolist()
    rows = [[False] * s + [True] * (max_seqlen - s) for s in seqlens]
    return torch.tensor(rows, dtype=torch.bool, device=cu_seqlens.device).view(
        batch_size, 1, 1, max_seqlen
    )


def _cu_seqlens(seqlens, device="cuda", length=None):
    """Cumulative sequence lengths, optionally padded to `length` entries as the
    KV cache does (it allocates cu_seqlens for the maximum batch size)."""
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)
    if length is not None:
        cu = torch.cat([cu, cu[-1].repeat(length - cu.numel())])
    return cu


@pytest.mark.parametrize(
    "seqlens,max_seqlen",
    [
        ([4, 9, 12, 16], 16),  # mixed lengths
        ([16, 16], 16),  # no padding at all
        ([0, 5], 8),  # empty sequence
        ([7], 7),  # single sequence, full
        ([1, 2, 3, 4, 5], 8),
    ],
)
def test_padding_mask_matches_reference(seqlens, max_seqlen):
    cu = _cu_seqlens(seqlens)
    mask = dpa_utils.get_padding_mask(len(seqlens), cu, None, max_seqlen)
    ref = _reference_mask(len(seqlens), cu, max_seqlen)
    assert mask.dtype == torch.bool
    assert mask.shape == (len(seqlens), 1, 1, max_seqlen)
    torch.testing.assert_close(mask, ref)


def test_padding_mask_cross_attention():
    seqlens_q, seqlens_kv, max_q, max_kv = [3, 5], [7, 2], 8, 8
    cu_q, cu_kv = _cu_seqlens(seqlens_q), _cu_seqlens(seqlens_kv)
    mask_q, mask_kv = dpa_utils.get_padding_mask(2, cu_q, cu_kv, max_q, max_kv, "cross")
    torch.testing.assert_close(mask_q, _reference_mask(2, cu_q, max_q))
    torch.testing.assert_close(mask_kv, _reference_mask(2, cu_kv, max_kv))


def test_padding_mask_cu_seqlens_longer_than_batch():
    """Inference allocates cu_seqlens for the maximum batch size, so only its
    first batch_size + 1 entries describe the current batch."""
    seqlens, max_seqlen, batch_size = [5, 3], 8, 2
    cu = _cu_seqlens(seqlens, length=len(seqlens) + 4)
    mask = dpa_utils.get_padding_mask(batch_size, cu, None, max_seqlen)
    torch.testing.assert_close(mask, _reference_mask(batch_size, cu, max_seqlen))


def test_padding_mask_does_not_synchronize():
    """The mask must be built on the device: a host-side read costs a
    synchronization per sequence on every forward pass."""
    cu = _cu_seqlens([4, 9, 12, 16])
    dpa_utils.get_padding_mask(4, cu, None, 16)  # warm up any lazy initialization
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        dpa_utils.get_padding_mask(4, cu, None, 16)
    finally:
        torch.cuda.set_sync_debug_mode("default")
