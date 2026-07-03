# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile coverage for DotProductAttention.

This is a *diagnostic* test suite. Its purpose is to drive
``DotProductAttention`` through ``torch.compile`` and surface every piece of
the module that does not trace cleanly (graph breaks).

We start with the FlashAttention backend, whose underlying ``flash_attn``
kernels are already registered as custom ops that Dynamo can capture.

Current state of the world (see ``test_forward_is_dynamo_disabled``):
``DotProductAttention.forward`` is wrapped with ``@no_torch_dynamo`` (i.e.
``torch._dynamo.disable``) in ``transformer_engine/pytorch/jit.py``. As a
result ``torch.compile`` never traces *into* attention -- it silently runs the
whole module eagerly. So "get the FlashAttention backend to run under
torch.compile" has two layers:

1. It already *executes* correctly under torch.compile (as an eager island).
   ``test_flash_backend_runs_under_compile`` pins this down.
2. To actually *trace* it we must bypass the dynamo-disable and then chip away
   at the internal graph breaks. ``test_flash_backend_graph_breaks`` traces the
   unwrapped forward and prints the full inventory of breaks to attack next.
"""

import os

import pytest
import torch
from torch._dynamo.utils import counters

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

# Small, backend-agnostic problem shape used across the suite.
_BATCH, _SEQLEN, _HEADS, _HEAD_DIM = 2, 128, 8, 64


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _force_flash_backend():
    """Pin DotProductAttention to the FlashAttention backend."""
    os.environ["NVTE_FLASH_ATTN"] = "1"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    # Force backend re-selection now that the env vars changed.
    _attention_backends["backend_selection_requires_update"] = True


def _make_qkv(dtype, requires_grad=False):
    """Create a (q, k, v) triple in ``bshd`` layout on CUDA."""
    shape = (_BATCH, _SEQLEN, _HEADS, _HEAD_DIM)
    return tuple(
        torch.randn(shape, dtype=dtype, device="cuda", requires_grad=requires_grad)
        for _ in range(3)
    )


def _build_dpa(mask_type="causal"):
    return DotProductAttention(
        num_attention_heads=_HEADS,
        kv_channels=_HEAD_DIM,
        attention_dropout=0.0,
        attn_mask_type=mask_type,
        qkv_format="bshd",
    ).cuda()


def _forward_traceable(dpa):
    """Return a callable that runs DPA.forward *without* the dynamo-disable.

    ``@no_torch_dynamo`` wraps ``forward`` with ``functools.wraps``, so the raw
    (undecorated) function is reachable via ``__wrapped__``. Calling it lets
    ``torch.compile`` trace into attention and expose the internal graph breaks
    that the decorator otherwise hides.
    """
    raw_forward = type(dpa).forward.__wrapped__

    def run(q, k, v):
        return raw_forward(dpa, q, k, v)

    return run


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def test_forward_is_dynamo_disabled():
    """Document the current blocker: DPA.forward is dynamo-disabled.

    This is what makes torch.compile skip attention entirely today. When the
    branch removes/loosens that decorator, this test is the canary that tells us
    the tracing surface changed.
    """
    fwd = DotProductAttention.forward
    assert hasattr(fwd, "__wrapped__"), "expected @no_torch_dynamo to wrap forward"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flash_backend_runs_under_compile(dtype):
    """FlashAttention backend must run under torch.compile and match eager.

    With the dynamo-disable in place this exercises the "eager island" path:
    the flash kernel is reached and the result is bit-exact against eager.
    """
    _force_flash_backend()
    q, k, v = _make_qkv(dtype)
    dpa = _build_dpa()

    out_eager = dpa(q, k, v)

    torch._dynamo.reset()
    compiled = torch.compile(dpa, fullgraph=False, dynamic=False)
    out_compiled = compiled(q, k, v)

    torch.testing.assert_close(out_compiled, out_eager, atol=0.0, rtol=0.0)


def test_flash_backend_graph_breaks():
    """Inventory the graph breaks on the (unwrapped) FlashAttention forward.

    Bypasses the dynamo-disable and traces the real forward so we can see every
    construct Dynamo cannot handle. Does not assert a specific count yet -- it
    prints the inventory so the breaks can be attacked one at a time. It only
    fails if flash attention is never reached under tracing (op_count == 0).
    """
    _force_flash_backend()
    q, k, v = _make_qkv(torch.bfloat16)
    dpa = _build_dpa()

    torch._dynamo.reset()
    counters.clear()

    explanation = torch._dynamo.explain(_forward_traceable(dpa))(q, k, v)

    print("\n===== DotProductAttention / FlashAttention torch.compile report =====")
    print(f"graph count       : {explanation.graph_count}")
    print(f"graph break count : {explanation.graph_break_count}")
    print(f"op count          : {explanation.op_count}")

    # `explain().break_reasons` is unreliable when an inner frame is
    # dynamo-disabled; the counters dict is the authoritative inventory.
    print("----- graph break inventory (counters) -----")
    gb = counters.get("graph_break", {})
    if not gb:
        print("  (none recorded via counters)")
    for reason, count in sorted(gb.items(), key=lambda kv: -kv[1]):
        print(f"  [{count:>3}x] {reason}")

    print("----- break reasons (explain) -----")
    for i, reason in enumerate(explanation.break_reasons):
        print(f"[{i}] {getattr(reason, 'reason', reason)}")
    print("=====================================================================")

    # The whole point of step 1: flash attention must actually be reached while
    # tracing (i.e. Dynamo captured real ops, not an empty graph).
    assert explanation.op_count >= 1
