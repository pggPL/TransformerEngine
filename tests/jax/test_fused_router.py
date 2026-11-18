# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for fused MoE router CUDA kernels (JAX wrappers)."""

import sys
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utils import pytest_parametrize_wrapper


@pytest.fixture(autouse=True, scope="function")
def _inject_router():
    """Inject the router API as module-level names for every test in this file."""
    from transformer_engine.jax.router import (
        fused_topk_with_score_function,
        fused_moe_aux_loss,
    )

    mod = sys.modules[__name__]
    mod.fused_topk_with_score_function = fused_topk_with_score_function
    mod.fused_moe_aux_loss = fused_moe_aux_loss
    yield


# =============================================================================
# Test case definitions (L0 = fast smoke, L2 = comprehensive)
# =============================================================================

# (num_tokens, num_experts, topk)
ALL_TOPK_CASES = [
    (128, 32, 4),
    (2048, 32, 4),
    (2048, 128, 8),
    (7168, 128, 4),
    (7168, 32, 8),
]
TOPK_CASES = {
    "L0": ALL_TOPK_CASES[0:2],
    "L2": ALL_TOPK_CASES,
}

ALL_GROUP_TOPK_OPTIONS = [None, 4]
GROUP_TOPK_OPTIONS = {
    "L0": [None],
    "L2": ALL_GROUP_TOPK_OPTIONS,
}

ALL_SCALING_FACTOR_OPTIONS = [None, 1.2]
SCALING_FACTOR_OPTIONS = {
    "L0": [None],
    "L2": ALL_SCALING_FACTOR_OPTIONS,
}

ALL_ENABLE_BIAS_OPTIONS = [True, False]
ENABLE_BIAS_OPTIONS = {
    "L0": [False],
    "L2": ALL_ENABLE_BIAS_OPTIONS,
}

ALL_USE_PRE_SOFTMAX_OPTIONS = [True, False]
USE_PRE_SOFTMAX_OPTIONS = {
    "L0": [False],
    "L2": ALL_USE_PRE_SOFTMAX_OPTIONS,
}

# (num_tokens, num_experts, topk)
ALL_SCORE_AUX_LOSS_CASES = [
    (128, 32, 4),
    (2048, 128, 4),
    (2048, 256, 8),
    (7168, 128, 8),
    (7168, 32, 4),
]
SCORE_AUX_LOSS_CASES = {
    "L0": ALL_SCORE_AUX_LOSS_CASES[0:2],
    "L2": ALL_SCORE_AUX_LOSS_CASES,
}

ALL_SCORE_FUNCTIONS = ["softmax", "sigmoid", "sqrtsoftplus"]
SCORE_FUNCTIONS = {
    "L0": ["softmax", "sqrtsoftplus"],
    "L2": ALL_SCORE_FUNCTIONS,
}

# (num_tokens, num_experts, topk)
ALL_AUX_LOSS_CASES = [
    (128, 32, 4),
    (2048, 128, 4),
    (2048, 256, 4),
    (7168, 128, 4),
    (7168, 32, 4),
]
AUX_LOSS_CASES = {
    "L0": ALL_AUX_LOSS_CASES[0:2],
    "L2": ALL_AUX_LOSS_CASES,
}

ALL_DTYPES = [jnp.float32]
DTYPES = {
    "L0": [jnp.float32],
    "L2": ALL_DTYPES,
}

SEED = 42


def reference_qb_histogram(logits, expert_bias, bounds, topk, num_bins):
    scores = np.asarray(jax.nn.sigmoid(jnp.asarray(logits, jnp.float32)))
    selected, _ = jax.lax.top_k(jnp.asarray(scores + np.asarray(expert_bias)), topk + 1)
    required_bias = np.asarray(selected)[:, -1:] - scores
    lower, upper = np.asarray(bounds)
    bins = np.floor((required_bias - lower) * (num_bins / (upper - lower))).astype(np.int32)
    bins = np.clip(bins, 0, num_bins - 1)
    histogram = np.zeros((scores.shape[-1], num_bins), np.int32)
    np.add.at(histogram, (np.arange(scores.shape[-1])[None, :], bins), 1)
    return histogram


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
@pytest.mark.parametrize("routing_map_format", ["bytemap", "bitmap_u8"])
@pytest.mark.parametrize("topk", [1, 4])
def test_qb_router_forward_backward_and_remat(histogram_mode, routing_map_format, topk):
    from transformer_engine.jax.router import fused_topk_with_score_function_qb

    logits = jax.random.normal(jax.random.PRNGKey(801), (128, 32))
    bias = jax.random.uniform(jax.random.PRNGKey(802), (32,), minval=-0.1, maxval=0.1)
    bounds = jnp.asarray([-1.1, 1.1], jnp.float32)
    coefficients = jax.random.normal(jax.random.PRNGKey(803), logits.shape)

    def loss(x, b, limits):
        probs, routing, histogram = fused_topk_with_score_function_qb(
            x,
            topk,
            b,
            limits,
            num_bins=64,
            scaling_factor=1.5,
            histogram_mode=histogram_mode,
            routing_map_format=routing_map_format,
        )
        return jnp.sum(probs * coefficients), (probs, routing, histogram)

    (_, (probs, routing, histogram)), (dx, db, dbounds) = jax.jit(
        jax.value_and_grad(jax.checkpoint(loss), argnums=(0, 1, 2), has_aux=True)
    )(logits, bias, bounds)

    def reference(x):
        probs, _ = reference_topk_with_score_function(
            x,
            topk,
            score_function="sigmoid",
            expert_bias=bias,
            scaling_factor=1.5,
        )
        return jnp.sum(probs * coefficients), probs

    (_, expected_probs), expected_dx = jax.value_and_grad(reference, has_aux=True)(logits)
    np.testing.assert_allclose(probs, expected_probs, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(dx, expected_dx, rtol=2e-4, atol=2e-6)
    np.testing.assert_array_equal(db, jnp.zeros_like(bias))
    np.testing.assert_array_equal(dbounds, jnp.zeros_like(bounds))
    np.testing.assert_array_equal(histogram, reference_qb_histogram(logits, bias, bounds, topk, 64))
    expected_map = np.asarray(expected_probs != 0)
    if routing_map_format == "bitmap_u8":
        expected_map = np.packbits(expected_map, axis=-1, bitorder="little")
    np.testing.assert_array_equal(routing, expected_map)


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
def test_qb_scan_accumulation_preserves_initial_histogram(histogram_mode):
    from transformer_engine.jax.router import fused_topk_with_score_function_qb

    logits = jax.random.normal(jax.random.PRNGKey(811), (3, 64, 32))
    bias = jnp.linspace(-0.1, 0.1, 32, dtype=jnp.float32)
    bounds = jnp.asarray([-1.1, 1.1], jnp.float32)
    initial = jnp.full((32, 64), 7, jnp.int32)

    def accumulate(histogram, x):
        _, _, updated = fused_topk_with_score_function_qb(
            x,
            4,
            bias,
            bounds,
            num_bins=64,
            histogram=histogram,
            histogram_mode=histogram_mode,
        )
        return updated, None

    @jax.jit
    def run(histogram, x):
        total, _ = jax.lax.scan(accumulate, histogram, x)
        return total, histogram + 1

    actual, still_live = run(initial, logits)
    expected = np.full((32, 64), 7, np.int32)
    for microbatch in np.asarray(logits):
        expected += reference_qb_histogram(microbatch, bias, bounds, 4, 64)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(still_live, np.full((32, 64), 8, np.int32))
    np.testing.assert_array_equal(initial, np.full((32, 64), 7, np.int32))


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
def test_qb_bounds_change_and_clamp(histogram_mode):
    from transformer_engine.jax.router import fused_topk_with_score_function_qb

    logits = jax.random.normal(jax.random.PRNGKey(821), (64, 32))
    bias = jnp.linspace(-0.2, 0.2, 32, dtype=jnp.float32)
    route = jax.jit(
        partial(
            fused_topk_with_score_function_qb,
            topk=4,
            expert_bias=bias,
            num_bins=32,
            histogram_mode=histogram_mode,
        )
    )
    results = []
    for limits in ([-1.2, 1.2], [-0.025, 0.025]):
        bounds = jnp.asarray(limits, jnp.float32)
        _, _, histogram = route(logits, bin_bounds=bounds)
        np.testing.assert_array_equal(
            histogram, reference_qb_histogram(logits, bias, bounds, 4, 32)
        )
        np.testing.assert_array_equal(np.asarray(histogram).sum(axis=-1), np.full(32, 64))
        results.append(np.asarray(histogram))
    assert not np.array_equal(*results)


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
def test_qb_tied_cutoff_counts_each_token_once(histogram_mode):
    from transformer_engine.jax.router import fused_topk_with_score_function_qb

    _, routing, histogram = jax.jit(
        partial(
            fused_topk_with_score_function_qb,
            topk=4,
            num_bins=32,
            histogram_mode=histogram_mode,
        )
    )(jnp.zeros((64, 32)), expert_bias=jnp.zeros(32), bin_bounds=jnp.asarray([-1.0, 1.0]))
    np.testing.assert_array_equal(np.asarray(routing).sum(axis=-1), np.full(64, 4))
    expected = np.zeros((32, 32), np.int32)
    expected[:, 16] = 64
    np.testing.assert_array_equal(histogram, expected)


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
@pytest.mark.parametrize("valid_rows", [0, 17, 64])
def test_qb_masked_histogram_and_gradients(histogram_mode, valid_rows):
    from transformer_engine.jax.router import fused_topk_with_score_function_qb

    logits = jax.random.normal(jax.random.PRNGKey(851), (64, 32))
    bias = jnp.linspace(-0.1, 0.1, 32)
    bounds = jnp.asarray([-1.1, 1.1], jnp.float32)
    mask = jnp.arange(64) < valid_rows

    def loss(x):
        probs, _, counts = fused_topk_with_score_function_qb(
            x,
            4,
            bias,
            bounds,
            token_mask=mask,
            num_bins=64,
            histogram_mode=histogram_mode,
        )
        return jnp.sum(probs * x), counts

    (_, counts), gradient = jax.jit(jax.value_and_grad(jax.checkpoint(loss), has_aux=True))(logits)
    if valid_rows:
        expected = reference_qb_histogram(logits[:valid_rows], bias, bounds, 4, 64)
    else:
        expected = np.zeros((32, 64), np.int32)
    np.testing.assert_array_equal(counts, expected)
    np.testing.assert_array_equal(np.asarray(counts).sum(axis=-1), np.full(32, valid_rows))
    assert np.isfinite(np.asarray(gradient)).all()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"topk": 32}, "topk"),
        ({"expert_bias": None}, "expert_bias"),
        ({"num_bins": 0}, "num_bins"),
        ({"histogram_mode": "unknown"}, "histogram_mode"),
        ({"bin_bounds": jnp.zeros(3, jnp.float32)}, "bin_bounds"),
        ({"histogram": jnp.zeros((32, 64), jnp.float32)}, "histogram"),
    ],
)
def test_qb_router_rejects_invalid_inputs(overrides, match):
    from transformer_engine.jax.router import fused_topk_with_score_function_qb

    kwargs = dict(
        topk=4, expert_bias=jnp.zeros(32), bin_bounds=jnp.asarray([-1.0, 1.0]), num_bins=64
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        fused_topk_with_score_function_qb(jnp.zeros((64, 32)), **kwargs)


# =============================================================================
# Reference Implementations
# =============================================================================


def reference_group_limited_topk(
    scores: jnp.ndarray,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
):
    """Reference implementation for grouped top-k.

    Only valid when num_groups and group_topk are both positive integers.
    For plain top-k without grouping, use jax.lax.top_k directly.
    """
    assert num_groups is not None and num_groups > 0, (
        "reference_group_limited_topk requires valid num_groups > 0. "
        "For plain top-k, use jax.lax.top_k directly."
    )
    assert (
        group_topk is not None and group_topk > 0
    ), "reference_group_limited_topk requires valid group_topk > 0."
    assert (
        num_experts % num_groups == 0
    ), f"num_experts ({num_experts}) must be divisible by num_groups ({num_groups})"
    group_size = num_experts // num_groups
    experts_per_group = topk // group_topk

    group_scores = (
        scores.reshape(num_tokens, num_groups, group_size)
        .sort(axis=-1)[..., -experts_per_group:]
        .sum(axis=-1)
    )
    group_idx = jax.lax.top_k(group_scores, k=group_topk)[1]
    group_mask = jnp.zeros_like(group_scores).at[jnp.arange(num_tokens)[:, None], group_idx].set(1)

    score_mask = (group_mask[:, :, None] * jnp.ones((num_tokens, num_groups, group_size))).reshape(
        num_tokens, -1
    )

    masked_scores = jnp.where(score_mask.astype(bool), scores, -jnp.inf)
    probs, top_indices = jax.lax.top_k(masked_scores, k=topk)
    return probs, top_indices


def reference_topk_with_score_function(
    logits: jnp.ndarray,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[jnp.ndarray] = None,
):
    """Reference implementation for topk with a supported score function."""
    num_tokens, num_experts = logits.shape

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            return reference_group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return jax.lax.top_k(scores, k=topk)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(logits.dtype)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(logits.dtype)
    elif score_function in ("sigmoid", "sqrtsoftplus"):
        if score_function == "sigmoid":
            scores = jax.nn.sigmoid(logits.astype(jnp.float32)).astype(logits.dtype)
        else:
            scores = jnp.sqrt(jax.nn.softplus(logits.astype(jnp.float32))).astype(logits.dtype)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = jnp.take_along_axis(scores, top_indices, axis=1).astype(logits.dtype)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    topk_masked_gates = (
        jnp.zeros_like(logits).at[jnp.arange(num_tokens)[:, None], top_indices].set(probs)
    )
    topk_map = (
        jnp.zeros_like(logits, dtype=jnp.bool_)
        .at[jnp.arange(num_tokens)[:, None], top_indices]
        .set(True)
    )

    return topk_masked_gates, topk_map


def reference_compute_scores_for_aux_loss(logits: jnp.ndarray, topk: int, score_function: str):
    """Reference implementation for computing routing scores for aux loss."""
    if score_function == "softmax":
        scores = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    elif score_function == "sigmoid":
        scores = jax.nn.sigmoid(logits.astype(jnp.float32))
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20) if topk > 1 else scores
    elif score_function == "sqrtsoftplus":
        scores = jnp.sqrt(jax.nn.softplus(logits.astype(jnp.float32)))
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    _, top_indices = jax.lax.top_k(scores, k=topk)
    num_tokens = logits.shape[0]
    routing_map = (
        jnp.zeros_like(logits, dtype=jnp.bool_)
        .at[jnp.arange(num_tokens)[:, None], top_indices]
        .set(True)
    )
    return routing_map, scores


def reference_aux_loss(
    probs: jnp.ndarray,
    tokens_per_expert: jnp.ndarray,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
):
    """Reference implementation for MoE auxiliary loss."""
    aggregated_probs_per_expert = probs.sum(axis=0)
    aux_loss = jnp.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss


# =============================================================================
# Helper: logits generation
# =============================================================================


def make_logits(num_tokens, num_experts, score_function, dtype=jnp.float32):
    """Create deterministic logits for testing."""
    if score_function in ("sigmoid", "sqrtsoftplus"):
        offset = jnp.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype) * 1e-4
        logits = jnp.arange(-num_experts // 2, num_experts // 2, dtype=dtype) * 1e-2
        logits = logits[None, :].repeat(num_tokens, axis=0) + offset[:, None]
    else:
        logits = (
            jnp.arange(
                -num_tokens * num_experts // 2,
                num_tokens * num_experts // 2,
                dtype=dtype,
            )
            * 1e-4
        )
        logits = logits.reshape(num_tokens, num_experts)
    return logits


# =============================================================================
# Test: Fused Top-K with Score Function
# =============================================================================


def run_topk_comparison(
    dtype,
    num_tokens,
    num_experts,
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
    enable_bias,
):
    """Compare fused vs reference top-k implementation, both jitted."""
    logits = make_logits(num_tokens, num_experts, score_function, dtype)

    if enable_bias and score_function in ("sigmoid", "sqrtsoftplus"):
        expert_bias = jnp.arange(num_experts, dtype=jnp.float32) * 0.1
        expert_bias = jnp.flip(expert_bias)
    else:
        expert_bias = None

    # Forward: reference (jitted)
    ref_fwd_fn = jax.jit(
        partial(
            reference_topk_with_score_function,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=expert_bias,
        )
    )
    probs_ref, routing_map_ref = ref_fwd_fn(logits)

    # Forward: fused (jitted)
    fused_fwd_fn = jax.jit(
        partial(
            fused_topk_with_score_function,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups if num_groups else -1,
            group_topk=group_topk if group_topk else -1,
            scaling_factor=scaling_factor if scaling_factor else 1.0,
            score_function=score_function,
            expert_bias=expert_bias,
        )
    )
    probs_fused, routing_map_fused = fused_fwd_fn(logits)

    assert jnp.allclose(
        probs_ref, probs_fused, atol=1e-5, rtol=1e-5
    ), f"Probs mismatch: max diff = {jnp.abs(probs_ref - probs_fused).max()}"
    assert jnp.array_equal(routing_map_ref, routing_map_fused), "Routing map mismatch"

    # Backward: reference (jitted)
    # Use random weights so the backward pass receives non-uniform gradients instead of
    # the all-ones gradient produced by an unweighted jnp.sum.
    grad_weights = jax.random.uniform(jax.random.PRNGKey(SEED), (1, num_experts), dtype=jnp.float32)

    def loss_ref(logits_):
        p, _ = reference_topk_with_score_function(
            logits_,
            topk,
            use_pre_softmax,
            num_groups,
            group_topk,
            scaling_factor,
            score_function,
            expert_bias,
        )
        return jnp.sum(p * grad_weights)

    def loss_fused(logits_):
        p, _ = fused_topk_with_score_function(
            logits_,
            topk,
            use_pre_softmax,
            num_groups if num_groups else -1,
            group_topk if group_topk else -1,
            scaling_factor if scaling_factor else 1.0,
            score_function,
            expert_bias,
        )
        return jnp.sum(p * grad_weights)

    grad_ref = jax.jit(jax.grad(loss_ref))(logits)
    grad_fused = jax.jit(jax.grad(loss_fused))(logits)
    assert jnp.allclose(
        grad_ref, grad_fused, atol=1e-5, rtol=1e-5
    ), f"Grad mismatch: max diff = {jnp.abs(grad_ref - grad_fused).max()}"


@pytest_parametrize_wrapper("dtype", DTYPES)
@pytest_parametrize_wrapper(
    "num_tokens,num_experts,topk",
    TOPK_CASES,
)
@pytest_parametrize_wrapper("group_topk", GROUP_TOPK_OPTIONS)
@pytest_parametrize_wrapper("scaling_factor", SCALING_FACTOR_OPTIONS)
@pytest_parametrize_wrapper("enable_bias", ENABLE_BIAS_OPTIONS)
@pytest_parametrize_wrapper("use_pre_softmax", USE_PRE_SOFTMAX_OPTIONS)
@pytest_parametrize_wrapper("score_function", SCORE_FUNCTIONS)
def test_topk(
    dtype,
    num_tokens,
    num_experts,
    topk,
    group_topk,
    scaling_factor,
    enable_bias,
    use_pre_softmax,
    score_function,
):
    if use_pre_softmax and score_function != "softmax":
        pytest.skip("Pre-softmax is only applicable to the 'softmax' router score function.")
    if score_function == "softmax" and enable_bias:
        pytest.skip("Bias is not supported with 'softmax' router score function. Skipping.")
    num_groups = 8 if group_topk else None
    run_topk_comparison(
        dtype=dtype,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
        enable_bias=enable_bias,
    )


# =============================================================================
# Test: Fused Score for MoE Aux Loss
# =============================================================================


@pytest_parametrize_wrapper("dtype", DTYPES)
@pytest_parametrize_wrapper(
    "num_tokens,num_experts,topk",
    SCORE_AUX_LOSS_CASES,
)
@pytest_parametrize_wrapper("score_function", SCORE_FUNCTIONS)
def test_fused_scores_for_aux_loss(dtype, num_tokens, num_experts, topk, score_function):
    logits = make_logits(num_tokens, num_experts, score_function, dtype)

    # Forward: reference (jitted)
    ref_fwd_fn = jax.jit(
        partial(
            reference_compute_scores_for_aux_loss,
            topk=topk,
            score_function=score_function,
        )
    )
    routing_map_ref, scores_ref = ref_fwd_fn(logits)

    # Forward: fused (jitted)
    fused_fwd_fn = jax.jit(
        partial(
            fused_topk_with_score_function,
            topk=topk,
            score_function=score_function,
            compute_aux_scores=True,
        )
    )
    scores_fused, routing_map_fused = fused_fwd_fn(logits)

    assert jnp.allclose(
        scores_ref, scores_fused, atol=1e-5, rtol=1e-5
    ), f"Scores mismatch: max diff = {jnp.abs(scores_ref - scores_fused).max()}"
    assert jnp.array_equal(routing_map_ref, routing_map_fused), "Routing map mismatch"

    # Backward (jitted)
    # Use random weights so the backward pass receives non-uniform gradients instead of
    # the all-ones gradient produced by an unweighted jnp.sum.
    grad_weights = jax.random.uniform(jax.random.PRNGKey(SEED), (1, num_experts), dtype=jnp.float32)

    def loss_ref(logits_):
        _, s = reference_compute_scores_for_aux_loss(logits_, topk, score_function)
        return jnp.sum(s * grad_weights)

    def loss_fused(logits_):
        s, _ = fused_topk_with_score_function(
            logits_,
            topk,
            score_function=score_function,
            compute_aux_scores=True,
        )
        return jnp.sum(s * grad_weights)

    grad_ref = jax.jit(jax.grad(loss_ref))(logits)
    grad_fused = jax.jit(jax.grad(loss_fused))(logits)
    assert jnp.allclose(
        grad_ref, grad_fused, atol=1e-5, rtol=1e-5
    ), f"Grad mismatch: max diff = {jnp.abs(grad_ref - grad_fused).max()}"


# =============================================================================
# Test: Fused MoE Aux Loss
# =============================================================================


@pytest_parametrize_wrapper("dtype", DTYPES)
@pytest_parametrize_wrapper(
    "num_tokens,num_experts,topk",
    AUX_LOSS_CASES,
)
def test_fused_moe_aux_loss(dtype, num_tokens, num_experts, topk):
    key = jax.random.PRNGKey(SEED)

    offset = jnp.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype) * 1e-4
    probs = jnp.arange(-num_experts // 2, num_experts // 2, dtype=dtype) * 1e-2
    probs = probs[None, :].repeat(num_tokens, axis=0) + offset[:, None]
    probs = probs.reshape(num_tokens, num_experts)

    tokens_per_expert = jax.random.randint(key, (num_experts,), 1, 1000).astype(jnp.int32)
    coeff = 0.01

    # Forward: reference (jitted)
    ref_fwd_fn = jax.jit(
        partial(
            reference_aux_loss,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=num_tokens,
            topk=topk,
            num_experts=num_experts,
            moe_aux_loss_coeff=coeff,
        )
    )
    aux_loss_ref = ref_fwd_fn(probs)

    # Forward: fused (jitted)
    fused_fwd_fn = jax.jit(
        partial(
            fused_moe_aux_loss,
            tokens_per_expert=tokens_per_expert,
            topk=topk,
            coeff=coeff,
        )
    )
    aux_loss_fused = fused_fwd_fn(probs)

    assert jnp.allclose(
        aux_loss_ref, aux_loss_fused, atol=1e-5, rtol=1e-5
    ), f"Aux loss mismatch: ref={aux_loss_ref}, fused={aux_loss_fused}"

    # Backward (jitted)
    def loss_ref_fn(probs_):
        return reference_aux_loss(probs_, tokens_per_expert, num_tokens, topk, num_experts, coeff)

    def loss_fused_fn(probs_):
        return fused_moe_aux_loss(probs_, tokens_per_expert, topk, coeff)

    grad_ref = jax.jit(jax.grad(loss_ref_fn))(probs)
    grad_fused = jax.jit(jax.grad(loss_fused_fn))(probs)
    assert jnp.allclose(
        grad_ref, grad_fused, atol=1e-5, rtol=1e-5
    ), f"Grad mismatch: max diff = {jnp.abs(grad_ref - grad_fused).max()}"


# =============================================================================
# Test: routing_map BITMAP_U8 vs BYTEMAP parity (fwd + bwd)
# =============================================================================


def _bytemap_to_bitmap_u8(bytemap):
    """Reference packer: bool[T, E] -> uint8[T, ceil(E/8)] LSB-first."""
    import numpy as np

    flat = np.asarray(bytemap).astype(np.uint8)
    return np.packbits(flat, axis=-1, bitorder="little")


@pytest_parametrize_wrapper("dtype", DTYPES)
@pytest_parametrize_wrapper(
    "num_tokens,num_experts,topk",
    TOPK_CASES,
)
@pytest_parametrize_wrapper("score_function", SCORE_FUNCTIONS)
def test_topk_bitmap_vs_bytemap(dtype, num_tokens, num_experts, topk, score_function):
    """fused_topk_with_score_function should produce the same probs and an
    LSB-packed bitmap routing_map when routing_map_format=BITMAP_U8, and
    backward gradients should match the bytemap path exactly."""
    from transformer_engine.jax.router import RoutingMapFormat

    logits = make_logits(num_tokens, num_experts, score_function, dtype)
    expert_bias = None

    fwd_byte = jax.jit(
        partial(
            fused_topk_with_score_function,
            topk=topk,
            score_function=score_function,
            expert_bias=expert_bias,
            routing_map_format=RoutingMapFormat.BYTEMAP,
        )
    )
    fwd_bit = jax.jit(
        partial(
            fused_topk_with_score_function,
            topk=topk,
            score_function=score_function,
            expert_bias=expert_bias,
            routing_map_format=RoutingMapFormat.BITMAP_U8,
        )
    )
    probs_byte, routing_map_byte = fwd_byte(logits)
    probs_bit, routing_map_bit = fwd_bit(logits)

    assert probs_byte.dtype == probs_bit.dtype
    assert jnp.array_equal(probs_byte, probs_bit), "Probs must be identical across formats"

    packed_expected = _bytemap_to_bitmap_u8(routing_map_byte)
    assert routing_map_bit.shape == (
        num_tokens,
        (num_experts + 7) // 8,
    ), f"Bitmap shape {routing_map_bit.shape} != ({num_tokens}, {(num_experts + 7) // 8})"
    assert routing_map_bit.dtype == jnp.uint8
    assert jnp.array_equal(
        routing_map_bit, packed_expected
    ), "Bitmap routing_map disagrees with np.packbits(bytemap, bitorder='little')"

    # Backward parity: grad of probs.sum() must be identical for both formats.
    def loss_byte(logits_):
        p, _ = fused_topk_with_score_function(
            logits_,
            topk,
            score_function=score_function,
            expert_bias=expert_bias,
            routing_map_format=RoutingMapFormat.BYTEMAP,
        )
        return p.sum()

    def loss_bit(logits_):
        p, _ = fused_topk_with_score_function(
            logits_,
            topk,
            score_function=score_function,
            expert_bias=expert_bias,
            routing_map_format=RoutingMapFormat.BITMAP_U8,
        )
        return p.sum()

    grad_byte = jax.jit(jax.grad(loss_byte))(logits)
    grad_bit = jax.jit(jax.grad(loss_bit))(logits)
    assert jnp.allclose(grad_byte, grad_bit, atol=0.0, rtol=0.0), (
        "Backward grad must be bit-identical across routing_map_format; "
        f"max diff = {jnp.abs(grad_byte - grad_bit).max()}"
    )


@pytest_parametrize_wrapper("dtype", DTYPES)
@pytest_parametrize_wrapper(
    "num_tokens,num_experts,topk",
    SCORE_AUX_LOSS_CASES,
)
@pytest_parametrize_wrapper("score_function", SCORE_FUNCTIONS)
def test_score_for_aux_loss_bitmap_vs_bytemap(dtype, num_tokens, num_experts, topk, score_function):
    """compute_aux_scores=True path: bitmap routing_map must equal LSB-packed
    bytemap; scores must be bitwise identical across formats."""
    from transformer_engine.jax.router import RoutingMapFormat

    logits = make_logits(num_tokens, num_experts, score_function, dtype)

    fwd_byte = jax.jit(
        partial(
            fused_topk_with_score_function,
            topk=topk,
            score_function=score_function,
            compute_aux_scores=True,
            routing_map_format=RoutingMapFormat.BYTEMAP,
        )
    )
    fwd_bit = jax.jit(
        partial(
            fused_topk_with_score_function,
            topk=topk,
            score_function=score_function,
            compute_aux_scores=True,
            routing_map_format=RoutingMapFormat.BITMAP_U8,
        )
    )
    scores_byte, routing_map_byte = fwd_byte(logits)
    scores_bit, routing_map_bit = fwd_bit(logits)

    assert jnp.array_equal(scores_byte, scores_bit), "Scores must be identical across formats"
    packed_expected = _bytemap_to_bitmap_u8(routing_map_byte)
    assert routing_map_bit.shape == (num_tokens, (num_experts + 7) // 8)
    assert routing_map_bit.dtype == jnp.uint8
    assert jnp.array_equal(
        routing_map_bit, packed_expected
    ), "Bitmap routing_map (aux-loss path) disagrees with packed bytemap"
