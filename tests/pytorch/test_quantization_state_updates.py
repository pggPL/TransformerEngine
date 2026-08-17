# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for delayed-scaling quantization state update scheduling.

The backward amax reduction must run exactly once per training step,
regardless of activation checkpointing (which re-runs forwards) or the
topology of the autograd graph (branches, unused outputs).
"""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling
from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

HIDDEN = 128
BATCH = 32
STEPS = 3


class _UpdateCounter:
    """Counts reduce_and_update_quantization_state calls by direction."""

    def __init__(self):
        self.forward = 0
        self.backward = 0
        self._original = None

    def __enter__(self):
        self._original = FP8GlobalStateManager.reduce_and_update_quantization_state.__func__
        original = self._original
        counter = self

        def counted(cls, forward=True):
            if forward:
                counter.forward += 1
            else:
                counter.backward += 1
            return original(cls, forward=forward)

        FP8GlobalStateManager.reduce_and_update_quantization_state = classmethod(counted)
        return self

    def __exit__(self, *exc):
        FP8GlobalStateManager.reduce_and_update_quantization_state = classmethod(self._original)


def _make_model(num_layers=3, seed=1234):
    torch.manual_seed(seed)
    return torch.nn.ModuleList(
        [te.Linear(HIDDEN, HIDDEN, bias=True).cuda() for _ in range(num_layers)]
    )


def _run_layers(layers, x):
    for layer in layers:
        x = layer(x)
    return x


def _train_step(model, x, forward_fn, recipe):
    with te.autocast(enabled=True, recipe=recipe):
        out = forward_fn(model, x)
    loss = out.float().sum()
    loss.backward()
    return loss


def _forward_plain(model, x):
    return _run_layers(model, x)


def _forward_reentrant(model, x):
    return te_checkpoint(_run_layers, model, x, use_reentrant=True)


def _forward_non_reentrant(model, x):
    return te_checkpoint(_run_layers, model, x, use_reentrant=False)


def _forward_per_layer_reentrant(model, x):
    for layer in model:
        x = te_checkpoint(layer, x, use_reentrant=True)
    return x


def _forward_per_layer_non_reentrant(model, x):
    for layer in model:
        x = te_checkpoint(layer, x, use_reentrant=False)
    return x


def _forward_nested(model, x):
    def inner(x):
        return te_checkpoint(model[1], x, use_reentrant=True)

    def outer(x):
        x = model[0](x)
        x = inner(x)
        return model[2](x)

    return te_checkpoint(outer, x, use_reentrant=True)


def _forward_torch_checkpoint(model, x):
    return torch.utils.checkpoint.checkpoint(_run_layers, model, x, use_reentrant=False)


FORWARD_FNS = {
    "plain": _forward_plain,
    "reentrant": _forward_reentrant,
    "non_reentrant": _forward_non_reentrant,
    "per_layer_reentrant": _forward_per_layer_reentrant,
    "per_layer_non_reentrant": _forward_per_layer_non_reentrant,
    "nested": _forward_nested,
    "torch_checkpoint": _forward_torch_checkpoint,
}


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("mode", FORWARD_FNS.keys())
def test_single_update_per_step(mode):
    """Exactly one forward and one backward state update per training step."""
    forward_fn = FORWARD_FNS[mode]
    model = _make_model()
    recipe = DelayedScaling()

    with _UpdateCounter() as counter:
        for step in range(STEPS):
            x = torch.randn(BATCH, HIDDEN, device="cuda", requires_grad=True)
            _train_step(model, x, forward_fn, recipe)
            # The backward update of this step is flushed at the next
            # top-level autocast entry; flush explicitly to count it here.
            FP8GlobalStateManager.flush_backward_quantization_update()
            assert counter.forward == step + 1, f"{mode}: duplicate/missing forward update"
            assert counter.backward == step + 1, f"{mode}: duplicate/missing backward update"


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_flush_at_next_autocast_entry():
    """Without an explicit flush, the update runs when the next autocast begins."""
    model = _make_model()
    recipe = DelayedScaling()

    with _UpdateCounter() as counter:
        x = torch.randn(BATCH, HIDDEN, device="cuda", requires_grad=True)
        _train_step(model, x, _forward_plain, recipe)
        assert counter.backward == 0
        x = torch.randn(BATCH, HIDDEN, device="cuda", requires_grad=True)
        _train_step(model, x, _forward_plain, recipe)
        assert counter.backward == 1
    FP8GlobalStateManager.flush_backward_quantization_update()


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("checkpoint_first_branch", [True, False])
def test_branched_graph(checkpoint_first_branch):
    """Two checkpointed sibling branches merging into one loss."""
    branch_a = _make_model(num_layers=2, seed=1)
    branch_b = _make_model(num_layers=2, seed=2)
    recipe = DelayedScaling()

    with _UpdateCounter() as counter:
        for step in range(STEPS):
            x = torch.randn(BATCH, HIDDEN, device="cuda", requires_grad=True)
            with te.autocast(enabled=True, recipe=recipe):
                if checkpoint_first_branch:
                    ya = te_checkpoint(_run_layers, branch_a, x, use_reentrant=True)
                else:
                    ya = _run_layers(branch_a, x)
                yb = te_checkpoint(_run_layers, branch_b, x, use_reentrant=True)
                out = ya + yb
            loss = out.float().sum()
            loss.backward()
            FP8GlobalStateManager.flush_backward_quantization_update()
            assert counter.forward == step + 1
            assert counter.backward == step + 1
            assert x.grad is not None and torch.isfinite(x.grad).all()
            x.grad = None


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_checkpointed_branch_without_backward():
    """A checkpointed branch whose output never receives a gradient must not
    strand the backward update (regression test for first-module ownership)."""
    used = _make_model(num_layers=2, seed=1)
    unused = _make_model(num_layers=2, seed=2)
    recipe = DelayedScaling()

    with _UpdateCounter() as counter:
        for step in range(STEPS):
            x = torch.randn(BATCH, HIDDEN, device="cuda", requires_grad=True)
            with te.autocast(enabled=True, recipe=recipe):
                # The unused branch runs first, so any "first module owns the
                # backward update" scheme would assign ownership to a frame
                # whose backward never executes.
                y_unused = te_checkpoint(_run_layers, unused, x, use_reentrant=True)
                y = _run_layers(used, x)
            loss = y.float().sum()
            loss.backward()
            del y_unused
            FP8GlobalStateManager.flush_backward_quantization_update()
            assert counter.backward == step + 1, "backward update was lost"
            x.grad = None


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize(
    "mode", ["reentrant", "non_reentrant", "per_layer_reentrant", "nested", "torch_checkpoint"]
)
def test_checkpointing_matches_plain_numerics(mode):
    """Delayed-scaling state must evolve identically with and without
    activation checkpointing (duplicate updates would advance it faster)."""
    forward_fn = FORWARD_FNS[mode]
    recipe = DelayedScaling()

    def train(forward):
        model = _make_model(seed=99)
        losses = []
        torch.manual_seed(777)
        for _ in range(STEPS + 1):
            x = torch.randn(BATCH, HIDDEN, device="cuda", requires_grad=True)
            losses.append(_train_step(model, x, forward, recipe).detach())
        FP8GlobalStateManager.flush_backward_quantization_update()
        state = []
        for layer in model:
            for key in ("scaling_fwd", "scaling_bwd"):
                meta = layer.fp8_meta[key]
                state.append((meta.scale.clone(), meta.amax_history.clone()))
        return losses, state

    losses_ref, state_ref = train(_forward_plain)
    losses_ckpt, state_ckpt = train(forward_fn)

    for step, (l_ref, l_ckpt) in enumerate(zip(losses_ref, losses_ckpt)):
        torch.testing.assert_close(l_ckpt, l_ref, rtol=0, atol=0, msg=f"loss diverged @ {step}")
    for (scale_ref, hist_ref), (scale_ckpt, hist_ckpt) in zip(state_ref, state_ckpt):
        assert torch.equal(scale_ckpt, scale_ref), "scale diverged under checkpointing"
        assert torch.equal(hist_ckpt, hist_ref), "amax history diverged under checkpointing"


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_exception_in_checkpointed_forward():
    """A failing checkpointed forward must not corrupt update scheduling."""
    model = _make_model()
    recipe = DelayedScaling()

    def failing(model, x):
        def fn(x):
            model[0](x)
            raise RuntimeError("boom")

        return te_checkpoint(fn, x, use_reentrant=True)

    with _UpdateCounter() as counter:
        x = torch.randn(BATCH, HIDDEN, device="cuda", requires_grad=True)
        with pytest.raises(RuntimeError, match="boom"):
            _train_step(model, x, failing, recipe)

        x = torch.randn(BATCH, HIDDEN, device="cuda", requires_grad=True)
        _train_step(model, x, _forward_reentrant, recipe)
        FP8GlobalStateManager.flush_backward_quantization_update()
        assert counter.backward == 1
        assert torch.isfinite(x.grad).all()
