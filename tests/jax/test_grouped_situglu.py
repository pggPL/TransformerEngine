# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Native grouped SiTU values, derivatives and inactive-capacity isolation."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transformer_engine.jax import grouped_situglu


def reference(x, counts, beta1, beta2):
    active = jnp.arange(x.shape[0]) < counts.sum()
    safe_x = jnp.where(active[:, None], x, 0).astype(jnp.float32)
    gate, up = jnp.split(safe_x, 2, axis=-1)
    result = beta1 * jnp.tanh(gate / beta1) * jax.nn.sigmoid(gate)
    result *= beta2 * jnp.tanh(up / beta2)
    return jnp.where(active[:, None], result.astype(x.dtype), 0)


def reference_backward_fp64(x, dy, active, beta1, beta2):
    gate, up = np.split(np.asarray(x, np.float64)[:active], 2, axis=-1)
    upstream = np.asarray(dy, np.float64)[:active]
    tg, tu = np.tanh(gate / beta1), np.tanh(up / beta2)
    sigmoid = 1 / (1 + np.exp(-gate))
    dx = np.zeros(x.shape, np.float64)
    hidden = x.shape[1] // 2
    dx[:active, :hidden] = (
        upstream * ((1 - tg**2) * sigmoid + beta1 * tg * sigmoid * (1 - sigmoid)) * beta2 * tu
    )
    dx[:active, hidden:] = upstream * beta1 * tg * sigmoid * (1 - tu**2)
    return dx


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("caps", [(4.0, 25.0), (2.0, 8.0)])
@pytest.mark.parametrize(
    "capacity,hidden,counts",
    [
        (13, 6, [3, 0, 5]),
        (256, 128, [128, 128]),
        (513, 3584, [128, 0, 129]),
        (7, 3, [0] * 257 + [3, 4]),
        (9, 10, [0, 0]),
        (0, 32, [0, 0]),
    ],
)
def test_grouped_situ_forward_and_backward(dtype, caps, capacity, hidden, counts):
    rng = np.random.default_rng(72)
    host = rng.uniform(-30, 30, (capacity, 2 * hidden)).astype(np.float32)
    active = sum(counts)
    host[active:] = np.nan
    x = jnp.asarray(host, dtype)
    counts = jnp.asarray(counts, jnp.int32)
    upstream = rng.normal(size=(capacity, hidden)).astype(np.float32)
    upstream[active:] = np.nan
    dy = jnp.asarray(upstream, dtype)
    fn = partial(grouped_situglu, beta1=caps[0], beta2=caps[1])
    ref = partial(reference, beta1=caps[0], beta2=caps[1])
    y = jax.jit(fn)(x, counts)
    _, pullback = jax.vjp(lambda inputs: fn(inputs, counts), x)
    dx = jax.jit(pullback)(dy)[0]
    expected_dx = reference_backward_fp64(x, dy, active, *caps)
    tolerance = {jnp.float32: 2.0e-5, jnp.float16: 3.0e-3, jnp.bfloat16: 2.0e-2}[dtype]
    np.testing.assert_allclose(
        y.astype(jnp.float32), ref(x, counts).astype(jnp.float32), rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(dx.astype(jnp.float32), expected_dx, rtol=tolerance, atol=tolerance)
    assert np.isfinite(np.asarray(y, dtype=np.float32)).all()
    assert np.isfinite(np.asarray(dx, dtype=np.float32)).all()
    np.testing.assert_array_equal(np.asarray(y[active:], dtype=np.float32), 0)
    np.testing.assert_array_equal(np.asarray(dx[active:], dtype=np.float32), 0)


def test_grouped_situ_dynamic_counts_replay_and_checkpoint():
    x = jnp.arange(13 * 24, dtype=jnp.float32).reshape(13, 24) / 30 - 5
    compiled = jax.jit(
        grouped_situglu, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"}
    )
    counts = jnp.array([3, 0, 5], jnp.int32)
    baseline = compiled(x, counts)
    poisoned = x.at[8:].set(jnp.nan)
    np.testing.assert_array_equal(baseline, compiled(poisoned, counts))
    changed = jnp.array([1, 2, 2], jnp.int32)
    np.testing.assert_allclose(
        compiled(x, changed), reference(x, changed, 4.0, 25.0), rtol=2.0e-5, atol=2.0e-5
    )
    loss = lambda inputs: jnp.sum(grouped_situglu(inputs, counts))
    np.testing.assert_array_equal(
        jax.jit(jax.grad(jax.checkpoint(loss)))(poisoned), jax.jit(jax.grad(loss))(poisoned)
    )
    hlo = compiled.lower(x, counts).compile().as_text()
    assert "te_grouped_situglu_ffi" in hlo


def test_grouped_situ_vmap():
    x = jnp.arange(2 * 9 * 16, dtype=jnp.float32).reshape(2, 9, 16) / 20 - 5
    counts = jnp.array([[2, 0, 3], [1, 1, 7]], jnp.int32)
    actual = jax.jit(jax.vmap(grouped_situglu))(x, counts)
    expected = jax.vmap(partial(reference, beta1=4.0, beta2=25.0))(x, counts)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5)
    actual_grad = jax.jit(jax.grad(lambda inputs: jax.vmap(grouped_situglu)(inputs, counts).sum()))(
        x
    )
    expected_grad = jax.grad(
        lambda inputs: jax.vmap(partial(reference, beta1=4.0, beta2=25.0))(inputs, counts).sum()
    )(x)
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.parametrize("name", ["beta1", "beta2"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), 1.0e100, 1.0e-100])
def test_grouped_situ_invalid_caps(name, value):
    with pytest.raises(ValueError, match=name):
        grouped_situglu(jnp.ones((3, 16)), jnp.array([3], jnp.int32), **{name: value})


@pytest.mark.parametrize(
    "shape,counts,error",
    [
        ((3, 15), [3], ValueError),
        ((3, 0), [3], ValueError),
        ((2, 3, 16), [3], ValueError),
        ((3, 16), [], ValueError),
    ],
)
def test_grouped_situ_invalid_shapes(shape, counts, error):
    with pytest.raises(error):
        grouped_situglu(jnp.ones(shape), jnp.asarray(counts, jnp.int32))
