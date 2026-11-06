# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MoE resource and activation API contracts without EP kernels."""

import importlib
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from transformer_engine.jax.flax import _MoEBlock
from transformer_engine.jax.moe import (
    RouterComputationInfo,
    SiTUActivation,
    _apply_ffn_activation,
    _moe_mesh_axes,
    _resolve_moe_mesh_resource,
)
from transformer_engine.jax.sharding import MeshResource, global_mesh_resource, global_shard_guard


@pytest.mark.parametrize("betas", [(4.0, 25.0), (2.0, 7.0)])
def test_situ_value_and_both_branch_derivatives(betas):
    gate_beta, up_beta = betas
    gate = jnp.asarray([-10000.0, -12.0, -0.5, 0.0, 0.5, 12.0, 10000.0])
    up = jnp.asarray([10000.0, 40.0, -3.0, 2.0, 3.0, -40.0, -10000.0])
    activation = SiTUActivation(gate_beta, up_beta)
    output, pullback = jax.vjp(lambda g, u: _apply_ffn_activation(activation, g, u), gate, up)
    dg, du = pullback(jnp.ones_like(output))

    g, u = np.asarray(gate, np.float64), np.asarray(up, np.float64)
    sigmoid = 0.5 + 0.5 * np.tanh(g / 2)
    tg, tu = np.tanh(g / gate_beta), np.tanh(u / up_beta)
    capped_gate, capped_up = gate_beta * tg * sigmoid, up_beta * tu
    derivative_gate = (1 - tg**2) * sigmoid + gate_beta * tg * sigmoid * (1 - sigmoid)
    for actual, expected in (
        (output, capped_gate * capped_up),
        (dg, derivative_gate * capped_up),
        (du, capped_gate * (1 - tu**2)),
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_situ_bf16_uses_fp32_before_final_cast():
    gate = jnp.asarray([0.25, 1.5, -3.0, 12.0, 64.0], dtype=jnp.bfloat16)
    up = jnp.asarray([0.75, -2.0, 32.0, 64.0, -128.0], dtype=jnp.bfloat16)
    actual = _apply_ffn_activation(SiTUActivation(), gate, up)
    g, u = np.asarray(gate, np.float64), np.asarray(up, np.float64)
    expected = jnp.asarray(100 * np.tanh(g / 4) / (1 + np.exp(-g)) * np.tanh(u / 25), jnp.bfloat16)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
@pytest.mark.parametrize("field", ["gate_beta", "up_beta"])
def test_situ_rejects_invalid_soft_caps(field, value):
    with pytest.raises(ValueError, match=field):
        SiTUActivation(**{field: value})


@pytest.fixture(autouse=True)
def no_global_resource():
    with global_shard_guard(None):
        yield


def test_resource_required_for_function_and_block():
    module = importlib.import_module("transformer_engine.jax.moe")
    with pytest.raises(ValueError, match="active global_shard_guard"):
        module.moe(None, None, None, None, num_experts=2)
    with pytest.raises(ValueError, match="active global_shard_guard"):
        _MoEBlock().init(jax.random.PRNGKey(0), jnp.ones((1, 1, 4)))


@pytest.mark.parametrize("shape", [(1, 4), (2, 1, 8), (1, 2, 8)])
def test_router_input_shape_is_validated(shape):
    module = importlib.import_module("transformer_engine.jax.moe")
    router = RouterComputationInfo(
        gate_kernel=jnp.ones((8, 2)), num_experts_per_tok=1, router_input=jnp.ones(shape)
    )
    with pytest.raises(ValueError, match="router_input must have shape"):
        module.moe(
            jnp.ones((1, 1, 4)),
            router,
            None,
            None,
            num_experts=2,
            mesh_resource=MeshResource(ep_resource="ep"),
        )


def test_gate_kernel_uses_router_width():
    module = importlib.import_module("transformer_engine.jax.moe")
    router = RouterComputationInfo(
        gate_kernel=jnp.ones((4, 2)), num_experts_per_tok=1, router_input=jnp.ones((1, 1, 8))
    )
    with pytest.raises(ValueError, match="gate_kernel must have shape"):
        module.moe(
            jnp.ones((1, 1, 4)),
            router,
            None,
            None,
            num_experts=2,
            mesh_resource=MeshResource(ep_resource="ep"),
        )


def test_explicit_resource_overrides_global_and_is_snapshotted():
    explicit = MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="ep")
    enclosing = MeshResource(ep_resource="other")
    with global_shard_guard(enclosing):
        resolved, quantize = _resolve_moe_mesh_resource(explicit, True)
        assert global_mesh_resource() is enclosing
    explicit.ep_resource = "changed"
    assert _moe_mesh_axes(resolved) == ("ep", ("dp", "fsdp"))
    assert quantize is True


def test_global_default_and_duplicate_outer_axis():
    with global_shard_guard(
        MeshResource(dp_resource="fsdp", fsdp_resource="fsdp", ep_resource="ep")
    ):
        resource, quantize = _resolve_moe_mesh_resource()
    assert _moe_mesh_axes(resource) == ("ep", ("fsdp",))
    assert quantize is False


def test_ep_partitioning_retains_bind_time_resources():
    from transformer_engine.jax.cpp_extensions import ep
    from jax.sharding import PartitionSpec as P

    with global_shard_guard(MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="ep")):
        captured = ep._capture_ep_resource_axes()
    spec = P(("dp", "fsdp", "ep"), None, None)
    assert ep._leading_axis_ok(spec, captured) == (True, "ep", ("dp", "fsdp"))
    assert ep._ep_output_spec(None, None, resource_axes=captured) == spec
    assert ep._ep_spec_ok(spec, 2, resource_axes=captured)
    assert not ep._ep_spec_ok(P(("wrong", "ep"), None, None), 2, resource_axes=captured)


def test_fused_ep_partitioning_retains_bind_time_resources(monkeypatch):
    from types import SimpleNamespace
    from jax.sharding import NamedSharding, PartitionSpec as P
    from transformer_engine.jax.cpp_extensions import ep

    captured = {}

    def bind(*args, **kwargs):
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(ep.EpPrepareAndDispatchPrimitive.outer_primitive, "bind", bind)
    cfg = SimpleNamespace(top_k=1, dispatch_output_per_expert_alignment=256, topk_idx_dtype=0)
    with global_shard_guard(MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="ep")):
        inspect.unwrap(ep._ep_fused_prepare_and_dispatch_fwd)(cfg, None, None, None, 256)

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1), ("dp", "fsdp", "ep"))
    sharding = NamedSharding(mesh, P(("dp", "fsdp", "ep"), None, None))
    args = [SimpleNamespace(sharding=sharding)] * 3
    _, _, outputs, _ = ep.EpPrepareAndDispatchPrimitive.partition(
        captured["top_k"],
        captured["dispatch_output_per_expert_alignment"],
        captured["topk_idx_dtype"],
        captured["recv_capacity_per_rank"],
        captured["is_outer"],
        captured["resource_axes"],
        mesh,
        args,
        None,
    )
    assert all(output.spec[0] == ("dp", "fsdp", "ep") for output in outputs)


@pytest.mark.parametrize(
    "resource,quantize,error",
    [
        (MeshResource(), False, "ep_resource"),
        (MeshResource(ep_resource="ep", dp_resource="ep"), False, "distinct"),
        (MeshResource(ep_resource="ep"), True, "fsdp_resource"),
    ],
)
def test_invalid_resource(resource, quantize, error):
    with pytest.raises(ValueError, match=error):
        _resolve_moe_mesh_resource(resource, quantize)


def test_bool_and_resource_types():
    with pytest.raises(TypeError, match="must be a bool"):
        _resolve_moe_mesh_resource(MeshResource(ep_resource="ep"), "true")
    with pytest.raises(TypeError, match="must be a MeshResource"):
        _resolve_moe_mesh_resource("ep")


@pytest.mark.parametrize("quantize", [False, True])
def test_public_api_uses_selected_resource(monkeypatch, quantize):
    module = importlib.import_module("transformer_engine.jax.moe")
    signature = inspect.signature(module._moe)
    captured = {}

    def fake_vjp(*args):
        captured.update(signature.bind(*args).arguments)
        assert global_mesh_resource().ep_resource == "ep"
        return args[0], None, jnp.zeros((1,), jnp.int32), jnp.zeros((2,), jnp.int32)

    monkeypatch.setattr(module, "_moe", fake_vjp)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1), ("dp", "fsdp", "ep"))
    kwargs = dict(
        num_experts=2,
        mesh_resource=MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="ep"),
        quant_before_fsdp_ag=quantize,
    )
    with jax.set_mesh(mesh):
        module.moe(
            jnp.ones((1, 1, 4)),
            RouterComputationInfo(gate_kernel=jnp.ones((4, 2)), num_experts_per_tok=1),
            jnp.ones((2, 4, 8)),
            jnp.ones((2, 4, 4)),
            **kwargs,
        )
    assert _moe_mesh_axes(captured["mesh_resource"]) == ("ep", ("dp", "fsdp"))
    assert captured["quant_before_fsdp_ag"] is quantize
    assert "ep_axis" not in captured
    with pytest.raises(AssertionError, match="Global mesh resource is not set"):
        global_mesh_resource()


@pytest.mark.parametrize("quantize", [False, True])
def test_flax_block_forwards_boolean(monkeypatch, quantize):
    module = importlib.import_module("transformer_engine.jax.flax.moe")
    resource = MeshResource(fsdp_resource="fsdp", ep_resource="ep")
    captured = {}

    def fake_moe(inputs, *args, **kwargs):
        captured.update(kwargs)
        return inputs, None, jnp.zeros((1,), jnp.int32), jnp.zeros((2,), jnp.int32)

    monkeypatch.setattr(module, "moe", fake_moe)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("fsdp", "ep"))
    block = _MoEBlock(
        num_experts=2,
        intermediate_size=4,
        mesh_resource=resource,
        quant_before_fsdp_ag=quantize,
    )
    with jax.set_mesh(mesh):
        block.init(jax.random.PRNGKey(0), jnp.ones((1, 1, 4)))
    assert captured["quant_before_fsdp_ag"] is quantize
    assert _moe_mesh_axes(captured["mesh_resource"]) == ("ep", ("fsdp",))
