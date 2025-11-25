# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_FP8_AUTOCAST

import jax
import jax.numpy as jnp
import optax
import transformer_engine.jax as te
from transformer_engine.jax.flax import TransformerLayer
from transformer_engine.jax.sharding import MeshResource, global_shard_guard
from transformer_engine.jax.quantize import get_delayed_scaling_recipe

# Set up mesh resource and FP8 recipe
recipe = get_delayed_scaling_recipe()

with global_shard_guard(MeshResource()):
    with te.fp8_autocast(enabled=True, recipe=recipe, mesh_resource=MeshResource()):
        # Create layer and initialize
        layer = TransformerLayer(
            hidden_size=1024,
            mlp_hidden_size=4096,
            num_attention_heads=16,
        )
        
        init_key, dropout_key = jax.random.split(jax.random.PRNGKey(0))
        x = jax.random.normal(init_key, (32, 128, 1024), dtype=jnp.bfloat16)
        params = layer.init({'params': init_key, 'dropout': dropout_key}, x)
        
        # Forward and backward pass
        def loss_fn(params):
            output = layer.apply(params, x, rngs={'dropout': dropout_key})
            return output.sum()
        
        loss, grads = jax.value_and_grad(loss_fn)(params)

# END_FP8_AUTOCAST

