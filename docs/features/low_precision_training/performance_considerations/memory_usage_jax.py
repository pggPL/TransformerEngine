# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import jax

# Requires Ada (SM89) or Hopper (SM90), different results on Blackwell+
cc = jax.devices()[0].device_kind
assert ("RTX 40" in cc or "L40" in cc or "H100" in cc or "H200" in cc or "GH" in cc) \
    and "B100" not in cc and "B200" not in cc and "GB" not in cc, \
    "This example requires SM89 (Ada) or SM90 (Hopper)"

# START_MEMORY_USAGE_1

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral

# Initialize a dense layer with high precision parameters
key = jax.random.PRNGKey(0)

layer = DenseGeneral(features=1024, dtype=jnp.bfloat16)
x = jax.random.normal(key, (1024, 1024), dtype=jnp.bfloat16)
params = layer.init(key, x)

# Calculate layer size (1024 * 1024 * 2 bytes for BF16)
param_count = 1024 * 1024
layer_size = param_count * 2 / (1024**2)
print(f"Layer size: {layer_size:.2f} MB")

# Forward pass
output = layer.apply(params, x)

# Memory after forward: weight (2 MB) + input (2 MB) + output (2 MB) = 6 MB
print(f"Memory usage after forward pass: 6.00 MB")

# END_MEMORY_USAGE_1

# START_MEMORY_USAGE_2

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.common.recipe import DelayedScaling

key = jax.random.PRNGKey(0)
recipe = DelayedScaling()

# Initialize layer with BF16 parameters
layer = DenseGeneral(features=1024, dtype=jnp.bfloat16)
x = jax.random.normal(key, (1024, 1024), dtype=jnp.bfloat16)
params = layer.init(key, x)

# Forward pass with FP8 autocast
with te.autocast(enabled=True, recipe=recipe):
    output = layer.apply(params, x)

# Memory usage: 2 MB (weight) + 1 MB (weight in FP8) + 2 MB (input) + 1 MB (input in FP8) + 2 MB (output) = 8 MB
print(f"Memory after forward pass: 8.00 MB")

# END_MEMORY_USAGE_2

# START_MEMORY_USAGE_3

import jax
import jax.numpy as jnp
import transformer_engine.jax as te
from transformer_engine.jax.flax import DenseGeneral
from transformer_engine.common.recipe import DelayedScaling

key = jax.random.PRNGKey(0)
recipe = DelayedScaling()

# Initialize layer with FP8 autocast - stores weights in FP8
with te.autocast(enabled=True, recipe=recipe):
    layer = DenseGeneral(features=1024, dtype=jnp.float8_e4m3fn)
    x = jax.random.normal(key, (1024, 1024), dtype=jnp.bfloat16)
    params = layer.init(key, x)

    # Layer size with FP8 weights (1024 * 1024 * 1 byte + scaling factors)
    param_count = 1024 * 1024
    layer_size_fp8 = param_count * 1 / (1024**2)
    print(f"Layer size: {layer_size_fp8:.2f} MB")

    # Forward pass
    output = layer.apply(params, x)

    # Memory: 1 MB (weight in FP8) + 2 MB (input) + 1 MB (input in FP8) + 2 MB (output) = 6 MB
    print(f"Memory after forward pass: 6.00 MB")

# END_MEMORY_USAGE_3

