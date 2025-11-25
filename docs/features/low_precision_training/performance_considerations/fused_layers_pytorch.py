# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_FUSED_LAYERS

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling

# Example 1: Separate LayerNorm and Linear layers
layer_norm = te.LayerNorm(1024)
linear = te.Linear(1024, 1024)

inp = torch.randn(32, 128, 1024, dtype=torch.bfloat16, device='cuda')

# Two separate operations: LayerNorm produces FP32, then Linear quantizes it
normalized = layer_norm(inp)
output_separate = linear(normalized)

# Example 2: Fused LayerNormLinear layer
fused_layer = te.LayerNormLinear(1024, 1024)

# Single operation: LayerNorm output is directly quantized
recipe = DelayedScaling()
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    output_fused = fused_layer(inp)

# The fused layer is more efficient as it avoids redundant quantization

# END_FUSED_LAYERS

