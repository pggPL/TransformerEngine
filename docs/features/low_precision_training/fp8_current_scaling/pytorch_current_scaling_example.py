# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_CURRENT_SCALING_EXAMPLE

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8CurrentScaling

# Create FP8 Current Scaling recipe
recipe = Float8CurrentScaling()

# Create a simple linear layer
layer = te.Linear(1024, 1024)
optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)

# Training with FP8 Current Scaling
inp = torch.randn(32, 128, 1024, dtype=torch.bfloat16, device='cuda')

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    output = layer(inp)
    loss = output.sum()

loss.backward()
optimizer.step()

# END_CURRENT_SCALING_EXAMPLE

