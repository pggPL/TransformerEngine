# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

# Requires Ada (SM89) or Hopper (SM90), different results on Blackwell+
cc = torch.cuda.get_device_capability()
assert cc[0] == 8 and cc[1] >= 9 or cc[0] == 9, "This example requires SM89 (Ada) or SM90 (Hopper)"

# START_MEMORY_USAGE_1
import torch
import transformer_engine.pytorch as te

init_memory = torch.cuda.memory_allocated()
layer = te.Linear(1024, 1024, params_dtype=torch.bfloat16)
memory = torch.cuda.memory_allocated() - init_memory
print(f"Layer size: {memory/1024**2:.2f} MB")

inp = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
out = layer(inp)
mem_after_forward = torch.cuda.memory_allocated() - init_memory
print(f"Memory usage after forward pass: {mem_after_forward/1024**2:.2f} MB")
# END_MEMORY_USAGE_1

# START_MEMORY_USAGE_2
import torch
import transformer_engine.pytorch as te

init_memory = torch.cuda.memory_allocated()
layer = te.Linear(1024, 1024, params_dtype=torch.bfloat16)
inp = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')

with te.fp8_autocast(enabled=True):
    out = layer(inp)
mem_after_forward = torch.cuda.memory_allocated() - init_memory
print(f"Memory after forward pass: {mem_after_forward/1024**2:.2f} MB")
# END_MEMORY_USAGE_2

# START_MEMORY_USAGE_3
import torch
import transformer_engine.pytorch as te

init_memory = torch.cuda.memory_allocated()

# FP8 forward and backward with FP8 weights
with te.fp8_model_init(enabled=True), torch.no_grad():
    layer_fp8 = te.Linear(1024, 1024, params_dtype=torch.bfloat16)
memory = torch.cuda.memory_allocated() - init_memory
print(f"Layer size: {memory/1024**2:.2f} MB")

inp = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
with te.fp8_autocast(enabled=True):
    out = layer_fp8(inp)

mem_after_forward = torch.cuda.memory_allocated() - init_memory
print(f"Memory after forward pass: {mem_after_forward/1024**2:.2f} MB")
# END_MEMORY_USAGE_3

# START_SAVE_ORIGINAL_INPUT
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8CurrentScaling

recipe = Float8CurrentScaling()

def residual_block(layer, inp):
    """Residual connection: input is saved for addition after linear."""
    out = layer(inp)
    return out + inp  # inp must be kept for this addition

for use_save_original in [False, True]:
    layer = te.Linear(1024, 1024, params_dtype=torch.bfloat16, save_original_input=use_save_original)
    inp = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    
    torch.cuda.reset_peak_memory_stats()
    with te.fp8_autocast(enabled=True, recipe=recipe):
        out = residual_block(layer, inp)
    out.sum().backward()
    
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"save_original_input={use_save_original}: {peak:.1f} MB")
# END_SAVE_ORIGINAL_INPUT
