# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

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

