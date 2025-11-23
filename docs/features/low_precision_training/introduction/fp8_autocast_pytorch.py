# START_FP8_AUTOCAST

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling

recipe = DelayedScaling()
layer = te.Linear(1024, 1024)
inp = torch.randn(32, 1024, dtype=torch.float32, device='cuda')

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    output = layer(inp)
loss = output.sum()
loss.backward()

# END_FP8_AUTOCAST

