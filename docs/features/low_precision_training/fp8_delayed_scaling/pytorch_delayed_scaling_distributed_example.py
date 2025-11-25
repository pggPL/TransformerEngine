# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_AMAX_REDUCTION_EXAMPLE
import torch.distributed as dist
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling

# Create process group for amax reduction (e.g., all 8 GPUs)
fp8_group = dist.new_group(ranks=[0, 1, 2, 3, 4, 5, 6, 7])

recipe = DelayedScaling(reduce_amax=True)

with te.fp8_autocast(fp8_recipe=recipe, fp8_group=fp8_group):
    output = model(inp)

# END_AMAX_REDUCTION_EXAMPLE

