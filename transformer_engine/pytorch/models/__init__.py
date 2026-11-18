# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Model-specific transformer layers composed from Transformer Engine modules."""

from transformer_engine.pytorch.models.deepseek_v3 import (
    DeepSeekV3Layer,
    DeepSeekV3MoE,
    MultiLatentAttention,
)
from transformer_engine.pytorch.models.kimi_k3 import (
    KimiK3Layer,
    KimiK3MoE,
    KimiLinearAttention,
)

__all__ = [
    "DeepSeekV3Layer",
    "DeepSeekV3MoE",
    "MultiLatentAttention",
    "KimiK3Layer",
    "KimiK3MoE",
    "KimiLinearAttention",
]
