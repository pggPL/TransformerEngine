# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine bindings for pyTorch"""
from .module import (
    Linear,
    LayerNormLinear,
    LayerNormMLP,
    TransformerLayer
)

try:
    import torch
    torch._dynamo.config.error_on_nested_jit_trace = False
except: # pylint: disable=bare-except
    pass
