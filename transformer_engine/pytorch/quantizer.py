# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Quantization metadata class"""

from transformer_engine.common.recipe import (
        Recipe, DelayedScaling, BlockScaling
)
import torch

from .fp8 import get_fp8_te_dtype
from .tensor import QuantizedTensor, Float8Tensor
from .tensor.float8_tensor import Float8ParamsProxy
from .tensor._internal.float8_tensor_base import Float8TensorBase

from .quantization_params import Float8Params

class Quantizer:
    def __init__(self,
                 recipe: Recipe,
                 num_tensors: int,
                 forward: bool):
        self.single_usage_sufficient = False
        if isinstance(recipe, DelayedScaling):
            self.recipe_type = DelayedScaling
            self.scale = torch.ones(num_tensors, dtype=torch.float32, device="cuda")
            self.amax_history = torch.zeros(
                recipe.amax_history_len,
                num_tensors,
                dtype=torch.float32,
                device="cuda",
            )
            self.fp8_type = get_fp8_te_dtype(recipe, forward)
            self.single_usage_sufficient = True
        elif isinstance(recipe, BlockScaling):
            self.recipe_type = BlockScaling
            self.fp8_type = get_fp8_te_dtype(recipe, forward)
        else:
            raise ValueError(f"Unknown recipe type {type(recipe)}.")

    def quantize(self,
                 tensor: torch.Tensor,
                 index: int,
                 *,
                 rowwise: bool = True,
                 columnwise: bool = True,
                 internal: bool = False) -> QuantizedTensor:
        if self.recipe_type == DelayedScaling:
            proxy = Float8ParamsProxy(self, index, self.fp8_type)
            if internal:
                return Float8TensorBase.quantize(tensor,
                                                 self.get_quantization_params(index),
                                                 rowwise_usage=rowwise,
                                                 columnwise_usage=columnwise,
                                                 proxy=proxy)
            else:
                return Float8Tensor.quantize(tensor,
                                             self.get_quantization_params(index),
                                             rowwise_usage=rowwise,
                                             columnwise_usage=columnwise,
                                             proxy=proxy)
        raise NotImplementedError("Not implemented yet!")

    def get_quantization_params(self,
                                index: int):
        # Could be cached
        if self.recipe_type == DelayedScaling:
            return Float8Params(scale=self.scale[index],
                                amax=self.amax_history[0][index],
                                dtype=self.fp8_type)
        raise NotImplementedError("Not implemented yet!")

    def calibrate(self,
                  tensor: torch.Tensor,
                  index: int):
        if self.recipe_type == DelayedScaling:
            amin, amax = tensor.aminmax()
            self.amax_history[0][index] = torch.max(-amin, amax).float()
            return
        raise NotImplementedError("Not implemented yet!")
