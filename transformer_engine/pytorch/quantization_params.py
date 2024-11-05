# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Parameters needed for quantization using different recipes."""

import torch
from transformer_engine_torch import DType as TE_DType

class QuantizationParams:
    def __init__(self):
        pass

class Float8Params(QuantizationParams):
    scale: torch.Tensor
    amax: torch.Tensor
    dtype: TE_DType

    def __init__(self,
                 scale: torch.Tensor,
                 amax: torch.Tensor,
                 dtype: TE_DType):
        super().__init__()
        self.scale = scale
        self.amax = amax
        self.dtype = dtype

class QuantizationParamsProxy:
    def __init__(self):
        pass

    def get_quantization_params(self) -> QuantizationParams:
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement get_quantization_params function"
        )

class Float8ParamsProxy(QuantizationParamsProxy):
    def __init__(self,
                 meta,
                 index,
                 dtype):
        super().__init__()
        self.meta = meta
        self.index = index
        self.dtype = dtype

    def get_quantization_params(self) -> QuantizationParams:
        return Float8Params(
                    self.meta.scale[self.index],
                    self.meta.amax[0][self.index],
                    self.dtype)

