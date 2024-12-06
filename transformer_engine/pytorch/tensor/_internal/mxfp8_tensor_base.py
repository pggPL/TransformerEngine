# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for MXFP8Tensor"""

from __future__ import annotations
from typing import Optional, Dict, Any, Union, Tuple
import torch

import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from ...constants import TE_DType as torch_to_transformer_engine_dtype

from ..quantized_tensor import _QuantizeFunc, Quantizer


class _FromMXFP8Func(torch.autograd.Function):
    """Cast from MXFP8 to other dtype"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: MXFP8TensorBase,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        dtype = torch_to_transformer_engine_dtype[dtype]

        # Make sure FP8 data is in expected format
        if tensor._data is not None:
            raise NotImplementedError("MXFP8 dequantize is WIP!")  # TODO
        else:
            raise NotImplementedError("Casting back from the transpose not implemented yet!")

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None

class MXFP8TensorBase:
    _rowwise_data: Optional[torch.Tensor]
    _columnwise_data: Optional[torch.Tensor]
    _quantizer: Optional[Quantizer]
    _fp8_dtype: TE_DType
    _rowwise_scale_inv: torch.Tensor
    _columnwise_scale_inv: torch.Tensor

    def __new__(cls,
                *args,
                rowwise_data: Optional[torch.Tensor],
                rowwise_scale_inv: torch.Tensor,
                columnwise_data: Optional[torch.Tensor],
                columnwise_scale_inv: torch.Tensor,
                fp8_dtype: TE_DType,
                quantizer: Optional[Quantizer] = None,
                **kwargs
    ):
        instance = super().__new__(cls, *args, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._quantizer = quantizer
        instance._fp8_dtype = fp8_dtype
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv

        return instance

    def get_metadata(self) -> Dict[str, Any]:
        return {"rowwise_data": self._rowwise_data,
                "rowwise_scale_inv": self._rowwise_scale_inv,
                "columnwise_data": self._columnwise_data,
                "columnwise_scale_inv": self._columnwise_scale_inv,
                "fp8_dtype": self._fp8_dtype,
                "quantizer": self._quantizer,
        }

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], MXFP8TensorBase]:
        """Prepare the tensor base for saving for backward.
        After calling this, this tensor base does not hold any data."""
        tensors = [self._rowwise_data, self._columnwise_data]
        # self._rowwise_data = None
        # self._columnwise_data = None
        return tensors, self

    def restore_from_saved(self,
                           tensors: list[Optional[torch.Tensor]]) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list."""
        self._rowwise_data = tensors[0]
        self._columnwise_data = tensors[1]
        return tensors[2:]

    def dequantize(self,
                   *,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return _FromMXFP8Func.forward(None, self, dtype)

    def __repr__(self):
        # TODO Do we need to print both?
        # TODO Figure out dequantize

        data_rowwise, data_colwise = self.dequantize()

        return (
            "MXFP8TensorBase("
            f"fp8_dtype={self._fp8_dtype}, "
            f"rowwise_scale_inv={self._rowwise_scale_inv.item()}, "
            f"rowwise_scaled_data={data_rowwise}"
            f"columnwise_scale_inv={self._columnwise_scale_inv.item()}, "
            f"columnwise_scaled_data={data_colwise}"
            ")"
        )
