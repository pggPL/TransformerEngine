# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for Float8Tensor"""

from __future__ import annotations
from typing import Optional, Dict, Any, Union, Tuple
import torch

import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from ...constants import TE_DType as torch_to_transformer_engine_dtype

from ..quantized_tensor import _QuantizeFunc, Quantizer


class _FromFloat8Func(torch.autograd.Function):
    """Cast from FP8 to other dtype"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: Float8TensorBase,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        dtype = torch_to_transformer_engine_dtype[dtype]

        # Make sure FP8 data is in expected format
        if tensor._data is not None:
            # Cast from FP8
            return tex.dequantize(tensor, dtype)
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


class Float8TensorBase:
    _data: Optional[torch.Tensor]
    _quantizer: Optional[Quantizer]
    _fp8_dtype: TE_DType
    _scale_inv: torch.Tensor

    # FP8 transpose cache
    _transpose: Optional[torch.Tensor]
    _transpose_invalid: bool

    def __new__(
        cls,
        *args,
        data: Optional[torch.Tensor],
        fp8_scale_inv: torch.Tensor,
        fp8_dtype: TE_DType,
        data_transpose: Optional[torch.Tensor] = None,
        quantizer: Optional[Quantizer] = None,
        **kwargs,
    ):
        instance = super().__new__(cls, *args, **kwargs)
        instance._data = data
        instance._quantizer = quantizer
        instance._fp8_dtype = fp8_dtype
        instance._scale_inv = fp8_scale_inv
        instance._transpose = data_transpose
        instance._transpose_invalid = instance._transpose is None

        return instance

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "data": self._data,
            "fp8_scale_inv": self._scale_inv,
            "fp8_dtype": self._fp8_dtype,
            "data_transpose": self._transpose,
            "quantizer": self._quantizer,
        }

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], Float8TensorBase]:
        """Prepare the tensor base for saving for backward.
        After calling this, this tensor base does not hold any data."""
        tensors = [self._data, self._transpose]
        # self._data = None
        # self._transpose = None # TODO - why
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list."""
        self._data = tensors[0]
        self._transpose = tensors[1]
        return tensors[2:]

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return _FromFloat8Func.forward(None, self, dtype)

    def size(self, *args, **kwargs):
        return self._data.size(*args, **kwargs)

    def __repr__(self):
        return (
            "Float8TensorBase("
            f"fp8_dtype={self._fp8_dtype}, "
            f"scale_inv={self._scale_inv.item()}, "
            f"data={self.dequantize()}"
            ")"
        )
