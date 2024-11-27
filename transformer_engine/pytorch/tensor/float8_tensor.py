# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data"""
from __future__ import annotations
from typing import Optional, Tuple, Iterable

import torch
import transformer_engine_torch as tex

from transformer_engine_torch import DType as TE_DType
from ..utils import devices_match, non_tn_fp8_gemm_supported

from ._internal.float8_tensor_base import Float8TensorBase, _FromFloat8Func
from .quantized_tensor import QuantizedTensor, Quantizer, _IdentityFunc

aten = torch.ops.aten

class Float8Quantizer(Quantizer):

    scale: torch.Tensor
    amax: torch.Tensor
    dtype: TE_DType
    single_usage_sufficient: bool = True

    def __init__(
        self,
        scale: torch.Tensor,
        amax: torch.Tensor,
        fp8_dtype: TE_DType,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.scale = scale
        self.amax = amax
        self.dtype = fp8_dtype

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
    ) -> QuantizedTensor:

        assert isinstance(dst, Float8Tensor)
        # Launch cast kernel
        tex.quantize(src, self, dst);

        # Update FP8 dtype
        dst._fp8_dtype = self.dtype

        return dst

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> Float8Tensor:

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        # Allocate FP8 data
        data = torch.empty(shape, dtype=torch.uint8, device=device)

        # Allocate FP8 data transpose if needed
        data_transpose = None
        if self.columnwise_usage:
            inner_dim = data.size(-1)
            data_transpose = torch.empty(
                inner_dim,
                data.numel() // inner_dim,
                dtype=torch.uint8,
                device=device,
            )

        # Construct FP8 tensor
        return Float8Tensor(
            shape=shape,
            dtype=dtype,
            data=data,
            fp8_dtype=self.fp8_dtype,
            requires_grad=requires_grad,
            data_transpose=data_transpose,
            quantizer=self,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        amin, amax = tensor.aminmax()
        self.amax.copy_(torch.max(-amin, amax))


class Float8Tensor(Float8TensorBase, QuantizedTensor):
    """Experimental tensor class with FP8 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    Parameters
    ----------
    data: torch.Tensor
          Raw FP8 data in a uint8 tensor
    fp8_dtype: transformer_engine_torch.DType, default = kFloat8E4M3
               FP8 format.
    fp8_scale_inv: torch.Tensor
                   Reciprocal of the scaling factor applied when
                   casting to FP8, i.e. the scaling factor that must
                   be applied when casting from FP8 to higher
                   precision. Can be inferred from fp8_meta if
                   provided.
    dtype: torch.dtype, default = torch.float32
           Nominal tensor datatype.

    """

    def __repr__(self,
                 *,
                 tensor_contents = None):
        return (
            "Float8Tensor("
            f"fp8_dtype={self._fp8_dtype}, "
            f"scale_inv={self._scale_inv.item()}, "
            f"data={self.dequantize(dtype=self.dtype)}"
            ")"
        )

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from Float8Tensor

        By default the resulting tensor's dtype is the
        Float8Tensor's nominal dtype.
        """
        # Convert PyTorch dtype to TE dtype
        if dtype is None:
            dtype = self.dtype

        if torch.is_grad_enabled():
            return _FromFloat8Func.apply(self, dtype)
        else:
            return _FromFloat8Func.forward(None, self, dtype)

    def _get_quantizer(self) -> Quantizer:
        """Get builder for quantized tensor

        Quantizer can be used for in-place operations.

        """
        if self._quantizer is not None:
            return self._quantizer
        return Float8Quantizer(
            scale=torch.reciprocal(self._scale_inv),
            amax=torch.empty(1, dtype=torch.float32, device=self.device),
            dtype=self._fp8_dtype,
        )

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Float8Tensor:
        """Update FP8 data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        ### TODO Support noop_flag
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize())
        self._get_quantizer().update_quantized(tensor, self)
        return self

    def detach(self) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return Float8Tensor.make_like(self)

    def _create_transpose(self):
        data = self._data
        if not data.is_contiguous():
            data = data.contiguous()
        self._transpose = tex.fp8_transpose(self._data, self._fp8_dtype, out=self._transpose)
        self._transpose_invalid = False

    def update_usage(self, rowwise_usage=True, columnwise_usage=True):
        assert rowwise_usage or columnwise_usage, \
               "Could not disable all usages of the tensor"
        if rowwise_usage:
            assert self._data is not None, \
                   "Rowwise usage of the tensor was already disabled"
        else:
            if not non_tn_fp8_gemm_supported():
                if self._transpose is None or self._transpose_invalid:
                    self._create_transpose()
                self._data = None
        if columnwise_usage:
            if self._transpose is None or self._transpose_invalid:
                assert self._data is not None, \
                       "The tensor does not hold any data anymore"
                if not non_tn_fp8_gemm_supported():
                    self._create_transpose()
        else:
            self._transpose = None
            self._transpose_invalid = True

    def clone(self) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        assert self._data is not None
        data = self._data.detach().clone()
        data_transpose = None
        if self._transpose is not None:
            data_transpose = self._transpose.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "data": data,
                "data_transpose": data_transpose,
            },
        )

    def view(self, *shape: Tuple[int]) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Float8Tensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if self._data is not None and self._data.is_contiguous(memory_format=memory_format):
            return self
        if (self._transpose is not None and
            self._transpose.is_contiguous(memory_format=memory_format)):
            return self
        raise ValueError("Float8Tensor does not support different memory formats!")

    def _reset_caches(self) -> None:
        """
        Set transpose cache as invalid.
        Should be called after any in-place operation.
        """
        self._transpose_invalid = True

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully.
        """
        self._data = torch.Tensor() if self._data is not None else None
        self._transpose = torch.Tensor() if self._transpose is not None else None
        self._transpose_invalid = True

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # View op
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._data
            out_data = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            out_shape = out_data.size()
            out_transpose = None if tensor._transpose_invalid else tensor._transpose
            if out_transpose is not None:
                out_transpose_shape = out_transpose.size()
                if (
                    out_transpose_shape[0] != out_shape[-1]
                    or out_transpose_shape[1:] != out_shape[:-1]
                ):
                    out_transpose = None
            return Float8Tensor(
                shape=out_shape,
                dtype=tensor.dtype,
                requires_grad=False,
                data=out_data,
                fp8_scale_inv=tensor._scale_inv,
                fp8_dtype=tensor._fp8_dtype,
                data_transpose=out_transpose,
                quantizer=tensor._quantizer,
            )

        # Default case
        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def _make_in_reduce_ex(
        cls,
        data: torch.Tensor,
        fp8_dtype: TE_DType,
        fp8_scale_inv: torch.Tensor,
        dtype: torch.dtype,
    ) -> Float8Tensor:
        """Build Float8Tensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return Float8Tensor(
            data=data,
            fp8_dtype=fp8_dtype,
            fp8_scale_inv=fp8_scale_inv,
            dtype=dtype,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling to remove references to FP8 metadata objects"""
        return (
            Float8Tensor._make_in_reduce_ex,
            (self._data, self._fp8_dtype, self._scale_inv, self.dtype),
        )

    def _get_data(self) -> Float8Tensor:
        """Get tensor data property"""
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes FP8 data if setting from a Float8Tensor. Otherwise
        casts to FP8.

        """

        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device

        # Just copy FP8 data if other tensor is Float8Tensor
        if isinstance(tensor, Float8Tensor):

            # PyTorch tensor attributes
            if (  # pylint: disable=too-many-boolean-expressions
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    Float8Tensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(Float8Tensor, type(self)).data.__set__(self, dummy_tensor)

            # Float8Tensor attributes
            self._data = tensor._data
            self._quantizer = tensor._quantizer
            self._fp8_dtype = tensor._fp8_dtype
            self._transpose = tensor._transpose
            self._transpose_invalid = tensor._transpose_invalid
            return

        # Quantize to FP8
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self.data = self._quantizer.quantize(tensor)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    # Cast to FP8 when setting Float8Tensor.data
    data = property(_get_data, _set_data)

class _ViewFunc(torch.autograd.Function):
    """View function

    View the Float8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8Tensor,
        shape: Optional[list[int]] = None,
    ) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        ctx.shape = tensor.shape
        if shape is None:
            return tensor.detach()
        out_data = tensor._data.view(*shape)
        out_shape = out_data.size()
        out_transpose = None if tensor._transpose_invalid else tensor._transpose
        if out_transpose is not None:
            out_transpose_shape = out_transpose.size()
            if (
                out_transpose_shape[0] != out_shape[-1]
                or out_transpose_shape[1:] != out_shape[:-1]
            ):
                out_transpose = None
        return Float8Tensor(
            shape=out_shape,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
            data=out_data,
            fp8_scale_inv=tensor._scale_inv,
            fp8_dtype=tensor._fp8_dtype,
            data_transpose=out_transpose,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        return grad.reshape(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the Float8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8Tensor,
        shape: Tuple[int],
    ) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        ctx.shape = tensor.shape
        if shape is None:
            return tensor.detach()
        out_data = tensor._data.reshape(*shape)
        out_shape = out_data.size()
        out_transpose = None if tensor._transpose_invalid else tensor._transpose
        if out_transpose is not None:
            out_transpose_shape = out_transpose.size()
            if (
                out_transpose_shape[0] != out_shape[-1]
                or out_transpose_shape[1:] != out_shape[:-1]
            ):
                out_transpose = None
        return Float8Tensor(
            shape=out_shape,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
            data=out_data,
            fp8_scale_inv=tensor._scale_inv,
            fp8_dtype=tensor._fp8_dtype,
            data_transpose=out_transpose,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        return grad.reshape(ctx.shape), None
