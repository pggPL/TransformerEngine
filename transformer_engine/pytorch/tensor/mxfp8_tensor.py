# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data"""
from __future__ import annotations
import math
from typing import Optional, Tuple, Iterable

import torch
import transformer_engine_torch as tex

from transformer_engine_torch import DType as TE_DType
from ..cpp_extensions.transpose import fp8_cast_transpose_fused
from ..cpp_extensions.cast import cast_to_fp8
from ..constant import MXFP8_BLOCK_SCALING_SIZE
from ..fp8 import FP8GlobalStateManager
from ..utils import devices_match, non_tn_fp8_gemm_supported

from ._internal.mxfp8_tensor_base import MXFP8TensorBase, _FromMXFP8Func
from .quantized_tensor import QuantizedTensor, Quantizer, _IdentityFunc

aten = torch.ops.aten

class MXFP8Quantizer(Quantizer):

    dtype: TE_DType

    def __init__(
        self,
        fp8_dtype: TE_DType,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp8_dtype

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
    ) -> QuantizedTensor:

        assert isinstance(dst, MXFP8Tensor), f"Cannot store quantized MXFP8 in {type(dst)} type."

        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Launch cast kernel
        tex.quantize(src, self, dst)

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
    ) -> MXFP8Tensor:

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        # Allocate FP8 data
        data = torch.empty(shape, dtype=torch.uint8, device=device)
        scale_inv = torch.empty(math.prod(shape[:-1]) // MXFP8_BLOCK_SCALING_SIZE,
                                shape[-1], dtype=torch.uint8, device=device)

        # Allocate FP8 data transpose if needed
        columnwise_data = None
        columnwise_scale_inv = None
        if self.columnwise_usage:
            columnwise_data = torch.empty_like(data)
            columnwise_scale_inv = torch.empty(math.prod(shape[:-1]),
                                               shape[-1] // MXFP8_BLOCK_SCALING_SIZE,
                                               dtype=torch.uint8, device=device)

        # Construct FP8 tensor
        return MXFP8Tensor(
            shape=shape,
            dtype=dtype,
            fp8_dtype=self.fp8_dtype,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            quantizer=self,
            requires_grad=requires_grad,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        # TODO(ksivamani): No calibration needed for mxfp8?
        pass


class MXFP8Tensor(MXFP8TensorBase, QuantizedTensor):
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
            "MXFP8Tensor("
            f"fp8_dtype={self._fp8_dtype}, "
            f"scale_inv={self._scale_inv.item()}, "
            f"data={self.dequantize(dtype=self.dtype)}"
            ")"
        )

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from MXFP8Tensor

        By default the resulting tensor's dtype is the
        MXFP8Tensor's nominal dtype.
        """
        # Convert PyTorch dtype to TE dtype
        if dtype is None:
            dtype = self.dtype

        if torch.is_grad_enabled():
            return _FromMXFP8Func.apply(self, dtype)
        else:
            return _FromMXFP8Func.forward(None, self, dtype)

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> MXFP8Tensor:
        """Update FP8 data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        scale: torch.Tensor, optional
            Scaling factor to use for FP8 quantization
        amax: torch.Tensor, optional
            History of maximum absolute values. The first entry will
            be updated with the absmax of `tensor`.
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        src = tensor
        dst = self

        # In-place operations invalidate transpose cache
        self._reset_caches()

        # Special logic if other tensor is MXFP8Tensor
        if isinstance(src, MXFP8Tensor):

            # Cast to plain tensor if FP8 dtypes don't match
            if dst._fp8_dtype != src._fp8_dtype:
                return dst.quantize_(src.dequantize())

            # Directly copy FP8 data
            if src._data is not None:
                assert dst._data is not None
                dst._data.copy_(src._data.detach())
            else:
                dst._data = None
            dst._scale_inv.copy_(src._scale_inv.detach())
            if amax is not None or dst._fp8_meta is not None:
                src_amax: torch.Tensor
                if src._fp8_meta is None:
                    src_min, src_max = src.dequantize().aminmax()
                    src_amax = torch.maximum(-src_min, src_max)
                else:
                    fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                        forward=src._fp8_meta_forward,
                    )
                    fp8_meta_index = src._fp8_meta_index
                    src_amax = src._fp8_meta[fp8_meta_key].amax_history[0, fp8_meta_index]
                dst_amax: torch.Tensor
                if amax is None:
                    fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                        forward=dst._fp8_meta_forward,
                    )
                    fp8_meta_index = dst._fp8_meta_index
                    dst_amax = dst._fp8_meta[fp8_meta_key].amax_history[0, fp8_meta_index]
                else:
                    dst_amax = amax
                    if dst_amax.dim() > 0:
                        dst_amax = dst_amax[tuple([0] * dst_amax.dim())]
                torch.maximum(src_amax, dst_amax, out=dst_amax)
            if dst._transpose is not None:
                if src._transpose is None:
                    dst.transpose_2d(force_compute=True, fill_cache=True)
                else:
                    dst._transpose.copy_(src._transpose)
                dst._transpose_invalid = False
            return self

        # Convert QuantizedTensor to plain tensor
        if isinstance(src, QuantizedTensor):
            return dst.quantize_(src.dequantize())

        # Make sure input is in expected format
        if src.size() != dst.size():
            src = src.expand(dst.size())
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if src.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            src = src.float()
        if not src.is_contiguous():
            src = src.contiguous()

        # Make sure FP8 scaling factors are in expected format
        if scale is not None:
            if not devices_match(scale.device, dst.device) or scale.dtype != torch.float32:
                scale = scale.to(device=dst.device, dtype=torch.float32)
        if amax is not None:
            while amax.dim() < 2:
                amax = amax.unsqueeze(0)
            if not devices_match(amax.device, dst.device):
                raise ValueError(
                    f"Invalid device for amax (expected {dst.device}, found {amax.device})"
                )
            if amax.dtype != torch.float32:
                raise ValueError(f"Invalid dtype for amax (expected float32, found {amax.type})")

        # Default FP8 scaling factors
        fp8_meta = None
        if dst._fp8_meta is None:
            if scale is None:
                scale = dst._scale_inv.reciprocal()
            if amax is None:
                amax = torch.empty((1, 1), dtype=torch.float32, device=dst.device)
        else:
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                forward=dst._fp8_meta_forward,
            )
            fp8_meta = dst._fp8_meta[fp8_meta_key]

        # Check local data
        assert dst._data is not None
        if not dst._data.is_contiguous():
            raise RuntimeError("Transformer Engine cast kernels require contiguous data")

        # Perform FP8 cast
        if dst._transpose is None:
            dst_data = dst._data
            if src.dim() != 2:
                src = src.view(1, -1)
                dst_data = dst_data.view(1, -1)
            cast_to_fp8(
                src,
                fp8_meta,
                dst._fp8_meta_index,
                dst._fp8_dtype,
                out=dst_data,
                scale=scale,
                amax=amax,
                scale_inv=dst._scale_inv,
            )
        else:
            fp8_cast_transpose_fused(
                src.view(-1, src.size(-1)),
                fp8_meta,
                dst._fp8_meta_index,
                dst._fp8_dtype,
                cast_out=dst._data,
                transpose_out=dst._transpose,
                scale=scale,
                amax=amax,
                scale_inv=dst._scale_inv,
                noop_flag=noop_flag,
            )
            dst._transpose_invalid = False

        return self

    def detach(self) -> MXFP8Tensor:
        # pylint: disable=missing-function-docstring
        return MXFP8Tensor.make_like(
            self,
            data=self._data,
            fp8_attrs=self._fp8_attrs,
        )

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

    def clone(self) -> MXFP8Tensor:
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

    def view(self, *shape: Tuple[int]) -> MXFP8Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> MXFP8Tensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> MXFP8Tensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if self._data is not None and self._data.is_contiguous(memory_format=memory_format):
            return self
        if (self._transpose is not None and
            self._transpose.is_contiguous(memory_format=memory_format)):
            return self
        raise ValueError("MXFP8Tensor does not support different memory formats!")

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

        # Slice op
        if func == aten.slice.Tensor:
            tensor = args[0]
            data = tensor._data
            data_slice = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            return MXFP8Tensor.make_like(tensor, data=data_slice)

        # View op
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._data
            data_view = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            return MXFP8Tensor.make_like(tensor, data=data_view)

        # Default case
        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def _make_in_reduce_ex(
        cls,
        data: torch.Tensor,
        fp8_dtype: TE_DType,
        fp8_scale_inv: torch.Tensor,
        dtype: torch.dtype,
    ) -> MXFP8Tensor:
        """Build MXFP8Tensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return MXFP8Tensor(
            data=data,
            fp8_dtype=fp8_dtype,
            fp8_scale_inv=fp8_scale_inv,
            dtype=dtype,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling to remove references to FP8 metadata objects"""
        return (
            MXFP8Tensor._make_in_reduce_ex,
            (self._data, self._fp8_dtype, self._scale_inv, self.dtype),
        )

    def _get_data(self) -> MXFP8Tensor:
        """Get tensor data property"""
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes FP8 data if setting from a MXFP8Tensor. Otherwise
        casts to FP8.

        """

        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device

        # Just copy FP8 data if other tensor is MXFP8Tensor
        if isinstance(tensor, MXFP8Tensor):
            if (  # pylint: disable=too-many-boolean-expressions
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    MXFP8Tensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(MXFP8Tensor, type(self)).data.__set__(self, dummy_tensor)
            self._data = tensor._data
            self._fp8_attrs = tensor._fp8_attrs
            if self.requires_grad != tensor.requires_grad:
                self.requires_grad_(requires_grad=tensor.requires_grad)
            return

        # Quantize to FP8
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self.data = self._quantizer.quantize(tensor)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    # Cast to FP8 when setting MXFP8Tensor.data
    data = property(_get_data, _set_data)

class _ViewFunc(torch.autograd.Function):
    """View function

    View the MXFP8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: MXFP8Tensor,
        shape: Optional[list[int]] = None,
    ) -> MXFP8Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Construct new tensor if shape is provided
        new_data = tensor._data.view(*shape) if tensor._data is not None else None
        if tensor._transpose is not None:
            new_transpose = tensor._transpose.view(shape[-1], -1)
        else:
            new_transpose = None
        return MXFP8Tensor(shape,
                            tensor.dtype,
                            data=new_data,
                            fp8_scale_inv=tensor._scale_inv,
                            fp8_dtype=tensor._fp8_dtype,
                            data_transpose=new_transpose,
                            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, MXFP8Tensor):
            new_data = grad._data.view(*ctx.shape) if grad._data is not None else None
            if grad._transpose is not None:
                new_transpose = grad._transpose.view(ctx.shape[-1], -1)
            else:
                new_transpose = None
            dgrad = MXFP8Tensor(ctx.shape,
                                 grad.dtype,
                                 data=new_data,
                                 fp8_scale_inv=grad._scale_inv,
                                 fp8_dtype=grad._fp8_dtype,
                                 data_transpose=new_transpose,
                                 quantizer=grad._quantizer,
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the MXFP8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: MXFP8Tensor,
        shape: Optional[Tuple[int]] = None,
    ) -> MXFP8Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Construct new tensor if shape is provided
        new_data = tensor._data.reshape(*shape) if tensor._data is not None else None
        if tensor._transpose is not None:
            new_transpose = tensor._transpose.reshape(shape[-1], -1)
        else:
            new_transpose = None
        return MXFP8Tensor(shape,
                            tensor.dtype,
                            data=new_data,
                            fp8_scale_inv=tensor._scale_inv,
                            fp8_dtype=tensor._fp8_dtype,
                            data_transpose=new_transpose,
                            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, MXFP8Tensor):
            new_data = grad._data.reshape(*ctx.shape) if grad._data is not None else None
            if grad._transpose is not None:
                new_transpose = grad._transpose.reshape(ctx.shape[-1], -1)
            else:
                new_transpose = None
            dgrad = MXFP8Tensor(ctx.shape,
                                 grad.dtype,
                                 data=new_data,
                                 fp8_scale_inv=grad._scale_inv,
                                 fp8_dtype=grad._fp8_dtype,
                                 data_transpose=new_transpose,
                                 quantizer=grad._quantizer,
            )
            return dgrad, None
        return grad.reshape(ctx.shape), None
