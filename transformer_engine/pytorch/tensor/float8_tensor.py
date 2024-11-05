# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Iterable

import torch
import transformer_engine_torch as tex

from transformer_engine_torch import DType as TE_DType
from ..constants import TE_DType as torch_to_transformer_engine_dtype
from ..cpp_extensions.transpose import fp8_cast_transpose_fused
from ..cpp_extensions.cast import (
    cast_to_fp8,
)
from ..fp8 import FP8GlobalStateManager
from ..utils import devices_match, supports_fp8_transposes
from .quantized_tensor import QuantizedTensor, _QuantizeFunc, _IdentityFunc
from ..quantization_params import Float8ParamsProxy

from ._internal.float8_tensor_base import Float8TensorBase, _FromFloat8Func

aten = torch.ops.aten
updated_fp8_params = {}

def post_optimizer_step_fwd_amax_reduction(param: Float8Tensor) -> None:
    """Amax scale and update when there is at least 1 trainable FP8 parameter."""
    param_id = id(param._data)

    if param_id not in FP8GlobalStateManager.fp8_param_to_autocast:
        return

    autocast_key = FP8GlobalStateManager.fp8_param_to_autocast[param_id]

    if autocast_key not in FP8GlobalStateManager.autocast_to_fp8_params:
        return

    if autocast_key in updated_fp8_params:
        updated_fp8_params[autocast_key].add(param_id)
    else:
        updated_fp8_params[autocast_key] = {param_id}

    current_fp8_params_set = FP8GlobalStateManager.autocast_to_fp8_params[autocast_key]
    # All FP8 trainable parameters have been updated.
    if updated_fp8_params[autocast_key] == current_fp8_params_set:
        FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=True, fp8_weights=True)
        del updated_fp8_params[autocast_key]


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
    fp8_attrs: dict, optional
               FP8 metadata, primarily managed by Float8Tensor. If
               provided, all other FP8 configuration is ignored.
    proxy: FP8ParamsProxy, optional
              FP8 metadata object, primarily managed by TE modules.
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

    def __repr__(self):
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

    @staticmethod
    def quantize(tensor: torch.Tensor,
                 params: QuantizationParams,
                 *,
                 proxy: Optional[QuantizationParamsProxy] = None,
                 rowwise_usage: bool = True,
                 columnwise_usage: bool = True,
    ) -> QuantizedTensor:
        if torch.is_grad_enabled():
            return _QuantizeFunc.apply(
                tensor,
                params,
                rowwise_usage,
                columnwise_usage,
                proxy,
            )
        else:
            return _QuantizeFunc.forward(
                None,
                tensor,
                params,
                rowwise_usage,
                columnwise_usage,
                proxy,
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

        # Special logic if other tensor is Float8Tensor
        if isinstance(src, Float8Tensor):

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

        # Callback hook to perform amax reduction after optimizer step
        post_optimizer_step_fwd_amax_reduction(self)

        return self

    def detach(self) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return Float8Tensor.make_like(
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
            if not supports_fp8_transposes():
                if self._transpose is None or self._transpose_invalid:
                    self._create_transpose()
                self._data = None
        if columnwise_usage:
            if self._transpose is None or self._transpose_invalid:
                assert self._data is not None, \
                       "The tensor does not hold any data anymore"
                if not supports_fp8_transposes():
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
        if self._data.is_contiguous(memory_format=memory_format):
            return self
        return _IdentityFunc.apply(
            self,
            {"data": self._data.detach().contiguous(memory_format=memory_format)},
        )

    def to_dtype(self, dtype: torch.dtype) -> Float8Tensor:
        """Create `Float8Tensor` with given nominal dtype

        The new tensor has the same underlying FP8 data.

        """
        return Float8Tensor.make_like(
            self,
            data=self._data,
            fp8_attrs=self._fp8_attrs,
            dtype=dtype,
        )

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
            return Float8Tensor.make_like(tensor, data=data_slice)

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
            return Float8Tensor.make_like(tensor, data=data_view)

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
            self._data = tensor._data
            self._fp8_attrs = tensor._fp8_attrs
            if self.requires_grad != tensor.requires_grad:
                self.requires_grad_(requires_grad=tensor.requires_grad)
            return

        assert self._proxy is not None, "Can't quantize without a proxy"

        self.data = Float8Tensor.quantize(tensor,
                                          self._proxy.get_quantization_params(),
                                          rowwise_usage=self._data is not None,
                                          columnwise_usage=self._transpose is not None,
                                          proxy=self._proxy)

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
        return Float8Tensor(shape,
                            tensor.dtype,
                            data=new_data,
                            fp8_scale_inv=tensor._scale_inv,
                            fp8_dtype=tensor._fp8_dtype,
                            data_transpose=new_transpose,
                            proxy=tensor._proxy,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8Tensor):
            new_data = grad._data.view(*ctx.shape) if grad._data is not None else None
            if grad._transpose is not None:
                new_transpose = grad._transpose.view(ctx.shape[-1], -1)
            else:
                new_transpose = None
            dgrad = Float8Tensor(ctx.shape,
                                 grad.dtype,
                                 data=new_data,
                                 fp8_scale_inv=grad._scale_inv,
                                 fp8_dtype=grad._fp8_dtype,
                                 data_transpose=new_transpose,
                                 proxy=grad._proxy,
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the Float8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8Tensor,
        shape: Optional[Tuple[int]] = None,
    ) -> Float8Tensor:
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
        return Float8Tensor(shape,
                            tensor.dtype,
                            data=new_data,
                            fp8_scale_inv=tensor._scale_inv,
                            fp8_dtype=tensor._fp8_dtype,
                            data_transpose=new_transpose,
                            proxy=tensor._proxy,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8Tensor):
            new_data = grad._data.reshape(*ctx.shape) if grad._data is not None else None
            if grad._transpose is not None:
                new_transpose = grad._transpose.reshape(ctx.shape[-1], -1)
            else:
                new_transpose = None
            dgrad = Float8Tensor(ctx.shape,
                                 grad.dtype,
                                 data=new_data,
                                 fp8_scale_inv=grad._scale_inv,
                                 fp8_dtype=grad._fp8_dtype,
                                 data_transpose=new_transpose,
                                 proxy=grad._proxy,
            )
            return dgrad, None
        return grad.reshape(ctx.shape), None
