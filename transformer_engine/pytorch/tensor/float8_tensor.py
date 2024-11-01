# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import warnings

import torch
from torch._prims_common import is_contiguous
import transformer_engine_torch as tex

from transformer_engine_torch import DType as TE_DType
from ...common.recipe import DelayedScaling, Recipe
from ..constants import TE_DType as torch_to_transformer_engine_dtype
from ..cpp_extensions.transpose import fp8_cast_transpose_fused
from ..cpp_extensions.cast import (
    cast_to_fp8,
)
from ..fp8 import FP8GlobalStateManager
from ..utils import devices_match
from .quantized_tensor import QuantizedTensor, Quantizer

aten = torch.ops.aten
updated_fp8_params = {}


class _QuantizeFunc(torch.autograd.Function):
    """Cast to FP8 from other dtype"""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        tensor: torch.Tensor,
        quantizer: Float8Quantizer,
    ) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return tex.generic_cast(tensor, quantizer)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None


class _DequantizeFunc(torch.autograd.Function):
    """Cast from FP8 to other dtype"""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        tensor: Float8Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Convert PyTorch dtype to TE dtype
        if dtype is None:
            dtype = tensor.dtype
        dtype = torch_to_transformer_engine_dtype[dtype]

        # Make sure FP8 data is in expected format
        data = tensor._data
        assert data is not None

        # Cast from FP8
        return tex.cast_from_fp8(data, tensor._scale_inv, tensor._fp8_dtype, dtype, 0)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None


class Float8Quantizer(Quantizer):

    scale: torch.Tensor
    amax: torch.Tensor
    dtype: TE_Dtype

    def __init__(
        self,
        scale: torch.Tensor,
        amax: torch.Tensor,
        fp8_dtype: TE_Dtype,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.amax = amax
        self.dtype = fp8_dtype
        self.rowwise_usage = rowwise
        self.columnwise_usage = columnwise

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: Float8Tensor,
    ) -> Float8Tensor:

        # Launch cast kernel
        if dst._transpose is None:
            dst_data = dst._data
            if src.dim() != 2:
                src = src.view(1, -1)
                dst_data = dst_data.view(1, -1)
            cast_to_fp8(
                src,
                None,
                None,
                self.dtype,
                out=dst_data,
                scale=self.scale,
                amax=self.amax,
                scale_inv=dst._scale_inv,
            )
        else:
            fp8_cast_transpose_fused(
                src.view(-1, src.size(-1)),
                None,
                None,
                self.dtype,
                cast_out=dst._data,
                transpose_out=dst._transpose,
                scale=self.scale,
                amax=self.amax,
                scale_inv=dst._scale_inv,
                noop_flag=noop_flag,  ### TODO How to handle?
            )
            dst._transpose_invalid = False

        # Update FP8 dtype
        dst._fp8_dtype = self.dtype

        return dst

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        out: Optional[Float8Tensor] = None,
    ) -> Float8Tensor:
        if out is not None:
            return self.update_quantized(tensor, out)
        if torch.is_grad_enabled():
            return _QuantizeFunc.apply(tensor, self)
        return _QuantizeFunc.forward(None, tensor, self)

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> Float8Tensor:

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")
        if dtype is None:
            dtype = torch.float32

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
            data=data,
            fp8_dtype=self.fp8_dtype,
            dtype=dtype,
            requires_grad=requires_grad,
            data_transpose=data_transpose,
            quantizer=self,
        )

    def calibrate(self, recipe: Recipe, tensor: torch.Tensor) -> None:
        if isinstance(recipe, DelayedScaling):
            amin, amax = tensor.aminmax()
            self.amax.copy_(torch.max(-amin, amax))
            return
        raise NotImplementedError("Not implemented yet!")


class FP8TensorMetaProxyQuantizer(Quantizer):

    meta: tex.FP8TensorMeta
    index: int

    def __init__(
        self,
        meta: dict,
        index: int,
        fp8_dtype: TE_Dtype,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ):
        super().__init__()
        self.meta = meta
        self.index = index
        self.dtype = fp8_dtype
        self.rowwise_usage = rowwise
        self.columnwise_usage = columnwise

    @property
    def scale(self) -> torch.Tensor:
        return self.meta.scale[self.index]

    @property
    def amax(self) -> torch.Tensor:
        return self.meta.amax_history[0][self.index]

    def resolve(self) -> Float8Quantizer:
        return Float8Quantizer(
            scale=self.scale,
            amax=self.amax,
            fp8_dtype=self.dtype,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
        )

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: Float8Tensor,
    ) -> Float8Tensor:
        self.resolve().update_quantized(src, dst)

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        out: Optional[Float8Tensor] = None,
    ) -> Float8Tensor:
        return self.resolve().quantize(tensor, out=out)

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> Float8Tensor:
        return self.resolve().make_empty(
            shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    def calibrate(self, recipe: Recipe, tensor: torch.Tensor) -> None:
        self.resolve().calibrate(recipe, tensor)


def _make_fp8_attr_property_funcs(name: str) -> Any:
    """Make accessors for an FP8 attribute

    We store FP8 attributes in a dictionary so we can share them
    between tensors with the same data, e.g. detached tensors. For
    convenience, we also expose them as property attributes. This
    function creates the accessors for property attributes.

    Parameters
    ----------
    name: str
          Key in dictionary of FP8 attributes

    """

    def get_func(self) -> Any:
        return self._fp8_attrs[name]

    def set_func(self, value: Any) -> None:
        self._fp8_attrs[name] = value

    def del_func(self) -> None:
        del self._fp8_attrs[name]

    return {"fget": get_func, "fset": set_func, "fdel": del_func}


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


class _IdentityFunc(torch.autograd.Function):
    """Identity function

    If constructor keyword-arguments are provided, then construct a
    new Float8Tensor using the provided tensor's attributes.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8Tensor,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if constructor kwargs are not provided
        ctx.input_dtype = tensor.dtype
        if init_kwargs is None:
            return tensor

        # Construct new tensor if constructor kwargs are provided
        default_kwargs = {
            "data": tensor._data,
            "quantizer": tensor._quantizer,
            "fp8_dtype": tensor._fp8_dtype,
            "fp8_scale_inv": tensor._scale_inv,
            "dtype": tensor.dtype,
        }
        for key, val in default_kwargs.items():
            if key not in init_kwargs:
                init_kwargs[key] = val
        return Float8Tensor(**init_kwargs)

    @staticmethod
    def backward(ctx, grad):
        # pylint: disable=missing-function-docstring
        return grad.to(ctx.input_dtype), None


class _ViewFunc(torch.autograd.Function):
    """View function

    View the Float8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        shape: Optional[Tuple[int]] = None,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Construct new tensor if shape is provided
        if isinstance(tensor, Float8Tensor):
            if tensor._data is None:
                return tensor
            return Float8Tensor.make_like(
                tensor,
                data=tensor._data.view(*shape),
            )
        return tensor.view(*shape)

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8Tensor):
            if grad._data is None:
                return grad, None
            dgrad = Float8Tensor.make_like(
                grad,
                data=grad._data.view(ctx.shape),
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
        tensor: torch.Tensor,
        shape: Optional[Tuple[int]] = None,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Construct new tensor if shape is provided
        if isinstance(tensor, Float8Tensor):
            if tensor._data is None:
                return tensor
            return Float8Tensor.make_like(
                tensor,
                data=tensor._data.reshape(*shape),
            )
        return tensor.reshape(*shape)

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8Tensor):
            if grad._data is None:
                return grad, None
            dgrad = Float8Tensor.make_like(
                grad,
                data=grad._data.reshape(ctx.shape),
            )
            return dgrad, None
        return grad.reshape(ctx.shape), None


class Float8Tensor(QuantizedTensor):
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

    _data: Optional[torch.Tensor]
    _fp8_attrs: Dict[str, Any]
    _quantizer: Optional[Float8Quantizer]
    _fp8_dtype: TE_DType
    _scale_inv: torch.Tensor

    # FP8 transpose cache
    _transpose: Optional[torch.Tensor]
    _transpose_invalid: bool

    def __new__(
        cls,
        *,
        data: torch.Tensor,
        fp8_scale_inv: torch.Tensor,
        fp8_dtype: TE_DType,
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        data_transpose: Optional[torch.Tensor] = None,
        quantizer: Optional[Float8Quantizer] = None,
        fp8_attrs: Optional[Dict[str, Any]] = None,
    ):
        # Initialize tensor object
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=dtype,
            layout=data.layout,
            requires_grad=requires_grad,
            device=data.device,
        )
        self._data = data

        # Initialize dict of class attributes
        # Note: We store FP8 attributes in a dictionary so we can
        # share them between tensors with the same data, e.g. detached
        # tensors.
        if fp8_attrs is None:
            self._fp8_attrs = {}
        else:
            self._fp8_attrs = fp8_attrs
            return self

        # Builder class for Float8Tensor
        self._quantizer = quantizer

        # FP8 dtype
        self._fp8_dtype = fp8_dtype

        # FP8 scale-inverse
        self._scale_inv = fp8_scale_inv

        # FP8 transpose cache
        self._transpose = data_transpose
        self._transpose_invalid = self._transpose is None

        return self

    @classmethod
    def make_like(
        cls,
        tensor: Float8Tensor,
        *,
        data: torch.Tensor,
        fp8_attrs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Float8Tensor:
        """Use attributes of a Float8Tensor to create another Float8Tensor

        See constructor for list of keyword arguments.

        """
        default_kwargs = {
            "quantizer": tensor._quantizer,
            "fp8_dtype": tensor._fp8_dtype,
            "fp8_scale_inv": tensor._scale_inv,
            "dtype": tensor.dtype,
        }
        for key, val in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = val
        return Float8Tensor(data=data, fp8_attrs=fp8_attrs, **kwargs)

    def __repr__(self):
        return (
            "Float8Tensor("
            f"fp8_dtype={self._fp8_dtype}, "
            f"scale_inv={self._scale_inv.item()}, "
            f"data={self.from_float8(dtype=self.dtype)}"
            ")"
        )

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if torch.is_grad_enabled():
            return _DequantizeFunc.apply(self, dtype)
        return _DequantizeFunc.forward(None, self, dtype)

    def from_float8(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from Float8Tensor

        By default the resulting tensor's dtype is the
        Float8Tensor's nominal dtype.
        """
        return _DequantizeFunc.apply(self, dtype)

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

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        quantizer: Float8Quantizer,
    ) -> Float8Tensor:
        """Construct Float8Tensor from plain PyTorch tensor"""
        return quantizer.quantize(tensor)

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

        # Quantize to FP8
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self.data = self._quantizer.quantize(tensor)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    # Cast to FP8 when setting Float8Tensor.data
    data = property(_get_data, _set_data)

    # Accessors for objects in self._fp8_attrs
    # Note: We store FP8 attributes in a dictionary so we can share
    # them between tensors with the same data, e.g. detached tensors.
    # For convenience, we also expose them as property attributes.
    _quantizer = property(**_make_fp8_attr_property_funcs("quantizer"))
    _fp8_dtype = property(**_make_fp8_attr_property_funcs("dtype"))
    _transpose = property(**_make_fp8_attr_property_funcs("transpose"))
    _transpose_invalid = property(**_make_fp8_attr_property_funcs("transpose_invalid"))
    _scale_inv = property(**_make_fp8_attr_property_funcs("scale_inv"))
