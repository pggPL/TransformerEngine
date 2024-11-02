# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor with quantized data"""

from __future__ import annotations
from typing import Optional, Tuple, List

import torch
from torch.utils._pytree import tree_map

import transformer_engine_torch as tex

from ..quantization_params import QuantizationParams

class QuantizationParamsProxy:
    def __init__(self):
        pass

    def get_quantization_params(self) -> QuantizationParams:
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement get_quantization_params function"
        )

class _DequantizeFunc(torch.autograd.Function):
    """Autograd function to convert quantized tensor to standard tensor"""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        tensor: QuantizedTensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return tensor.dequantize(dtype=dtype)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        return grad, None


class _IdentityFunc(torch.autograd.Function):
    """Autograd function to create quantized tensor with same data"""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        tensor: QuantizedTensor,
    ) -> QuantizedTensor:
        # pylint: disable=missing-function-docstring
        return tensor.detach()

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return grad

class _QuantizeFunc(torch.autograd.Function):
    """Autograd function to convert standard tensor to quantized tensor"""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        tensor: torch.Tensor,
        qparams: QuantizationParams,
        rowwise_usage: bool = True,
        columnwise_usage: bool = True,
        proxy: Optional[QuantizationParamsProxy] = None,
    ) -> QuantizedTensor:
        # pylint: disable=missing-function-docstring
        if isinstance(tensor, QuantizedTensor):
            tensor = tensor.dequantize()

        out = tex.generic_cast(tensor,
                               qparams,
                               rowwise_usage,
                               columnwise_usage,
                               proxy)

        return out

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None, None, None, None, None, None, None

class QuantizedTensor(torch.Tensor):
    """Abstract base class for tensor with quantized data

    This is a proxy class with the interface of a standard PyTorch
    tensor, but with data that has been encoded with some quantization
    scheme. Derived classes should implement the quantization scheme
    by overriding the `quantize_` and `dequantize` functions.

    """

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convert quantized data to standard PyTorch tensor"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement dequantize function"
        )

    def quantize_(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Update quantized data in-place"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement quantize_ function"
        )

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


    def detach(self) -> QuantizedTensor:
        """Create new quantized tensor with same data

        Output tensor must be detached from the current autograd
        graph.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement detach function"
        )

    def update_usage(self, rowwise_usage=True, columnwise_usage=True):
        """Indicate to the tensor how it is going to be used.
        This enables optimizations to memory usage in some cases
        where forward and backward passes use the tensor in
        different directions.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement update_usage function"
        )

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self.dequantize(dtype=self.dtype)})"

    def float(self) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return _DequantizeFunc.apply(self, torch.float32)

    def bfloat16(self) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return _DequantizeFunc.apply(self, torch.bfloat16)

    def half(self) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return _DequantizeFunc.apply(self, torch.float16)

    def cpu(self) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return _DequantizeFunc.apply(self).cpu()

    def expand_as(self, other: torch.Tensor) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if other is self:
            # Note: expand_as is hackily used to create dummy autograd nodes
            # and access the backward graph (see
            # https://github.com/pytorch/pytorch/blob/238fb660851268f44ff88127887041fea352fe48/torch/nn/parallel/distributed.py#L1026).
            # We hackily add a dummy function to handle this case.
            return _IdentityFunc.apply(self)
        return super().expand_as(other)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # Detach op
        if func == torch.ops.aten.detach.default:
            return args[0].detach()

        # In-place copy op
        if func == torch.ops.aten.copy_.default:
            dst = args[0]
            src = args[1]
            if isinstance(dst, QuantizedTensor):
                dst.quantize_(src)
            else:
                if isinstance(src, QuantizedTensor):
                    src = src.dequantize()
                dst.copy_(src)
            return None

        # View op
        if func == torch.ops.aten.view.default:
            raise NotImplementedError("{cls.__name__} class does not support tensor views")

        def maybe_unwrap(arg):
            if isinstance(arg, QuantizedTensor):
                return arg.dequantize(dtype=arg.dtype)
            return arg

        def maybe_update_inplace(arg, new_arg, schema_arg):
            if (
                isinstance(arg, QuantizedTensor)
                and isinstance(new_arg, torch.Tensor)
                and hasattr(schema_arg, "alias_info")
                and hasattr(schema_arg.alias_info, "is_write")
                and schema_arg.alias_info.is_write
            ):
                arg.quantize_(new_arg)

        # In-place op: dequantize, perform op, and quantize
        if func._schema.is_mutable:
            new_args = tree_map(maybe_unwrap, args)
            new_kwargs = tree_map(maybe_unwrap, kwargs)
            schema_args = func._schema.arguments
            args_len = len(args)
            super().__torch_dispatch__(func, types, new_args, new_kwargs)
            for arg, new_arg, schema_arg in zip(args, new_args, schema_args):
                maybe_update_inplace(arg, new_arg, schema_arg)
            for kwarg, new_kwarg, schema_arg in zip(kwargs, new_kwargs, schema_args[args_len:]):
                assert kwarg == new_kwarg == schema_arg.name, "name of the kw argument should match"
                maybe_update_inplace(kwargs[kwarg], new_kwargs[new_kwarg], schema_arg)
            return None

        # Default op: dequantize and perform op
        args = tree_map(maybe_unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(maybe_unwrap, kwargs)
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Do not force the QuantizedTensor type on the returned tensor
        return torch._C._disabled_torch_function_impl(func, types, args, kwargs)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> QuantizedTensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement contiguous function"
        )
