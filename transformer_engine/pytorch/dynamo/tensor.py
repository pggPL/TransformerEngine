# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor allocation and metadata reconstruction for custom-op fake kernels."""

from __future__ import annotations

import torch
from torch._prims_common import make_contiguous_strides_for

from ..device import te_device_type
from ..quantized_tensor import QuantizedTensorStorage


def empty_tensor(
    shape,
    *,
    dtype: torch.dtype,
    device=None,
    quantizer=None,
    requires_grad: bool = False,
):
    """Allocate a plain tensor or quantized storage using the active dispatch mode."""
    shape = tuple(shape)
    device = device if device is not None else te_device_type()
    if quantizer is None:
        # Custom-op autograd determines output requires_grad.
        return torch.empty(shape, dtype=dtype, device=device)
    quantizer = quantizer.copy()
    inner = quantizer.alloc_tensors(shape, device=device)
    metadata = quantizer.create_metadata(shape, dtype=dtype, requires_grad=requires_grad)
    return QuantizedTensorStorage.__tensor_unflatten__(
        inner, metadata, shape, make_contiguous_strides_for(shape)
    )


def tensor_dtype(value):
    """Logical dtype of a tensor or bare quantized storage."""
    return value._dtype if isinstance(value, QuantizedTensorStorage) else value.dtype


def inner_names(value) -> tuple[str, ...]:
    """Flat payload fields, in the storage's canonical order."""
    if isinstance(value, QuantizedTensorStorage):
        return tuple(
            name for name, _ in type(value)._INNER_TENSORS if getattr(value, name) is not None
        )
    return ("data",)


def assemble_tensor(prototype, tensors):
    """Replace a prototype's buffers with a custom op's actual outputs."""
    if not isinstance(prototype, QuantizedTensorStorage):
        return tensors[0]
    names, metadata = QuantizedTensorStorage.__tensor_flatten__(prototype)
    shape = tuple(prototype.shape)
    stride = (
        prototype.stride()
        if isinstance(prototype, torch.Tensor)
        else make_contiguous_strides_for(shape)
    )
    return QuantizedTensorStorage.__tensor_unflatten__(
        dict(zip(names, tensors)), metadata, shape, stride
    )


def to_meta(value):
    """Create meta tensors with preserved memory format and existing storage metadata."""
    if value is None:
        return None
    if isinstance(value, QuantizedTensorStorage):
        names, _ = QuantizedTensorStorage.__tensor_flatten__(value)
        buffers = [to_meta(getattr(value, name)) for name in names]
        return assemble_tensor(value, buffers)
    return torch.empty_like(value, device="meta", memory_format=torch.preserve_format)


def with_usage(value, *, rowwise_usage=None, columnwise_usage=None):
    """Allocate a fake layout without invoking native transpose/quantization kernels."""
    quantizer = value._quantizer.copy()
    quantizer.set_usage(rowwise=rowwise_usage, columnwise=columnwise_usage)
    result = empty_tensor(
        value.shape, dtype=tensor_dtype(value), device=value.device, quantizer=quantizer
    )
    if hasattr(value, "_with_gemm_swizzled_scales"):
        result._with_gemm_swizzled_scales = value._with_gemm_swizzled_scales
    return result
