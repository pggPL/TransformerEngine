# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pure-Python, torch.compile-traceable quantized tensor allocation."""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union

import torch


def _contiguous_stride(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Row-major (contiguous) stride for ``shape``."""
    stride: list = []
    acc = 1
    for dim in reversed(shape):
        stride.append(acc)
        acc *= dim
    return tuple(reversed(stride))


def make_empty_traceable(
    quantizer,
    shape: Tuple[int, ...],
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[torch.device, str]] = None,
    requires_grad: bool = False,
) -> Any:
    """Allocate a tensor purely in Python (traceable under torch.compile).

    When ``quantizer`` is not None, produces a quantized tensor via
    ``alloc_tensors`` + ``__tensor_unflatten__`` (the compile-friendly
    equivalent of ``Quantizer.make_empty``).

    When ``quantizer`` is None, falls back to ``torch.empty`` for a plain tensor.
    """
    device = torch.device(device if device is not None else "cuda")
    shape = tuple(shape)
    if quantizer is None:
        return torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    from ..quantized_tensor import _STORAGE_REGISTRY  # pylint: disable=import-outside-toplevel

    ctx = quantizer.create_metadata(shape, dtype=dtype, requires_grad=requires_grad)
    inner = quantizer.alloc_tensors(shape, device=device)
    storage_cls = _STORAGE_REGISTRY[ctx["cls"]]
    result = storage_cls.__tensor_unflatten__(inner, ctx, shape, _contiguous_stride(shape))
    if requires_grad and hasattr(result, "requires_grad_"):
        result.requires_grad_(True)
    return result
