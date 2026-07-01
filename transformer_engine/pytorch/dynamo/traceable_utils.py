# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pure-Python, torch.compile-traceable quantized tensor allocation and reassembly."""

from __future__ import annotations
import copy as _copy
from typing import Any, Dict, List, Optional, Tuple, Union

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
    equivalent of ``Quantizer.make_empty``).  The quantizer is copied first
    so the caller's instance is never mutated.

    When ``quantizer`` is None, falls back to ``torch.empty`` for a plain tensor.

    Stashed metadata (``_te_flat_names``, ``_te_flat_ctx``)
    ---------------------------------------------------------
    The resulting quantized tensor has these attributes stashed so that
    ``forward_fn`` (in custom_op.py) can read them at Dynamo trace time.

    Why: ``forward_fn`` runs inside torch.compile's trace.  It needs the flat
    buffer names and unflatten context to decode the custom op's flat Tensor[]
    return back into a structured QuantizedTensor.  Calling
    ``__tensor_flatten__()`` for this would cause a graph break (it returns
    non-Tensor Python objects -- List[str] and Dict -- that Dynamo cannot
    represent as graph nodes).  Accessing ``t._quantizer`` and calling
    ``_describe_buffers()`` on it also fails: while Dynamo treats the retrieved
    quantizer as constant metadata, it wraps it in a generic VariableTracker
    that does not support method calls (unlike a closure-captured quantizer
    which is recognized as a value-opaque constant).  Stashing the buffer names
    and context as plain attributes sidesteps both issues -- Dynamo reads them
    as constants without needing to call any methods.

    Allocation cost: when ``forward_fn`` calls the fake impl to obtain these
    templates, the ``torch.empty`` calls appear as nodes in the initial Dynamo
    FX graph.  However, because the tensors themselves are never used (only
    the stashed metadata is read), AOT autograd's dead-code elimination removes
    them before any kernel code is generated.  They do not appear in the final
    compiled graph.
    """
    device = torch.device(device if device is not None else "cuda")
    shape = tuple(shape)
    if quantizer is None:
        return torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    from ..quantized_tensor import _STORAGE_REGISTRY  # pylint: disable=import-outside-toplevel

    # Copy so the caller's quantizer is not mutated by alloc_tensors internals.
    # The caller is expected to have already called set_usage() on the quantizer
    # before passing it here -- Dynamo tracks those mutations as explicit setattr
    # nodes in the graph, so the copy captures the post-mutation state and the
    # stashed _te_flat_names reflects the correct buffer layout.
    q = quantizer.copy() if hasattr(quantizer, "copy") else _copy.copy(quantizer)
    ctx = q.create_metadata(shape, dtype=dtype, requires_grad=requires_grad)
    inner = q.alloc_tensors(shape, device=device)
    storage_cls = _STORAGE_REGISTRY[ctx["cls"]]
    result = storage_cls.__tensor_unflatten__(inner, ctx, shape, _contiguous_stride(shape))
    if requires_grad and hasattr(result, "requires_grad_"):
        result.requires_grad_(True)
    # TODO: understand why Dynamo does not recognize the quantizer retrieved via
    # t._quantizer as the same value-opaque type it would if captured from a
    # closure. If that is fixed upstream, the stashed attributes become
    # unnecessary and we could compute slot counts directly from the quantizer.
    result._te_flat_names = tuple(inner.keys())
    result._te_flat_ctx = ctx
    return result


# --------------------------------------------------------------------------- #
# Slot counting and reassembly for the custom-op flat Tensor[] protocol.
# --------------------------------------------------------------------------- #


def _slot_count(value: Any) -> int:
    """Number of flat tensor slots a value occupies in the op's Tensor[] return.

    Reads ``_te_flat_names`` stashed by :func:`make_empty_traceable`, which is
    safe to access at Dynamo trace time (treated as constant metadata on a
    traceable wrapper subclass).  Plain tensors (no stashed names) occupy 1 slot.
    """
    if value is None:
        return 1
    names = getattr(value, "_te_flat_names", None)
    if names is not None:
        return len(names)
    return 1


def _maybe_reassemble_tensor_subclass(
    template: Any,
    chunk: List[Optional[torch.Tensor]],
) -> Optional[Union[torch.Tensor, Any]]:
    """Rebuild a value from its flat tensors using ``template`` for geometry.

    ``template`` is a tensor produced by the fake impl via
    :func:`make_empty_traceable`.  Uses the stashed ``_te_flat_names`` /
    ``_te_flat_ctx`` attributes for reassembly (trace-safe).  For plain tensors
    (no stashed metadata), returns the single chunk element directly.
    """
    if template is None:
        return None
    names = getattr(template, "_te_flat_names", None)
    ctx = getattr(template, "_te_flat_ctx", None)
    if names is None or ctx is None:
        return chunk[0]
    inner_dict = dict(zip(names, chunk))
    shape = tuple(template.shape) if hasattr(template, "shape") else tuple(template.size())
    stride = _contiguous_stride(shape)
    return type(template).__tensor_unflatten__(inner_dict, ctx, shape, stride)
