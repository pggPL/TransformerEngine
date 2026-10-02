# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Save bulk allocations without retaining their individual tensor wrappers."""

from typing import NamedTuple

import torch


class _BufferView(NamedTuple):
    """Metadata for one tensor in a saved byte buffer."""

    index: int
    offset: int
    nbytes: int
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype


def pack_tensor_buffers(tensors, buffers):
    """Replace tensors inside contiguous uint8 buffers with owners and view metadata."""
    ranges = [(buffer.data_ptr(), buffer.numel(), buffer.device) for buffer in buffers]
    packed, metadata, buffer_indices = [], [], {}
    for tensor in tensors:
        view = None
        if type(tensor) is torch.Tensor:  # pylint: disable=unidiomatic-typecheck
            if tensor.numel():
                ptr = tensor.data_ptr()
                nbytes = tensor.element_size() * (
                    1
                    + sum(
                        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
                    )
                )
            else:
                ptr = (
                    tensor.untyped_storage().data_ptr()
                    + tensor.storage_offset() * tensor.element_size()
                )
                nbytes = 0
            for i, (base_ptr, buffer_size, device) in enumerate(ranges):
                offset = ptr - base_ptr
                if tensor.device == device and 0 <= offset and offset + nbytes <= buffer_size:
                    if i not in buffer_indices:
                        buffer_indices[i] = len(packed)
                        packed.append(buffers[i])
                    view = _BufferView(
                        buffer_indices[i],
                        offset,
                        nbytes,
                        tensor.shape,
                        tensor.stride(),
                        tensor.dtype,
                    )
                    break
        if view is None:
            metadata.append(len(packed))
            packed.append(tensor)
        else:
            metadata.append(view)
    return packed, metadata


def unpack_tensor_buffers(tensors, metadata):
    """Rebuild typed views after the owning byte buffers have been restored."""
    return [
        (
            tensors[view.index]
            .narrow(0, view.offset, view.nbytes)
            .view(view.dtype)
            .as_strided(view.shape, view.stride)
            if isinstance(view, _BufferView)
            else tensors[view]
        )
        for view in metadata
    ]
