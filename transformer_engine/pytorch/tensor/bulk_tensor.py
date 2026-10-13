# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Typed regions of one byte buffer."""

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import torch
import transformer_engine_torch as tex


class BulkTensorRegion(NamedTuple):
    """A contiguous region; offset and alignment are expressed in bytes."""

    offset: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    alignment: int


@dataclass(frozen=True)
class BulkTensor:
    """One owning byte tensor and immutable, pointer-free region metadata.

    Save ``buffer`` through autograd and retain ``layout`` as metadata. After
    offload, rebuild this object with the restored buffer and the same layout.
    Native consumers can call ``unpack_bulk_tensor(buffer, layout)`` to obtain
    C++ tensors without creating Python objects for the individual regions.
    """

    buffer: torch.Tensor
    layout: tuple[BulkTensorRegion, ...]

    def __post_init__(self):
        if self.buffer.dtype != torch.uint8 or self.buffer.ndim != 1:
            raise ValueError("BulkTensor requires a one-dimensional uint8 buffer")
        if not self.buffer.is_contiguous() or self.buffer.storage_offset() != 0:
            raise ValueError("BulkTensor requires a contiguous buffer with zero storage offset")
        if not isinstance(self.layout, tuple) or any(
            not isinstance(region, BulkTensorRegion) or not isinstance(region.shape, tuple)
            for region in self.layout
        ):
            raise TypeError("BulkTensor layout must be a tuple of BulkTensorRegion objects")

    @classmethod
    def empty(
        cls,
        shapes: Sequence[Sequence[int]],
        dtypes: Sequence[torch.dtype],
        *,
        device,
        alignments: Optional[Sequence[int]] = None,
    ):
        """Allocate disjoint contiguous regions without materializing child tensors.

        Alignments must be powers of two and multiples of the element size.
        The device allocator must provide the corresponding base alignment;
        native unpacking checks this, including after an offload round trip.
        """
        if len(shapes) != len(dtypes) or (
            alignments is not None and len(alignments) != len(shapes)
        ):
            raise ValueError("BulkTensor shapes, dtypes and alignments must have equal lengths")
        layout = []
        nbytes = 0
        for index, (shape, dtype) in enumerate(zip(shapes, dtypes)):
            shape = tuple(shape)
            itemsize = dtype.itemsize
            alignment = itemsize if alignments is None else alignments[index]
            if alignment <= 0 or alignment & (alignment - 1) or alignment % itemsize:
                raise ValueError("BulkTensor alignment must be a power of two covering the dtype")
            size = itemsize
            for dim in shape:
                torch._check(dim >= 0, lambda: "BulkTensor dimensions must be nonnegative")
                size *= dim
            offset = (nbytes + alignment - 1) // alignment * alignment
            layout.append(BulkTensorRegion(offset, shape, dtype, alignment))
            nbytes = offset + size
        buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)
        return cls(buffer, tuple(layout))

    def unpack(self) -> list[torch.Tensor]:
        """Materialize aliases for eager Python callers; do not return them from a functional op."""
        return tex.unpack_bulk_tensor(self.buffer, self.layout)
