# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Save split-quantize allocations using layouts supplied by the allocator."""

import transformer_engine_torch as tex


class BulkTensorState:
    """Collect native layouts, including each branch of hybrid quantization."""

    def __init__(self):
        self.groups = []

    def offload_tensors(self, tensors):
        """Select owners and any non-bulk hybrid branch for offload."""
        if not self.groups:
            return tensors
        result = [buffer for _, buffers, _ in self.groups for buffer in buffers]
        branches = {branch for branch, _, _ in self.groups}
        if None not in branches:
            for branch in ("_rowwise_storage", "_columnwise_storage"):
                if branch not in branches:
                    result.extend(getattr(tensor, branch, None) for tensor in tensors)
        return result

    def add(self, buffers, layout, branch=None):
        """Record one native split-quantize result."""
        if buffers:
            self.groups.append((branch, buffers, layout))

    def prepare_for_saving(self, tensors):
        """Detach retained bulk fields and save their owners once."""
        saved_buffers, metadata = [], []
        for branch, buffers, layout in self.groups:
            objects = tensors if branch is None else [getattr(t, branch, None) for t in tensors]
            retained, descriptions = tex.save_tensor_buffers(objects, buffers, layout)
            if retained:
                saved_buffers.extend(retained)
                metadata.append((branch, len(retained), descriptions))
        return saved_buffers, metadata


def restore_tensor_buffers(tensors, buffers, metadata):
    """Restore native storage fields after their owning buffers are reloaded."""
    offset = 0
    for branch, count, descriptions in metadata:
        objects = tensors if branch is None else [getattr(t, branch, None) for t in tensors]
        tex.restore_tensor_buffers(objects, buffers[offset : offset + count], descriptions)
        offset += count
