/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "../extensions.h"

namespace transformer_engine {
namespace pytorch {

/* Allocate multiple PyTorch tensors backed by the same buffer.
 *
 * Use with caution and avoid exposing externally.
 *
 * In order to reduce CPU overhead, we compute pointer offsets
 * manually and construct PyTorch tensors with raw pointers. The
 * backing buffer is deallocated once the final tensor is destroyed.
 * Stream usage is not recorded, so there may be race conditions if
 * compute is performed on multiple streams.
 */
std::tuple<std::vector<at::Tensor>, at::Tensor, TensorLayout> bulk_allocate_with_buffer(
    const std::vector<std::vector<size_t>> &shapes, const std::vector<at::ScalarType> &dtypes,
    std::optional<c10::Device> device, std::optional<std::vector<size_t>> alignments) {
  // Check shapes and dtypes
  const size_t n = shapes.size();
  NVTE_CHECK(dtypes.size() == n, "Got ", shapes.size(), " shapes and ", dtypes.size(), " dtypes.");
  NVTE_CHECK(!alignments || alignments->size() == n, "Got ", shapes.size(), " shapes and ",
             alignments->size(), " alignments.");

  // Return immediately if no tensors are needed
  if (n == 0) return {};

  // Set defaults for optional arguments
  if (!device) {
    device = c10::Device(c10::kCUDA);
  }
  if (!alignments) {
    alignments = std::vector<size_t>{};
    alignments->reserve(n);
    for (const auto &dtype : dtypes) {
      alignments->push_back(c10::elementSize(dtype));
    }
  }

  // Compute offsets in base buffer
  TensorLayout layout;
  layout.tensors.reserve(n);
  size_t base_byte_size = 0;
  size_t base_alignment = 1;
  for (size_t i = 0; i < n; ++i) {
    const auto byte_size = product(shapes[i]) * at::elementSize(dtypes[i]);
    const auto offset = roundup(base_byte_size, (*alignments)[i]);
    layout.tensors.push_back({0, static_cast<int64_t>(offset),
                              c10::SmallVector<int64_t, 4>(shapes[i].begin(), shapes[i].end()),
                              dtypes[i]});
    base_byte_size = offset + byte_size;
    base_alignment = std::max(base_alignment, (*alignments)[i]);
  }
  if (base_alignment > 1) {
    // Pad in case data pointer is not aligned
    base_byte_size += base_alignment;
  }

  // Allocate base buffer
  auto base_buffer =
      at::empty({static_cast<int64_t>(base_byte_size)}, at::device(*device).dtype(torch::kUInt8));
  uint8_t *base_ptr = base_buffer.data_ptr<uint8_t>();
  base_ptr =
      reinterpret_cast<uint8_t *>(roundup(reinterpret_cast<uintptr_t>(base_ptr), base_alignment));

  const auto base_offset = base_ptr - base_buffer.data_ptr<uint8_t>();
  if (base_offset != 0) {
    for (auto &desc : layout.tensors) desc.offset += base_offset;
  }
  auto tensors = get_tensors({base_buffer}, layout);
  return {std::move(tensors), std::move(base_buffer), std::move(layout)};
}

std::vector<at::Tensor> get_tensors(const std::vector<at::Tensor> &buffers,
                                    const TensorLayout &layout) {
  std::vector<std::shared_ptr<at::Tensor>> owners;
  owners.reserve(buffers.size());
  for (const auto &buffer : buffers) {
    NVTE_CHECK(buffer.scalar_type() == at::kByte && buffer.dim() == 1 && buffer.is_contiguous(),
               "Expected a contiguous one-dimensional uint8 buffer");
    owners.emplace_back(std::make_shared<at::Tensor>(buffer));
  }
  std::vector<at::Tensor> tensors;
  tensors.reserve(layout.tensors.size());
  for (const auto &desc : layout.tensors) {
    if (desc.buffer_index < 0) {
      tensors.emplace_back();
      continue;
    }
    NVTE_CHECK(static_cast<size_t>(desc.buffer_index) < owners.size(), "Invalid buffer index");
    const auto &owner = owners[desc.buffer_index];
    int64_t nbytes = c10::elementSize(desc.dtype);
    for (const auto dim : desc.shape) {
      NVTE_CHECK(dim >= 0 && (dim == 0 || nbytes <= std::numeric_limits<int64_t>::max() / dim),
                 "Invalid tensor shape");
      nbytes *= dim;
    }
    NVTE_CHECK(
        desc.offset >= 0 && desc.offset <= owner->numel() && nbytes <= owner->numel() - desc.offset,
        "Tensor exceeds its backing buffer");
    if (nbytes == 0) {
      // Empty from_blob tensors with a non-null pointer can break TE kernels.
      tensors.emplace_back(at::empty(desc.shape, owner->options().dtype(desc.dtype)));
    } else {
      auto *ptr = owner->data_ptr<uint8_t>() + desc.offset;
      NVTE_CHECK(reinterpret_cast<uintptr_t>(ptr) % c10::elementSize(desc.dtype) == 0,
                 "Tensor data is not aligned for its dtype");
      tensors.emplace_back(
          at::from_blob(ptr, desc.shape, [owner](void *) {}, owner->options().dtype(desc.dtype)));
    }
  }
  return tensors;
}

std::vector<at::Tensor> bulk_allocate(const std::vector<std::vector<size_t>> &shapes,
                                      const std::vector<at::ScalarType> &dtypes,
                                      std::optional<c10::Device> device,
                                      std::optional<std::vector<size_t>> alignments) {
  return std::get<0>(bulk_allocate_with_buffer(shapes, dtypes, device, alignments));
}

}  // namespace pytorch
}  // namespace transformer_engine
