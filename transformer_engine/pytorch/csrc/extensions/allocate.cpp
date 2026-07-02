/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <memory>
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
std::vector<torch::stable::Tensor> bulk_allocate(
    const std::vector<std::vector<size_t>> &shapes,
    const std::vector<torch::headeronly::ScalarType> &dtypes,
    // TODO(stable-abi): needs a stable device type (torch::stable::Device) that
    // can be produced from a Python torch.device.
    std::optional<torch::stable::Device> device,
    std::optional<std::vector<size_t>> alignments) {
  // Check shapes and dtypes
  const size_t n = shapes.size();
  NVTE_CHECK(dtypes.size() == n, "Got ", shapes.size(), " shapes and ", dtypes.size(), " dtypes.");
  NVTE_CHECK(!alignments || alignments->size() == n, "Got ", shapes.size(), " shapes and ",
             alignments->size(), " alignments.");

  // Return immediately if no tensors are needed
  if (n == 0) return {};

  // Set defaults for optional arguments
  if (!device) {
    device = torch::stable::Device(torch::headeronly::DeviceType::CUDA);
  }
  if (!alignments) {
    alignments = std::vector<size_t>{};
    alignments->reserve(n);
    for (const auto &dtype : dtypes) {
      // TODO(stable-abi): needs torch::headeronly::elementSize(ScalarType).
      alignments->push_back(torch::headeronly::elementSize(dtype));
    }
  }

  // Compute offsets in base buffer
  std::vector<size_t> byte_sizes(n);
  std::vector<size_t> offsets(n);
  size_t base_byte_size = 0;
  size_t base_alignment = 1;
  for (size_t i = 0; i < n; ++i) {
    byte_sizes[i] = product(shapes[i]) * torch::headeronly::elementSize(dtypes[i]);
    offsets[i] = roundup(base_byte_size, (*alignments)[i]);
    base_byte_size = offsets[i] + byte_sizes[i];
    base_alignment = std::max(base_alignment, (*alignments)[i]);
  }
  if (base_alignment > 1) {
    // Pad in case data pointer is not aligned
    base_byte_size += base_alignment;
  }

  // Allocate base buffer
  // TODO(stable-abi): needs torch::stable::empty(IntArrayRef, ScalarType, Device).
  auto base_buffer = std::make_shared<torch::stable::Tensor>(torch::stable::empty(
      {static_cast<int64_t>(base_byte_size)}, torch::headeronly::ScalarType::Byte, *device));
  uint8_t *base_ptr = static_cast<uint8_t *>(base_buffer->data_ptr());
  base_ptr =
      reinterpret_cast<uint8_t *>(roundup(reinterpret_cast<uintptr_t>(base_ptr), base_alignment));

  // Create views into base buffer
  std::vector<torch::stable::Tensor> out;
  out.reserve(n);
  std::vector<int64_t> shape_int64;
  for (size_t i = 0; i < n; ++i) {
    shape_int64.assign(shapes[i].begin(), shapes[i].end());
    if (byte_sizes[i] == 0) {
      // Work around problems with from_blob when constructing an
      // empty tensor. Passing a null pointer fails because it checks
      // that the pointer is on GPU. Passing a non-null pointer can
      // cause bugs in TE kernels.
      out.emplace_back(torch::stable::empty(shape_int64, dtypes[i], *device));
    } else {
      // Construct tensor with custom deleter to keep base buffer alive
      // TODO(stable-abi): needs torch::stable::from_blob(void*, IntArrayRef,
      // deleter, ScalarType, Device) with a custom deleter.
      out.emplace_back(torch::stable::from_blob(
          base_ptr + offsets[i], shape_int64, [base_buffer](void *) {}, dtypes[i], *device));
    }
  }
  return out;
}

}  // namespace pytorch
}  // namespace transformer_engine
