/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "bulk_tensor.h"

#include <limits>

namespace transformer_engine::pytorch {

std::vector<at::Tensor> unpack_bulk_tensor(const at::Tensor &buffer,
                                           const BulkTensorLayout &layout) {
  TORCH_CHECK(buffer.scalar_type() == at::kByte && buffer.dim() == 1 && buffer.is_contiguous() &&
                  buffer.storage_offset() == 0,
              "BulkTensor requires a contiguous one-dimensional uint8 buffer with zero storage "
              "offset");
  TORCH_CHECK(buffer.device().is_cpu() || buffer.device().is_cuda(),
              "BulkTensor native unpacking requires a CPU or CUDA buffer");
  std::vector<at::Tensor> tensors;
  tensors.reserve(layout.size());
  for (const auto &[offset, shape, dtype, alignment] : layout) {
    TORCH_CHECK(!c10::isQIntType(dtype) && dtype != at::ScalarType::Undefined,
                "BulkTensor does not support quantized or undefined dtypes");
    const auto itemsize = static_cast<int64_t>(c10::elementSize(dtype));
    TORCH_CHECK(alignment > 0 && (alignment & (alignment - 1)) == 0 && alignment % itemsize == 0,
                "BulkTensor alignment must be a power of two covering the dtype");
    int64_t nbytes = itemsize;
    for (const auto dim : shape) {
      TORCH_CHECK(dim >= 0 && (dim == 0 || nbytes <= std::numeric_limits<int64_t>::max() / dim),
                  "BulkTensor has an invalid or overflowing shape");
      nbytes *= dim;
    }
    TORCH_CHECK(offset >= 0 && offset <= buffer.numel() && nbytes <= buffer.numel() - offset,
                "BulkTensor region exceeds its buffer");
    TORCH_CHECK(offset % alignment == 0, "BulkTensor region has an unaligned byte offset");
    if (nbytes == 0) {
      // TE kernels expect empty operands to have no data pointer.
      tensors.emplace_back(at::empty(shape, buffer.options().dtype(dtype)));
      continue;
    }
    TORCH_CHECK(reinterpret_cast<uintptr_t>(buffer.data_ptr()) % alignment == 0,
                "BulkTensor buffer does not satisfy the region alignment");
    tensors.emplace_back(buffer.narrow(0, offset, nbytes).view(dtype).view(shape));
  }
  return tensors;
}

}  // namespace transformer_engine::pytorch
