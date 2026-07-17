/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Implementation of the single TE<->PyTorch binary boundary. This is the only
// .cpp that is permitted to name at::/c10::/torch:: factory functions and dtype
// constants (see torch_backend.h and qa/L0_pytorch_lint/check_torch_boundary.sh).

#include "torch_backend.h"

#include "common/util/logging.h"

namespace transformer_engine::pytorch {

ScalarType GetATenDType(transformer_engine::DType t) {
  switch (t) {
    case transformer_engine::DType::kInt16:
      return torch::kInt16;
    case transformer_engine::DType::kInt32:
      return torch::kInt32;
    case transformer_engine::DType::kInt64:
      return torch::kInt64;
    case transformer_engine::DType::kFloat32:
      return at::kFloat;
    case transformer_engine::DType::kFloat16:
      return at::kHalf;
    case transformer_engine::DType::kBFloat16:
      return at::kBFloat16;
    case transformer_engine::DType::kByte:
      return at::kByte;
    case transformer_engine::DType::kFloat8E4M3:
      return at::kFloat8_e4m3fn;
    case transformer_engine::DType::kFloat8E5M2:
      return at::kFloat8_e5m2;
    case transformer_engine::DType::kFloat8E8M0:
      return at::kByte;  // e8m0 dtype requires PyTorch 2.7.0+
    default:
      NVTE_ERROR("Invalid type (", static_cast<int>(t), ").");
  }
}

transformer_engine::DType GetTransformerEngineDType(ScalarType t) {
  switch (t) {
    case at::kFloat8_e4m3fn:
      return transformer_engine::DType::kFloat8E4M3;
    case at::kFloat8_e5m2:
      return transformer_engine::DType::kFloat8E5M2;
    case at::kHalf:
      return transformer_engine::DType::kFloat16;
    case at::kFloat:
      return transformer_engine::DType::kFloat32;
    case at::kBFloat16:
      return transformer_engine::DType::kBFloat16;
    case at::kBool:
      return transformer_engine::DType::kByte;
    case torch::kByte:
      return transformer_engine::DType::kByte;
    case torch::kInt16:
      return transformer_engine::DType::kInt16;
    case torch::kInt32:
      return transformer_engine::DType::kInt32;
    case torch::kInt64:
      return transformer_engine::DType::kInt64;
    default:
      NVTE_ERROR("Invalid type (", static_cast<int>(t), ").");
  }
}

Tensor new_cuda_tensor(const std::vector<int64_t>& shape, ScalarType dtype, bool zero_init) {
  c10::IntArrayRef ar_shape(shape);
  if (zero_init) {
    return at::zeros(ar_shape, at::CUDA(dtype));
  }
  return at::empty(ar_shape, at::CUDA(dtype));
}

cudaStream_t getCurrentCUDAStream() { return at::cuda::getCurrentCUDAStream(); }

}  // namespace transformer_engine::pytorch
