/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_BULK_TENSOR_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_BULK_TENSOR_H_

#include <ATen/ATen.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace transformer_engine::pytorch {

using BulkTensorRegion = std::tuple<int64_t, std::vector<int64_t>, at::ScalarType, int64_t>;
using BulkTensorLayout = std::vector<BulkTensorRegion>;

std::vector<at::Tensor> unpack_bulk_tensor(const at::Tensor &buffer,
                                           const BulkTensorLayout &layout);

}  // namespace transformer_engine::pytorch

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_BULK_TENSOR_H_
