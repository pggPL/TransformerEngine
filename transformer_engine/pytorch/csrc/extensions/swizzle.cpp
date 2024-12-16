/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"
#include "transformer_engine/transformer_engine.h"

void swizzle_scaling_factors(transformer_engine::TensorWrapper& input, bool rowwise) {
  using namespace transformer_engine::pytorch;

  if (input.scaling_mode() == NVTE_INVALID_SCALING) {
    NVTE_ERROR("Invalid scaling mode for swizzle.");
  } else if (input.scaling_mode() == NVTE_DELAYED_TENSOR_SCALING) {
    return;
  }

  NVTE_CHECK(input.element_size() == 1, "8-bit input required for swizzling scaling factors.");

  NVTEBasicTensor scale_inv;
  if (rowwise) {
    scale_inv = input.get_rowwise_scale_inv();
  } else {
    scale_inv = input.get_columnwise_scale_inv();
  }

  auto input_shape = nvte_shape_to_vector(input.shape());
  auto scale_inv_shape = nvte_shape_to_vector(scale_inv.shape);

  // Allocate memory for swizzled output.
  auto options = at::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
  std::vector<int64_t> scale_inv_shape_int;
  for (size_t i = 0; i < scale_inv_shape.size(); ++i) {
    scale_inv_shape_int.push_back(static_cast<int64_t>(scale_inv_shape[i]));
  }
  auto swizzled_scale_inv = at::empty(scale_inv_shape_int, options);
  void* scale_inv_dptr = scale_inv.data_ptr;
  void* swizzled_scale_inv_dptr = getDataPtr(swizzled_scale_inv, 0);

  // Construct Transformer Engine tensors
  DType dtype = DType::kFloat8E4M3;  // Use any 8 bit dummy type.
  auto input_cu =
      makeTransformerEngineTensor(input.dptr(), input_shape, dtype, nullptr, nullptr,
                                  scale_inv_dptr, scale_inv_shape, NVTE_MXFP8_1D_SCALING);
  auto output_cu =
      makeTransformerEngineTensor(input.dptr(), input_shape, dtype, nullptr, nullptr,
                                  swizzled_scale_inv_dptr, scale_inv_shape, NVTE_MXFP8_1D_SCALING);

  // Launch kernel
  nvte_swizzle_scaling_factors(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  if (rowwise) {
    input.set_rowwise_scale_inv(swizzled_scale_inv_dptr, DType::kFloat8E8M0, scale_inv_shape);
  } else {
    input.set_columnwise_scale_inv(swizzled_scale_inv_dptr, DType::kFloat8E8M0, scale_inv_shape);
  }
}
