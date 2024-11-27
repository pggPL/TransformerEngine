/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"
#include "transformer_engine/transformer_engine.h"

at::Tensor swizzle_scaling_factors(at::Tensor input, at::Tensor scale_inv,
                                   std::vector<int64_t> scaling_mode) {
  using namespace transformer_engine::pytorch;

  NVTE_CHECK(input.element_size() == 1, "8-bit input required for swizzling scaling factors.");

  auto options = at::TensorOptions().dtype(scale_inv.dtype()).device(torch::kCUDA);
  auto swizzled_scale_inv = at::empty_like(scale_inv, options);

  void* scale_inv_dptr = getDataPtr(scale_inv, 0);
  void* swizzled_scale_inv_dptr = getDataPtr(swizzled_scale_inv, 0);

  NVTEScalingMode nvte_scaling_mode;
  if (scaling_mode[0] == -1 && scaling_mode[1] == -1) {
    nvte_scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
  } else {
    // TODO: error checking
    nvte_scaling_mode = NVTE_MXFP8_1D_SCALING;
  }

  // Construct Transformer Engine tensors
  DType dtype = DType::kFloat8E4M3;  // Use any 8 bit dummy type.
  auto input_cu =
      makeTransformerEngineTensor(input.data_ptr(), getTensorShape(input), dtype, nullptr, nullptr,
                                  scale_inv_dptr, getTensorShape(scale_inv), nvte_scaling_mode);
  auto output_cu = makeTransformerEngineTensor(
      input.data_ptr(), getTensorShape(input), dtype, nullptr, nullptr, swizzled_scale_inv_dptr,
      getTensorShape(swizzled_scale_inv), nvte_scaling_mode);

  // Launch kernel
  nvte_swizzle_scaling_factors(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return swizzled_scale_inv;
}
