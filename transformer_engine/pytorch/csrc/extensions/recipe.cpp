/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <string>

#include "../extensions.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine::pytorch {

namespace {
// TODO(stable-abi): torch::stable::accelerator lacks a native cudaStream_t handle.
inline cudaStream_t current_cuda_stream() {
  return static_cast<cudaStream_t>(
      torch::stable::accelerator::getCurrentStream(
          torch::stable::accelerator::getCurrentDeviceIndex())
          .stream());
}
}  // namespace

void compute_amax(const torch::stable::Tensor& tensor, torch::stable::Tensor& amax) {
  auto input_tensor = torch::stable::contiguous(tensor);
  const TensorWrapper& te_input = makeTransformerEngineTensor(input_tensor);

  STD_TORCH_CHECK(amax.scalar_type() == torch::headeronly::ScalarType::Float,
                  "amax must be a float tensor");
  STD_TORCH_CHECK(amax.numel() == 1, "amax must have exactly one element");
  auto* amax_ptr = static_cast<float*>(amax.data_ptr());
  TensorWrapper fake_te_output(
      /*dptr=*/nullptr, te_input.shape(),
      DType::kFloat32,  // It doesn't matter because we only compute amax.
      amax_ptr);

  nvte_compute_amax(te_input.data(), fake_te_output.data(), current_cuda_stream());
}

void fused_amax_and_scale_update_after_reduction(
    const torch::stable::Tensor& amax_reduction_buffer,
    std::vector<torch::stable::Tensor> amax_histories, std::vector<torch::stable::Tensor> scales,
    const std::string& amax_compute_algo, DType fp8_dtype, float margin) {
  size_t num_tensors = amax_histories.size();

  // Allocate amax history and scale NVTETensors as batches
  MultiTensorWrapper te_amax_histories(num_tensors, NVTE_DELAYED_TENSOR_SCALING);
  MultiTensorWrapper te_scales(num_tensors, NVTE_DELAYED_TENSOR_SCALING);

  for (size_t i = 0; i < num_tensors; i++) {
    std::vector<size_t> amax_shape_vec = getTensorShape(amax_histories[i]);
    NVTEShape amax_shape = nvte_make_shape(amax_shape_vec.data(), amax_shape_vec.size());
    NVTEBasicTensor amax_history_data = {amax_histories[i].data_ptr(),
                                         static_cast<NVTEDType>(DType::kFloat32), amax_shape};
    nvte_set_tensor_param_v2(te_amax_histories[i], kNVTERowwiseData, &amax_history_data,
                             sizeof(amax_history_data));

    std::vector<size_t> scale_shape_vec = getTensorShape(scales[i]);
    NVTEShape scale_shape = nvte_make_shape(scale_shape_vec.data(), scale_shape_vec.size());
    NVTEBasicTensor scale_data = {scales[i].data_ptr(), static_cast<NVTEDType>(DType::kFloat32),
                                  scale_shape};
    nvte_set_tensor_param_v2(te_scales[i], kNVTERowwiseData, &scale_data, sizeof(scale_data));
  }
  // The recipe function takes std::vector<NVTETensor> by value, so
  // construct fresh vectors from the batches.
  nvte_delayed_scaling_recipe_amax_and_scale_update_after_reduction(
      makeTransformerEngineTensor(amax_reduction_buffer).data(),
      std::vector<NVTETensor>(te_amax_histories.begin(), te_amax_histories.end()),
      std::vector<NVTETensor>(te_scales.begin(), te_scales.end()), amax_compute_algo.c_str(),
      static_cast<NVTEDType>(fp8_dtype), margin, current_cuda_stream());
}

}  // namespace transformer_engine::pytorch
