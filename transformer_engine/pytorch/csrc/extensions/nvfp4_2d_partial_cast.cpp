/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "../extensions.h"

namespace nb = nanobind;

namespace transformer_engine::pytorch {

namespace {
// TODO(stable-abi): torch::stable::accelerator lacks a native cudaStream_t handle.
inline cudaStream_t current_cuda_stream() {
  return static_cast<cudaStream_t>(
      torch::stable::accelerator::getCurrentStream(
          torch::stable::accelerator::getCurrentDeviceIndex())
          .nativeHandle());
}
}  // namespace

void nvfp4_2d_compute_partial_amax(const torch::stable::Tensor& tensor, torch::stable::Tensor amax,
                                   size_t h, size_t w, size_t start_offset, size_t block_len) {
  STD_TORCH_CHECK(block_len == 16, "Currently only block_len = 16 is supported for NVFP4 2D");
  STD_TORCH_CHECK(amax.dim() == 2, "amax must be a 2D tensor");
  STD_TORCH_CHECK(amax.scalar_type() == torch::headeronly::ScalarType::Float,
                  "amax must be a float tensor");
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Float ||
                      tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                  "tensor must be a float or bfloat16 tensor");

  const TensorWrapper tensor_cu = makeTransformerEngineTensor(torch::stable::contiguous(tensor));
  TensorWrapper amax_cu = makeTransformerEngineTensor(amax);

  nvte_nvfp4_2d_compute_partial_amax(tensor_cu.data(), amax_cu.data(), h, w, amax.stride(0),
                                     amax.stride(1), start_offset, block_len,
                                     current_cuda_stream());
}

void nvfp4_2d_partial_cast(const torch::stable::Tensor& inp, nb::handle out,
                           const torch::stable::Tensor& scale,
                           const torch::stable::Tensor& global_scale, size_t h, size_t w,
                           size_t start_offset, size_t block_len) {
  STD_TORCH_CHECK(block_len == 16, "Currently only block_len = 16 is supported for NVFP4 2D");
  STD_TORCH_CHECK(scale.dim() == 2, "scale must be a 2D tensor");
  STD_TORCH_CHECK(scale.scalar_type() == torch::headeronly::ScalarType::Float,
                  "scale must be a float tensor");
  STD_TORCH_CHECK(global_scale.numel() == 1, "global_scale must be a scalar tensor");
  STD_TORCH_CHECK(global_scale.scalar_type() == torch::headeronly::ScalarType::Float,
                  "global_scale must be a float tensor");
  STD_TORCH_CHECK(inp.scalar_type() == torch::headeronly::ScalarType::Float ||
                      inp.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                  "input must be a float or bfloat16 tensor");

  const TensorWrapper inp_cu = makeTransformerEngineTensor(torch::stable::contiguous(inp));
  const TensorWrapper out_cu = makeTransformerEngineTensor(out, nb::none());
  const TensorWrapper scale_cu = makeTransformerEngineTensor(scale);
  const TensorWrapper global_scale_cu = makeTransformerEngineTensor(global_scale);

  nvte_nvfp4_2d_partial_cast(inp_cu.data(), out_cu.data(), scale_cu.data(), global_scale_cu.data(),
                             h, w, scale.stride(0), scale.stride(1), start_offset, block_len,
                             current_cuda_stream());
}

void nvfp4_multi_tensor_2d_partial_cast(
    std::vector<torch::stable::Tensor> inp_list, std::vector<torch::stable::Tensor> out_list,
    std::vector<torch::stable::Tensor> scale_list,
    std::vector<torch::stable::Tensor> global_scale_list, std::vector<int64_t> h_list,
    std::vector<int64_t> w_list, std::vector<int64_t> start_offset_list, int64_t block_len) {
  STD_TORCH_CHECK(block_len == 16, "Currently only block_len = 16 is supported for NVFP4 2D");

  const size_t num_tensors = inp_list.size();
  STD_TORCH_CHECK(out_list.size() == num_tensors, "out_list size mismatch");
  STD_TORCH_CHECK(scale_list.size() == num_tensors, "scale_list size mismatch");
  STD_TORCH_CHECK(global_scale_list.size() == num_tensors, "global_scale_list size mismatch");
  STD_TORCH_CHECK(h_list.size() == num_tensors, "h_list size mismatch");
  STD_TORCH_CHECK(w_list.size() == num_tensors, "w_list size mismatch");
  STD_TORCH_CHECK(start_offset_list.size() == num_tensors, "start_offset_list size mismatch");

  if (num_tensors == 0) {
    return;
  }

  auto stream = current_cuda_stream();

  for (size_t i = 0; i < num_tensors; ++i) {
    const auto& inp = inp_list[i];
    const auto& out = out_list[i];
    const auto& scale = scale_list[i];
    const auto& global_scale = global_scale_list[i];
    const size_t h = static_cast<size_t>(h_list[i]);
    const size_t w = static_cast<size_t>(w_list[i]);
    const size_t start_offset = static_cast<size_t>(start_offset_list[i]);

    STD_TORCH_CHECK(scale.dim() == 2, "scale must be a 2D tensor");
    STD_TORCH_CHECK(scale.scalar_type() == torch::headeronly::ScalarType::Float,
                    "scale must be a float tensor");
    STD_TORCH_CHECK(global_scale.numel() == 1, "global_scale must be a scalar tensor");
    STD_TORCH_CHECK(global_scale.scalar_type() == torch::headeronly::ScalarType::Float,
                    "global_scale must be a float tensor");
    STD_TORCH_CHECK(inp.scalar_type() == torch::headeronly::ScalarType::Float ||
                        inp.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                    "input must be a float or bfloat16 tensor");

    const TensorWrapper inp_cu = makeTransformerEngineTensor(torch::stable::contiguous(inp));
    const TensorWrapper out_cu = makeTransformerEngineTensor(out);
    const TensorWrapper scale_cu = makeTransformerEngineTensor(scale);
    const TensorWrapper global_scale_cu = makeTransformerEngineTensor(global_scale);

    nvte_nvfp4_2d_partial_cast(inp_cu.data(), out_cu.data(), scale_cu.data(),
                               global_scale_cu.data(), h, w, scale.stride(0), scale.stride(1),
                               start_offset, static_cast<size_t>(block_len), stream);
  }
}

void nvfp4_multi_tensor_compute_partial_amax(
    std::vector<torch::stable::Tensor> master_weight_list,
    std::vector<torch::stable::Tensor> partial_amax_list,
    std::vector<torch::stable::Tensor> global_amax_list, std::vector<int64_t> h_list,
    std::vector<int64_t> w_list, std::vector<int64_t> start_offset_list, int64_t block_len) {
  STD_TORCH_CHECK(block_len == 16, "Currently only block_len = 16 is supported for NVFP4 2D");

  const size_t num_tensors = master_weight_list.size();
  STD_TORCH_CHECK(partial_amax_list.size() == num_tensors, "partial_amax_list size mismatch");
  STD_TORCH_CHECK(global_amax_list.size() == num_tensors, "global_amax_list size mismatch");
  STD_TORCH_CHECK(h_list.size() == num_tensors, "h_list size mismatch");
  STD_TORCH_CHECK(w_list.size() == num_tensors, "w_list size mismatch");
  STD_TORCH_CHECK(start_offset_list.size() == num_tensors, "start_offset_list size mismatch");

  if (num_tensors == 0) {
    return;
  }

  auto stream = current_cuda_stream();

  for (size_t i = 0; i < num_tensors; ++i) {
    const auto& master_weight = master_weight_list[i];
    auto& partial_amax = partial_amax_list[i];
    auto& global_amax = global_amax_list[i];
    const size_t h = static_cast<size_t>(h_list[i]);
    const size_t w = static_cast<size_t>(w_list[i]);
    const size_t start_offset = static_cast<size_t>(start_offset_list[i]);

    STD_TORCH_CHECK(partial_amax.dim() == 2, "partial_amax must be a 2D tensor");
    STD_TORCH_CHECK(partial_amax.scalar_type() == torch::headeronly::ScalarType::Float,
                    "partial_amax must be a float tensor");
    STD_TORCH_CHECK(master_weight.scalar_type() == torch::headeronly::ScalarType::Float ||
                        master_weight.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                    "master_weight must be a float or bfloat16 tensor");
    STD_TORCH_CHECK(global_amax.scalar_type() == torch::headeronly::ScalarType::Float,
                    "global_amax must be a float tensor");
    STD_TORCH_CHECK(global_amax.numel() == 1, "global_amax must have exactly one element");

    // Compute partial amax (per-block amax)
    const TensorWrapper tensor_cu =
        makeTransformerEngineTensor(torch::stable::contiguous(master_weight));
    TensorWrapper amax_cu = makeTransformerEngineTensor(partial_amax);

    nvte_nvfp4_2d_compute_partial_amax(tensor_cu.data(), amax_cu.data(), h, w,
                                       partial_amax.stride(0), partial_amax.stride(1), start_offset,
                                       static_cast<size_t>(block_len), stream);

    // Compute global amax
    auto* global_amax_ptr = static_cast<float*>(global_amax.data_ptr());
    TensorWrapper fake_te_output(
        /*dptr=*/nullptr, tensor_cu.shape(), DType::kFloat32, global_amax_ptr);

    nvte_compute_amax(tensor_cu.data(), fake_te_output.data(), stream);
  }
}

}  // namespace transformer_engine::pytorch
