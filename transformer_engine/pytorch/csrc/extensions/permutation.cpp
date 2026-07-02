/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "../extensions.h"

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

std::tuple<torch::stable::Tensor, torch::stable::Tensor, std::vector<torch::stable::Tensor>>
moe_permute_fwd(torch::stable::Tensor input, const DType dtype, torch::stable::Tensor indices,
                int64_t num_out_tokens, std::vector<torch::stable::Tensor> workspace,
                int64_t max_expanded_token_num) {
  const int num_tokens = input.size(0);
  int num_cols = input.size(1);
  const int topK = indices.size(1);

  const auto device = torch::stable::Device(torch::headeronly::DeviceType::CUDA);

  // Initialize the workspace on the first run
  if (workspace.empty()) {
    // TODO(stable-abi): needs torch::stable::empty(IntArrayRef, ScalarType, Device).
    torch::stable::Tensor sorted_indices =
        torch::stable::empty({max_expanded_token_num}, torch::headeronly::ScalarType::Int, std::nullopt, device);
    // TODO(stable-abi): needs torch::stable::arange(start, end, step, ScalarType, Device)
    // (replacement for torch::range over [0, max_expanded_token_num - 1]).
    torch::stable::Tensor row_id = torch::stable::arange(
        0, max_expanded_token_num, 1, torch::headeronly::ScalarType::Int, std::nullopt, device);
    torch::stable::Tensor sorted_row_id =
        torch::stable::empty({max_expanded_token_num}, torch::headeronly::ScalarType::Int, std::nullopt, device);

    size_t temp_storage_bytes = 0;
    nvte_device_radix_sort_pairs(nullptr, &temp_storage_bytes, nullptr, nullptr, nullptr, nullptr,
                                 max_expanded_token_num);
    torch::stable::Tensor temp_storage = torch::stable::empty(
        {static_cast<int64_t>(temp_storage_bytes)}, torch::headeronly::ScalarType::Char, std::nullopt, device);

    workspace.push_back(sorted_indices);
    workspace.push_back(row_id);
    workspace.push_back(sorted_row_id);
    workspace.push_back(temp_storage);
  }

  void *indices_ptr = getDataPtr(indices, 0);
  void *sorted_indices_ptr = getDataPtr(workspace[0], 0);
  void *row_id_ptr = getDataPtr(workspace[1], 0);
  void *sorted_row_id_ptr = getDataPtr(workspace[2], 0);

  void *d_temp_storage = getDataPtr(workspace[3], 0);
  size_t temp_storage_bytes = std::numeric_limits<size_t>::max();

  nvte_device_radix_sort_pairs(
      d_temp_storage, &temp_storage_bytes, reinterpret_cast<int *>(indices_ptr),
      reinterpret_cast<int *>(sorted_indices_ptr), reinterpret_cast<int *>(row_id_ptr),
      reinterpret_cast<int *>(sorted_row_id_ptr), num_tokens * topK);

  // Output buffer alloc
  num_out_tokens = (num_out_tokens > 0) ? num_out_tokens : num_tokens * topK;
  torch::stable::Tensor permuted_output =
      torch::stable::empty({num_out_tokens, num_cols}, input.scalar_type(), std::nullopt, device);
  torch::stable::Tensor row_id_map = torch::stable::empty(
      {static_cast<int64_t>(num_tokens) * topK}, torch::headeronly::ScalarType::Int, std::nullopt, device);

  auto stream = current_cuda_stream();

  auto input_cu = makeTransformerEngineTensor(
      input.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(input.size(0)), static_cast<size_t>(num_cols)},
      dtype);
  auto permuted_output_cu =
      makeTransformerEngineTensor(permuted_output.data_ptr(),
                                  std::vector<size_t>{static_cast<size_t>(permuted_output.size(0)),
                                                      static_cast<size_t>(num_cols)},
                                  dtype);
  auto sorted_row_id_cu = makeTransformerEngineTensor(
      sorted_row_id_ptr, std::vector<size_t>{static_cast<size_t>(num_tokens * topK)},
      DType::kInt32);
  auto row_id_map_cu = makeTransformerEngineTensor(row_id_map);

  nvte_permute(input_cu.data(), permuted_output_cu.data(), sorted_row_id_cu.data(),
               row_id_map_cu.data(), TensorWrapper().data(), TensorWrapper().data(),
               TensorWrapper().data(), num_tokens, topK, num_cols, num_out_tokens, stream);

  return std::make_tuple(permuted_output, row_id_map, workspace);
}

torch::stable::Tensor moe_permute_bwd(torch::stable::Tensor input, const DType dtype,
                                      torch::stable::Tensor row_id_map, torch::stable::Tensor prob,
                                      int64_t num_tokens, int64_t topK) {
  return moe_unpermute_fwd(input, dtype, row_id_map, prob, num_tokens, topK);
}

torch::stable::Tensor moe_unpermute_fwd(torch::stable::Tensor input, const DType dtype,
                                        torch::stable::Tensor row_id_map,
                                        torch::stable::Tensor prob, int64_t num_tokens,
                                        int64_t topK) {
  int num_cols = input.size(1);

  const auto device = torch::stable::Device(torch::headeronly::DeviceType::CUDA);

  // Output buffer alloc
  torch::stable::Tensor unpermuted_output =
      torch::stable::empty({num_tokens, num_cols}, input.scalar_type(), std::nullopt, device);

  auto stream = current_cuda_stream();

  auto input_cu = makeTransformerEngineTensor(
      input.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(input.size(0)), static_cast<size_t>(num_cols)},
      dtype);
  auto unpermuted_output_cu = makeTransformerEngineTensor(
      unpermuted_output.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(unpermuted_output.size(0)),
                          static_cast<size_t>(num_cols)},
      dtype);
  auto row_id_map_cu = makeTransformerEngineTensor(row_id_map);
  auto prob_cu = makeTransformerEngineTensor(prob);

  nvte_unpermute(input_cu.data(), unpermuted_output_cu.data(), row_id_map_cu.data(), prob_cu.data(),
                 num_tokens, topK, num_cols, stream);

  return unpermuted_output;
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor> moe_unpermute_bwd(
    torch::stable::Tensor input_bwd, torch::stable::Tensor input_fwd, const DType dtype,
    torch::stable::Tensor row_id_map, torch::stable::Tensor prob) {
  const int topK = (prob.numel() > 0) ? prob.size(1) : 1;
  const int num_tokens = (prob.numel() > 0) ? prob.size(0) : row_id_map.size(0);
  int num_cols = input_bwd.size(1);

  const auto device = torch::stable::Device(torch::headeronly::DeviceType::CUDA);

  // Output buffer alloc
  torch::stable::Tensor act_grad =
      torch::stable::empty({input_fwd.size(0), num_cols}, input_bwd.scalar_type(), std::nullopt, device);
  torch::stable::Tensor prob_grad = torch::stable::empty(
      {num_tokens, topK}, torch::headeronly::ScalarType::Float, std::nullopt, device);

  auto stream = current_cuda_stream();

  auto input_bwd_cu = makeTransformerEngineTensor(
      input_bwd.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(input_bwd.size(0)), static_cast<size_t>(num_cols)},
      dtype);
  auto act_grad_cu = makeTransformerEngineTensor(
      act_grad.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(act_grad.size(0)), static_cast<size_t>(num_cols)},
      dtype);
  auto input_fwd_cu = makeTransformerEngineTensor(
      input_fwd.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(input_fwd.size(0)), static_cast<size_t>(num_cols)},
      dtype);
  auto row_id_map_cu = makeTransformerEngineTensor(row_id_map);
  auto prob_cu = makeTransformerEngineTensor(prob);
  auto prob_grad_cu = makeTransformerEngineTensor(prob_grad);

  nvte_permute(input_bwd_cu.data(), act_grad_cu.data(), TensorWrapper().data(),
               row_id_map_cu.data(), prob_cu.data(), prob_grad_cu.data(), input_fwd_cu.data(),
               num_tokens, topK, num_cols, 0, stream);

  return std::make_tuple(act_grad, prob_grad);
}

}  // namespace transformer_engine::pytorch
