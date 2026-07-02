/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../../extensions.h"

namespace transformer_engine::pytorch {

std::tuple<torch::stable::Tensor, torch::stable::Tensor> multi_tensor_l2norm_cuda(
    int chunk_size, torch::stable::Tensor noop_flag,
    std::vector<std::vector<torch::stable::Tensor>> tensor_lists,
    std::optional<bool> per_tensor_python) {
  bool per_tensor = per_tensor_python.has_value() ? per_tensor_python.value() : false;

  // All bookkeeping tensors are float, on the same device as the first input.
  const auto& ref = tensor_lists[0][0];
  auto output = torch::stable::new_zeros(ref, {320}, torch::headeronly::ScalarType::Float);

  torch::stable::Tensor output_per_tensor;
  torch::stable::Tensor ret_per_tensor;
  auto ret = torch::stable::new_empty(output, {1});

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = -1;

  if (per_tensor) {
    for (int t = 0; t < ntensors; t++) {
      int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
      if (max_chunks_this_tensor > max_chunks_per_tensor)
        max_chunks_per_tensor = max_chunks_this_tensor;
    }
    output_per_tensor = torch::stable::new_zeros(
        ref, {static_cast<int64_t>(ntensors) * max_chunks_per_tensor},
        torch::headeronly::ScalarType::Float);
    ret_per_tensor = torch::stable::new_empty(ref, {ntensors}, torch::headeronly::ScalarType::Float);
  } else {
    output_per_tensor = torch::stable::new_empty(ref, {0}, torch::headeronly::ScalarType::Float);
    ret_per_tensor = torch::stable::new_empty(ref, {0}, torch::headeronly::ScalarType::Float);
  }

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto output_cu = makeTransformerEngineTensor(output);
  auto output_per_tensor_cu = makeTransformerEngineTensor(output_per_tensor);
  auto ret_cu = makeTransformerEngineTensor(ret);
  auto ret_per_tensor_cu = makeTransformerEngineTensor(ret_per_tensor);

  nvte_multi_tensor_l2norm_cuda(chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists,
                                num_tensors, output_cu.data(), output_per_tensor_cu.data(),
                                ret_cu.data(), ret_per_tensor_cu.data(), per_tensor,
                                max_chunks_per_tensor, getCurrentCUDAStream());

  return std::tuple<torch::stable::Tensor, torch::stable::Tensor>(ret, ret_per_tensor);
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor> multi_tensor_unscale_l2norm_cuda(
    int chunk_size, torch::stable::Tensor noop_flag,
    std::vector<std::vector<torch::stable::Tensor>> tensor_lists, torch::stable::Tensor inv_scale,
    std::optional<bool> per_tensor_python) {
  bool per_tensor = per_tensor_python.has_value() ? per_tensor_python.value() : false;

  const auto& ref = tensor_lists[0][0];
  auto output = torch::stable::new_zeros(ref, {320}, torch::headeronly::ScalarType::Float);

  torch::stable::Tensor output_per_tensor;
  torch::stable::Tensor ret_per_tensor;

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = -1;

  // Create output tensors for multi scale L2 norm kernel.
  if (per_tensor) {
    for (int t = 0; t < ntensors; t++) {
      int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
      if (max_chunks_this_tensor > max_chunks_per_tensor)
        max_chunks_per_tensor = max_chunks_this_tensor;
    }
    output_per_tensor = torch::stable::new_zeros(
        ref, {static_cast<int64_t>(ntensors) * max_chunks_per_tensor},
        torch::headeronly::ScalarType::Float);
    ret_per_tensor = torch::stable::new_empty(ref, {ntensors}, torch::headeronly::ScalarType::Float);
  } else {
    output_per_tensor = torch::stable::new_empty(ref, {0}, torch::headeronly::ScalarType::Float);
    ret_per_tensor = torch::stable::new_empty(ref, {0}, torch::headeronly::ScalarType::Float);
  }

  auto ret = torch::stable::new_empty(output, {1});

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto output_cu = makeTransformerEngineTensor(output);
  auto output_per_tensor_cu = makeTransformerEngineTensor(output_per_tensor);
  auto ret_cu = makeTransformerEngineTensor(ret);
  auto ret_per_tensor_cu = makeTransformerEngineTensor(ret_per_tensor);
  auto inv_scale_cu = makeTransformerEngineTensor(inv_scale);

  nvte_multi_tensor_unscale_l2norm_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors,
      output_cu.data(), output_per_tensor_cu.data(), ret_cu.data(), ret_per_tensor_cu.data(),
      inv_scale_cu.data(), per_tensor, max_chunks_per_tensor, getCurrentCUDAStream());

  return std::tuple<torch::stable::Tensor, torch::stable::Tensor>(ret, ret_per_tensor);
}

}  // namespace transformer_engine::pytorch
