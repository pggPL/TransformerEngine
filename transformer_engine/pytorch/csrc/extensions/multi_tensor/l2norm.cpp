/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../../extensions.h"

namespace transformer_engine::pytorch {

std::tuple<Tensor, Tensor> multi_tensor_l2norm_cuda(
    int chunk_size, Tensor noop_flag, std::vector<std::vector<Tensor>> tensor_lists,
    std::optional<bool> per_tensor_python) {
  bool per_tensor = per_tensor_python.has_value() ? per_tensor_python.value() : false;

  auto float_options = tensor_lists[0][0].options().dtype(kFloat);
  auto output = zeros({320}, float_options);

  Tensor output_per_tensor;
  Tensor ret_per_tensor;
  auto ret = empty({1}, output.options());

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = -1;

  if (per_tensor) {
    for (int t = 0; t < ntensors; t++) {
      int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
      if (max_chunks_this_tensor > max_chunks_per_tensor)
        max_chunks_per_tensor = max_chunks_this_tensor;
    }
    output_per_tensor = zeros({ntensors * max_chunks_per_tensor}, float_options);
    ret_per_tensor = empty({ntensors}, float_options);
  } else {
    output_per_tensor = empty({0}, float_options);
    ret_per_tensor = empty({0}, float_options);
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

  return std::tuple<Tensor, Tensor>(ret, ret_per_tensor);
}

std::tuple<Tensor, Tensor> multi_tensor_unscale_l2norm_cuda(
    int chunk_size, Tensor noop_flag, std::vector<std::vector<Tensor>> tensor_lists,
    Tensor inv_scale, std::optional<bool> per_tensor_python) {
  bool per_tensor = per_tensor_python.has_value() ? per_tensor_python.value() : false;

  auto float_options = tensor_lists[0][0].options().dtype(kFloat);
  auto output = zeros({320}, float_options);

  Tensor output_per_tensor;
  Tensor ret_per_tensor;

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = -1;

  // Create output tensors for multi scale L2 norm kernel.
  if (per_tensor) {
    for (int t = 0; t < ntensors; t++) {
      int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
      if (max_chunks_this_tensor > max_chunks_per_tensor)
        max_chunks_per_tensor = max_chunks_this_tensor;
    }
    output_per_tensor = zeros({ntensors * max_chunks_per_tensor}, float_options);
    ret_per_tensor = empty({ntensors}, float_options);
  } else {
    output_per_tensor = empty({0}, float_options);
    ret_per_tensor = empty({0}, float_options);
  }

  auto ret = empty({1}, output.options());

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

  return std::tuple<Tensor, Tensor>(ret, ret_per_tensor);
}

}  // namespace transformer_engine::pytorch
