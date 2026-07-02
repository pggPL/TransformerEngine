/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <numeric>

#include "../extensions.h"
#include "common.h"

namespace nb = nanobind;

namespace transformer_engine::pytorch {

namespace {
inline std::vector<int64_t> shape_vec(const torch::stable::Tensor &t) {
  std::vector<int64_t> s(t.dim());
  for (int i = 0; i < t.dim(); ++i) s[i] = t.size(i);
  return s;
}
}  // namespace

static std::map<std::string, int> score_function_map = {
    {"sigmoid", 0}, {"softmax", 1}, {"sqrtsoftplus", 2}};

static int get_score_function_value(const std::string &score_function) {
  auto it = score_function_map.find(score_function);
  STD_TORCH_CHECK(it != score_function_map.end(),
                  "score_function must be softmax, sigmoid or sqrtsoftplus for router fusion, got ",
                  score_function);
  return it->second;
}

// Allocate a routing_map output tensor (on the same device as `ref`):
//   BYTEMAP   -> bool [*leading_dims, num_experts]
//   BITMAP_U8 -> uint8[*leading_dims, ceil(num_experts/8)], LSB-first
static torch::stable::Tensor allocate_routing_map(const torch::stable::Tensor &ref,
                                                  const std::vector<int64_t> &leading_dims,
                                                  int64_t num_experts, int routing_map_format) {
  std::vector<int64_t> shape(leading_dims.begin(), leading_dims.end());
  if (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8) {
    shape.push_back((num_experts + 7) / 8);
    // TODO(stable-abi): needs new_empty dtype override overload.
    return torch::stable::new_empty(ref, shape, torch::headeronly::ScalarType::Byte);
  }
  shape.push_back(num_experts);
  return torch::stable::new_empty(ref, shape, torch::headeronly::ScalarType::Bool);
}

static void check_routing_map_format(int routing_map_format) {
  STD_TORCH_CHECK(
      routing_map_format == NVTE_ROUTING_MAP_FORMAT_BYTEMAP ||
          routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8,
      "routing_map_format must be BYTEMAP (0) or BITMAP_U8 (1), got ", routing_map_format);
}

static bool is_supported_dense_index_dtype(torch::headeronly::ScalarType dtype) {
  return dtype == torch::headeronly::ScalarType::Short ||
         dtype == torch::headeronly::ScalarType::Int ||
         dtype == torch::headeronly::ScalarType::Long;
}

static void check_dense_topk_indices(const torch::stable::Tensor &topk_indices,
                                     const torch::stable::Tensor &ref,
                                     const std::vector<int64_t> &leading_dims, int topk) {
  STD_TORCH_CHECK(topk_indices.is_cuda(), "topk_indices must be a CUDA tensor");
  STD_TORCH_CHECK(topk_indices.get_device() == ref.get_device(),
                  "topk_indices must be on the same device as ", "the logits/grad tensor");
  STD_TORCH_CHECK(topk_indices.is_contiguous(), "topk_indices must be contiguous");
  STD_TORCH_CHECK(is_supported_dense_index_dtype(topk_indices.scalar_type()),
                  "topk_indices dtype must be int16, int32, or int64");
  std::vector<int64_t> expected_shape(leading_dims.begin(), leading_dims.end());
  expected_shape.push_back(static_cast<int64_t>(topk));
  STD_TORCH_CHECK(shape_vec(topk_indices) == expected_shape,
                  "topk_indices shape must be [*leading_dims, topk]");
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
fused_topk_with_score_function_fwd(torch::stable::Tensor logits, int topk, bool use_pre_softmax,
                                   std::optional<int> num_groups, std::optional<int> group_topk,
                                   std::optional<float> scaling_factor, std::string score_function,
                                   std::optional<torch::stable::Tensor> expert_bias,
                                   int routing_map_format,
                                   std::optional<torch::stable::Tensor> topk_indices) {
  check_routing_map_format(routing_map_format);
  STD_TORCH_CHECK(logits.dim() >= 1, "logits must have at least 1 dim");
  STD_TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  auto sizes = shape_vec(logits);
  int64_t num_experts = sizes.back();
  int64_t num_tokens =
      std::accumulate(sizes.begin(), sizes.end() - 1, int64_t{1}, std::multiplies<int64_t>());
  STD_TORCH_CHECK(num_tokens > 0 && num_experts > 0,
                  "num_tokens and num_experts must be greater than 0");
  STD_TORCH_CHECK(topk > 0 && topk <= num_experts, "topk must be in [1, num_experts]");
  const std::vector<int64_t> leading_dims(sizes.begin(), sizes.end() - 1);
  // Expert bias only happens at the sigmoid case
  if (expert_bias.has_value()) {
    STD_TORCH_CHECK(score_function == "sigmoid" || score_function == "sqrtsoftplus",
                    "score_function must be sigmoid or sqrtsoftplus when expert_bias is not None");
    STD_TORCH_CHECK(expert_bias.value().scalar_type() == torch::headeronly::ScalarType::Float,
                    "expert_bias must be a float32 tensor");
  }
  STD_TORCH_CHECK(score_function == "softmax" || score_function == "sigmoid" ||
                      score_function == "sqrtsoftplus",
                  "score_function must be softmax, sigmoid or sqrtsoftplus for router fusion");
  if (score_function == "sigmoid" || score_function == "sqrtsoftplus") {
    use_pre_softmax = false;  // Pre-softmax only happens at the softmax case
  }
  if (topk_indices.has_value()) {
    STD_TORCH_CHECK(routing_map_format == NVTE_ROUTING_MAP_FORMAT_BYTEMAP,
                    "topk_indices output cannot be combined with non-default routing_map_format; "
                    "dense top-k indices are returned instead of a routing map.");
    check_dense_topk_indices(topk_indices.value(), logits, leading_dims, topk);
  }

  int group_topk_value = group_topk.has_value() ? group_topk.value() : -1;
  int num_groups_value = num_groups.has_value() ? num_groups.value() : -1;
  float scaling_factor_value = scaling_factor.has_value() ? scaling_factor.value() : 1.0f;

  torch::stable::Tensor probs = torch::stable::new_empty(logits, sizes);
  torch::stable::Tensor routing_map =
      topk_indices.has_value() ? topk_indices.value()
                               : allocate_routing_map(logits, leading_dims, num_experts,
                                                      routing_map_format);
  torch::stable::Tensor intermediate_output =
      torch::stable::new_empty(logits, sizes, torch::headeronly::ScalarType::Float);

  const std::vector<size_t> shape_2d = {static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  const std::vector<size_t> routing_map_shape_2d =
      topk_indices.has_value()
          ? std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(topk)}
          : std::vector<size_t>{
                static_cast<size_t>(num_tokens),
                static_cast<size_t>(routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8
                                        ? (num_experts + 7) / 8
                                        : num_experts)};
  auto logits_dtype = GetTransformerEngineDType(logits.scalar_type());
  auto routing_map_dtype = GetTransformerEngineDType(routing_map.scalar_type());

  auto logits_cu = makeTransformerEngineTensor(logits.data_ptr(), shape_2d, logits_dtype);
  auto probs_cu = makeTransformerEngineTensor(probs.data_ptr(), shape_2d, logits_dtype);
  auto routing_map_cu =
      makeTransformerEngineTensor(routing_map.data_ptr(), routing_map_shape_2d, routing_map_dtype);
  auto intermediate_output_cu =
      makeTransformerEngineTensor(intermediate_output.data_ptr(), shape_2d, DType::kFloat32);
  auto expert_bias_cu = TensorWrapper();  // empty expert_bias_cu tensor
  if (expert_bias.has_value()) {
    expert_bias_cu = makeTransformerEngineTensor(expert_bias.value());
  }

  if (topk_indices.has_value()) {
    nvte_fused_topk_with_score_function_forward_with_indices(
        logits_cu.data(), static_cast<int>(num_tokens), static_cast<int>(num_experts), topk,
        use_pre_softmax, num_groups_value, group_topk_value, scaling_factor_value,
        get_score_function_value(score_function), expert_bias_cu.data(), probs_cu.data(),
        routing_map_cu.data(), intermediate_output_cu.data(), getCurrentCUDAStream());
  } else {
    nvte_fused_topk_with_score_function_forward_v2(
        logits_cu.data(), static_cast<int>(num_tokens), static_cast<int>(num_experts), topk,
        use_pre_softmax, num_groups_value, group_topk_value, scaling_factor_value,
        get_score_function_value(score_function), expert_bias_cu.data(), probs_cu.data(),
        routing_map_cu.data(), static_cast<NVTERoutingMapFormat>(routing_map_format),
        intermediate_output_cu.data(), getCurrentCUDAStream());
  }

  return std::make_tuple(probs, routing_map, intermediate_output);
}

void fused_topk_with_score_function_bwd(torch::stable::Tensor routing_map,
                                        torch::stable::Tensor intermediate_output,
                                        torch::stable::Tensor grad_probs,
                                        torch::stable::Tensor grad_logits, int topk,
                                        bool use_pre_softmax, std::optional<float> scaling_factor,
                                        std::string score_function, bool use_dense_indices,
                                        int routing_map_format) {
  if (use_dense_indices) {
    STD_TORCH_CHECK(routing_map_format == NVTE_ROUTING_MAP_FORMAT_BYTEMAP,
                    "use_dense_indices cannot be combined with non-default routing_map_format; "
                    "dense top-k indices are consumed instead of a routing map.");
  } else {
    check_routing_map_format(routing_map_format);
  }
  STD_TORCH_CHECK(grad_probs.dim() >= 1, "grad_probs must have at least 1 dim");
  STD_TORCH_CHECK(grad_probs.is_contiguous(), "grad_probs must be contiguous");
  STD_TORCH_CHECK(grad_logits.is_contiguous(), "grad_logits must be contiguous");
  auto sizes = shape_vec(grad_probs);
  int64_t num_experts = sizes.back();
  int64_t num_tokens =
      std::accumulate(sizes.begin(), sizes.end() - 1, int64_t{1}, std::multiplies<int64_t>());
  STD_TORCH_CHECK(num_tokens > 0 && num_experts > 0,
                  "num_tokens and num_experts must be greater than 0");
  STD_TORCH_CHECK(topk > 0 && topk <= num_experts, "topk must be in [1, num_experts]");
  if (use_dense_indices) {
    const std::vector<int64_t> leading_dims(sizes.begin(), sizes.end() - 1);
    check_dense_topk_indices(routing_map, grad_probs, leading_dims, topk);
  }

  auto scaling_factor_value = scaling_factor.has_value() ? scaling_factor.value() : 1.0f;
  auto score_function_value = get_score_function_value(score_function);

  const std::vector<size_t> shape_2d = {static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  const std::vector<size_t> routing_map_shape_2d = {
      static_cast<size_t>(num_tokens),
      static_cast<size_t>(use_dense_indices
                              ? topk
                              : (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8
                                     ? (num_experts + 7) / 8
                                     : num_experts))};
  auto grad_dtype = GetTransformerEngineDType(grad_probs.scalar_type());
  auto routing_map_dtype = GetTransformerEngineDType(routing_map.scalar_type());

  auto routing_map_cu =
      makeTransformerEngineTensor(routing_map.data_ptr(), routing_map_shape_2d, routing_map_dtype);
  auto intermediate_output_cu =
      makeTransformerEngineTensor(intermediate_output.data_ptr(), shape_2d, DType::kFloat32);
  auto grad_probs_cu = makeTransformerEngineTensor(grad_probs.data_ptr(), shape_2d, grad_dtype);
  auto grad_logits_cu = makeTransformerEngineTensor(grad_logits.data_ptr(), shape_2d, grad_dtype);

  if (use_dense_indices) {
    nvte_fused_topk_with_score_function_backward_with_indices(
        routing_map_cu.data(), intermediate_output_cu.data(), grad_probs_cu.data(),
        static_cast<int>(num_tokens), static_cast<int>(num_experts), topk, use_pre_softmax,
        scaling_factor_value, score_function_value, grad_logits_cu.data(), getCurrentCUDAStream());
  } else {
    nvte_fused_topk_with_score_function_backward_v2(
        routing_map_cu.data(), static_cast<NVTERoutingMapFormat>(routing_map_format),
        intermediate_output_cu.data(), grad_probs_cu.data(), static_cast<int>(num_tokens),
        static_cast<int>(num_experts), topk, use_pre_softmax, scaling_factor_value,
        score_function_value, grad_logits_cu.data(), getCurrentCUDAStream());
  }
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
fused_score_for_moe_aux_loss_fwd(torch::stable::Tensor logits, int topk, std::string score_function,
                                 int routing_map_format) {
  check_routing_map_format(routing_map_format);
  STD_TORCH_CHECK(logits.dim() >= 1, "logits must have at least 1 dim");
  STD_TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  auto sizes = shape_vec(logits);
  int64_t num_experts = sizes.back();
  int64_t num_tokens =
      std::accumulate(sizes.begin(), sizes.end() - 1, int64_t{1}, std::multiplies<int64_t>());
  STD_TORCH_CHECK(num_tokens > 0 && num_experts > 0,
                  "num_tokens and num_experts must be greater than 0");
  STD_TORCH_CHECK(topk > 0, "topk must be greater than 0");
  STD_TORCH_CHECK(score_function == "softmax" || score_function == "sigmoid" ||
                      score_function == "sqrtsoftplus",
                  "score_function must be softmax, sigmoid or sqrtsoftplus for router fusion");
  int score_function_value = get_score_function_value(score_function);
  const std::vector<int64_t> leading_dims(sizes.begin(), sizes.end() - 1);

  torch::stable::Tensor scores =
      torch::stable::new_empty(logits, sizes, torch::headeronly::ScalarType::Float);
  torch::stable::Tensor routing_map =
      allocate_routing_map(logits, leading_dims, num_experts, routing_map_format);
  torch::stable::Tensor intermediate_output =
      torch::stable::new_empty(logits, sizes, torch::headeronly::ScalarType::Float);

  const std::vector<size_t> shape_2d = {static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  const std::vector<size_t> routing_map_shape_2d = {
      static_cast<size_t>(num_tokens),
      static_cast<size_t>(routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8
                              ? (num_experts + 7) / 8
                              : num_experts)};
  auto logits_dtype = GetTransformerEngineDType(logits.scalar_type());
  auto routing_map_dtype = GetTransformerEngineDType(routing_map.scalar_type());

  auto logits_cu = makeTransformerEngineTensor(logits.data_ptr(), shape_2d, logits_dtype);
  auto scores_cu = makeTransformerEngineTensor(scores.data_ptr(), shape_2d, DType::kFloat32);
  auto routing_map_cu =
      makeTransformerEngineTensor(routing_map.data_ptr(), routing_map_shape_2d, routing_map_dtype);
  auto intermediate_output_cu =
      makeTransformerEngineTensor(intermediate_output.data_ptr(), shape_2d, DType::kFloat32);

  nvte_fused_score_for_moe_aux_loss_forward_v2(
      logits_cu.data(), static_cast<int>(num_tokens), static_cast<int>(num_experts), topk,
      score_function_value, scores_cu.data(), routing_map_cu.data(),
      static_cast<NVTERoutingMapFormat>(routing_map_format), intermediate_output_cu.data(),
      getCurrentCUDAStream());

  return std::make_tuple(scores, routing_map, intermediate_output);
}

void fused_score_for_moe_aux_loss_bwd(torch::stable::Tensor intermediate_output,
                                      torch::stable::Tensor grad_scores,
                                      torch::stable::Tensor grad_logits, int topk,
                                      std::string score_function) {
  STD_TORCH_CHECK(grad_scores.dim() >= 1, "grad_scores must have at least 1 dim");
  STD_TORCH_CHECK(grad_scores.is_contiguous(), "grad_scores must be contiguous");
  STD_TORCH_CHECK(grad_logits.is_contiguous(), "grad_logits must be contiguous");
  auto sizes = shape_vec(grad_scores);
  int64_t num_experts = sizes.back();
  int64_t num_tokens =
      std::accumulate(sizes.begin(), sizes.end() - 1, int64_t{1}, std::multiplies<int64_t>());

  int score_function_value = get_score_function_value(score_function);

  const std::vector<size_t> shape_2d = {static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  auto grad_logits_dtype = GetTransformerEngineDType(grad_logits.scalar_type());

  auto intermediate_output_cu =
      makeTransformerEngineTensor(intermediate_output.data_ptr(), shape_2d, DType::kFloat32);
  auto grad_scores_cu =
      makeTransformerEngineTensor(grad_scores.data_ptr(), shape_2d, DType::kFloat32);
  auto grad_logits_cu =
      makeTransformerEngineTensor(grad_logits.data_ptr(), shape_2d, grad_logits_dtype);

  nvte_fused_score_for_moe_aux_loss_backward(
      intermediate_output_cu.data(), grad_scores_cu.data(), static_cast<int>(num_tokens),
      static_cast<int>(num_experts), topk, score_function_value, grad_logits_cu.data(),
      getCurrentCUDAStream());
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor> fused_moe_aux_loss_fwd(
    torch::stable::Tensor probs, torch::stable::Tensor tokens_per_expert, int total_num_tokens,
    int num_experts, int num_rows, int num_cols, int topk, float coeff) {
  STD_TORCH_CHECK(topk > 0, "topk must be greater than 0");
  STD_TORCH_CHECK(total_num_tokens > 0, "total_num_tokens must be greater than 0");
  STD_TORCH_CHECK(num_experts > 0, "num_experts must be greater than 0");

  // Create the output tensor
  torch::stable::Tensor aux_loss = torch::stable::new_empty(probs, std::vector<int64_t>{});
  torch::stable::Tensor Const_buf =
      torch::stable::new_empty(probs, {2}, torch::headeronly::ScalarType::Float);

  auto probs_cu = makeTransformerEngineTensor(probs);
  auto tokens_per_expert_cu = makeTransformerEngineTensor(tokens_per_expert);
  auto aux_loss_cu = makeTransformerEngineTensor(aux_loss);
  auto Const_buf_cu = makeTransformerEngineTensor(Const_buf);

  nvte_fused_moe_aux_loss_forward(probs_cu.data(), tokens_per_expert_cu.data(), total_num_tokens,
                                  num_experts, num_rows, num_cols, topk, coeff, aux_loss_cu.data(),
                                  Const_buf_cu.data(), getCurrentCUDAStream());

  return std::make_tuple(aux_loss, Const_buf);
}

torch::stable::Tensor fused_moe_aux_loss_bwd(torch::stable::Tensor Const_buf,
                                             torch::stable::Tensor tokens_per_expert, int num_rows,
                                             int num_cols, torch::stable::Tensor grad_aux_loss) {
  // Create the output tensor
  torch::stable::Tensor grad_probs = torch::stable::new_empty(grad_aux_loss, {num_rows, num_cols});

  auto Const_buf_cu = makeTransformerEngineTensor(Const_buf);
  auto tokens_per_expert_cu = makeTransformerEngineTensor(tokens_per_expert);
  auto grad_aux_loss_cu = makeTransformerEngineTensor(grad_aux_loss);
  auto grad_probs_cu = makeTransformerEngineTensor(grad_probs);

  // Meta data for the kernel
  nvte_fused_moe_aux_loss_backward(Const_buf_cu.data(), tokens_per_expert_cu.data(), num_rows,
                                   num_cols, grad_aux_loss_cu.data(), grad_probs_cu.data(),
                                   getCurrentCUDAStream());

  return grad_probs;
}

}  // namespace transformer_engine::pytorch
