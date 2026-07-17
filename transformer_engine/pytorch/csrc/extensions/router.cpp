/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <numeric>

#include "../extensions.h"
#include "common.h"

namespace transformer_engine::pytorch {

static std::map<std::string, int> score_function_map = {
    {"sigmoid", 0}, {"softmax", 1}, {"sqrtsoftplus", 2}};

// Allocate a routing_map output tensor:
//   BYTEMAP   -> bool [*leading_dims, num_experts]
//   BITMAP_U8 -> uint8[*leading_dims, ceil(num_experts/8)], LSB-first
static Tensor allocate_routing_map(IntArrayRef leading_dims, int64_t num_experts,
                                       int routing_map_format) {
  std::vector<int64_t> shape(leading_dims.begin(), leading_dims.end());
  if (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8) {
    shape.push_back((num_experts + 7) / 8);
    return empty(shape, dtype(kByte).device(kCUDA));
  }
  shape.push_back(num_experts);
  return empty(shape, dtype(kBool).device(kCUDA));
}

std::tuple<Tensor, Tensor, Tensor> fused_topk_with_score_function_fwd(
    Tensor logits, int topk, bool use_pre_softmax, std::optional<int> num_groups,
    std::optional<int> group_topk, std::optional<float> scaling_factor, std::string score_function,
    std::optional<Tensor> expert_bias, int routing_map_format) {
  TORCH_CHECK(logits.dim() >= 1, "logits must have at least 1 dim");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  auto sizes = logits.sizes();
  int64_t num_experts = sizes.back();
  int64_t num_tokens =
      std::accumulate(sizes.begin(), sizes.end() - 1, int64_t{1}, std::multiplies<int64_t>());
  TORCH_CHECK(num_tokens > 0 && num_experts > 0,
              "num_tokens and num_experts must be greater than 0");
  // Expert bias only happens at the sigmoid case
  if (expert_bias.has_value()) {
    TORCH_CHECK(score_function == "sigmoid" || score_function == "sqrtsoftplus",
                "score_function must be sigmoid or sqrtsoftplus when expert_bias is not None");
    TORCH_CHECK(expert_bias.value().scalar_type() == kFloat,
                "expert_bias must be a float32 tensor");
  }
  // Check if the score function is valid
  TORCH_CHECK(score_function == "softmax" || score_function == "sigmoid" ||
                  score_function == "sqrtsoftplus",
              "score_function must be softmax, sigmoid or sqrtsoftplus for router fusion");
  if (score_function == "sigmoid" || score_function == "sqrtsoftplus") {
    use_pre_softmax = false;  // Pre-softmax only happens at the softmax case
  }

  // Reformat the input to make it compatible with the kernel
  int group_topk_value = group_topk.has_value() ? group_topk.value() : -1;
  int num_groups_value = num_groups.has_value() ? num_groups.value() : -1;
  float scaling_factor_value = scaling_factor.has_value() ? scaling_factor.value() : 1.0f;

  Tensor probs = empty(sizes, dtype(logits.scalar_type()).device(kCUDA));
  Tensor routing_map =
      allocate_routing_map(sizes.slice(0, sizes.size() - 1), num_experts, routing_map_format);
  Tensor intermediate_output = empty(sizes, dtype(kFloat).device(kCUDA));

  // 2D shape for the kernel (common-layer NVTE_CHECKs require {num_tokens, trailing_dim}).
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
  auto probs_cu = makeTransformerEngineTensor(probs.data_ptr(), shape_2d, logits_dtype);
  auto routing_map_cu =
      makeTransformerEngineTensor(routing_map.data_ptr(), routing_map_shape_2d, routing_map_dtype);
  auto intermediate_output_cu =
      makeTransformerEngineTensor(intermediate_output.data_ptr(), shape_2d, DType::kFloat32);
  auto expert_bias_cu = TensorWrapper();  // empty expert_bias_cu tensor
  if (expert_bias.has_value()) {
    expert_bias_cu = makeTransformerEngineTensor(expert_bias.value());
  }

  nvte_fused_topk_with_score_function_forward_v2(
      logits_cu.data(), static_cast<int>(num_tokens), static_cast<int>(num_experts), topk,
      use_pre_softmax, num_groups_value, group_topk_value, scaling_factor_value,
      score_function_map[score_function], expert_bias_cu.data(), probs_cu.data(),
      routing_map_cu.data(), static_cast<NVTERoutingMapFormat>(routing_map_format),
      intermediate_output_cu.data(), getCurrentCUDAStream());

  return std::make_tuple(probs, routing_map, intermediate_output);
}

void fused_topk_with_score_function_bwd(Tensor routing_map, Tensor intermediate_output,
                                        Tensor grad_probs, Tensor grad_logits, int topk,
                                        bool use_pre_softmax, std::optional<float> scaling_factor,
                                        std::string score_function, int routing_map_format) {
  TORCH_CHECK(grad_probs.dim() >= 1, "grad_probs must have at least 1 dim");
  TORCH_CHECK(grad_probs.is_contiguous(), "grad_probs must be contiguous");
  TORCH_CHECK(grad_logits.is_contiguous(), "grad_logits must be contiguous");
  auto sizes = grad_probs.sizes();
  int64_t num_experts = sizes.back();
  int64_t num_tokens =
      std::accumulate(sizes.begin(), sizes.end() - 1, int64_t{1}, std::multiplies<int64_t>());

  auto scaling_factor_value = scaling_factor.has_value() ? scaling_factor.value() : 1.0f;
  auto score_function_value = score_function_map[score_function];

  const std::vector<size_t> shape_2d = {static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  const std::vector<size_t> routing_map_shape_2d = {
      static_cast<size_t>(num_tokens),
      static_cast<size_t>(routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8
                              ? (num_experts + 7) / 8
                              : num_experts)};
  auto grad_dtype = GetTransformerEngineDType(grad_probs.scalar_type());
  auto routing_map_dtype = GetTransformerEngineDType(routing_map.scalar_type());

  auto routing_map_cu =
      makeTransformerEngineTensor(routing_map.data_ptr(), routing_map_shape_2d, routing_map_dtype);
  auto intermediate_output_cu =
      makeTransformerEngineTensor(intermediate_output.data_ptr(), shape_2d, DType::kFloat32);
  auto grad_probs_cu = makeTransformerEngineTensor(grad_probs.data_ptr(), shape_2d, grad_dtype);
  auto grad_logits_cu = makeTransformerEngineTensor(grad_logits.data_ptr(), shape_2d, grad_dtype);

  nvte_fused_topk_with_score_function_backward_v2(
      routing_map_cu.data(), static_cast<NVTERoutingMapFormat>(routing_map_format),
      intermediate_output_cu.data(), grad_probs_cu.data(), static_cast<int>(num_tokens),
      static_cast<int>(num_experts), topk, use_pre_softmax, scaling_factor_value,
      score_function_value, grad_logits_cu.data(), getCurrentCUDAStream());
}

std::tuple<Tensor, Tensor, Tensor> fused_score_for_moe_aux_loss_fwd(
    Tensor logits, int topk, std::string score_function, int routing_map_format) {
  TORCH_CHECK(logits.dim() >= 1, "logits must have at least 1 dim");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  auto sizes = logits.sizes();
  int64_t num_experts = sizes.back();
  int64_t num_tokens =
      std::accumulate(sizes.begin(), sizes.end() - 1, int64_t{1}, std::multiplies<int64_t>());
  TORCH_CHECK(num_tokens > 0 && num_experts > 0,
              "num_tokens and num_experts must be greater than 0");
  TORCH_CHECK(topk > 0, "topk must be greater than 0");
  TORCH_CHECK(score_function == "softmax" || score_function == "sigmoid" ||
                  score_function == "sqrtsoftplus",
              "score_function must be softmax, sigmoid or sqrtsoftplus for router fusion");
  int score_function_value = score_function_map[score_function];

  Tensor scores = empty(sizes, dtype(kFloat).device(kCUDA));
  Tensor routing_map =
      allocate_routing_map(sizes.slice(0, sizes.size() - 1), num_experts, routing_map_format);
  Tensor intermediate_output = empty(sizes, dtype(kFloat).device(kCUDA));

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

void fused_score_for_moe_aux_loss_bwd(Tensor intermediate_output, Tensor grad_scores,
                                      Tensor grad_logits, int topk,
                                      std::string score_function) {
  TORCH_CHECK(grad_scores.dim() >= 1, "grad_scores must have at least 1 dim");
  TORCH_CHECK(grad_scores.is_contiguous(), "grad_scores must be contiguous");
  TORCH_CHECK(grad_logits.is_contiguous(), "grad_logits must be contiguous");
  auto sizes = grad_scores.sizes();
  int64_t num_experts = sizes.back();
  int64_t num_tokens =
      std::accumulate(sizes.begin(), sizes.end() - 1, int64_t{1}, std::multiplies<int64_t>());

  int score_function_value = score_function_map[score_function];

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

std::tuple<Tensor, Tensor> fused_moe_aux_loss_fwd(Tensor probs,
                                                          Tensor tokens_per_expert,
                                                          int total_num_tokens, int num_experts,
                                                          int num_rows, int num_cols, int topk,
                                                          float coeff) {
  TORCH_CHECK(topk > 0, "topk must be greater than 0");
  TORCH_CHECK(total_num_tokens > 0, "total_num_tokens must be greater than 0");
  TORCH_CHECK(num_experts > 0, "num_experts must be greater than 0");

  // Create the output tensor
  Tensor aux_loss = empty({}, dtype(probs.scalar_type()).device(kCUDA));
  Tensor Const_buf = empty({2}, dtype(kFloat).device(kCUDA));

  auto probs_cu = makeTransformerEngineTensor(probs);
  auto tokens_per_expert_cu = makeTransformerEngineTensor(tokens_per_expert);
  auto aux_loss_cu = makeTransformerEngineTensor(aux_loss);
  auto Const_buf_cu = makeTransformerEngineTensor(Const_buf);

  nvte_fused_moe_aux_loss_forward(probs_cu.data(), tokens_per_expert_cu.data(), total_num_tokens,
                                  num_experts, num_rows, num_cols, topk, coeff, aux_loss_cu.data(),
                                  Const_buf_cu.data(), getCurrentCUDAStream());

  return std::make_tuple(aux_loss, Const_buf);
}

Tensor fused_moe_aux_loss_bwd(Tensor Const_buf, Tensor tokens_per_expert, int num_rows,
                                  int num_cols, Tensor grad_aux_loss) {
  // Create the output tensor
  Tensor grad_probs =
      empty({num_rows, num_cols}, dtype(grad_aux_loss.scalar_type()).device(kCUDA));

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
