/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <nanobind/nanobind.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/python/interop.h>
#include <torch/csrc/stable/tensor.h>

#include "../extensions.h"

namespace nb = nanobind;

namespace transformer_engine::pytorch {

torch::stable::Tensor scaled_softmax_forward(torch::stable::Tensor input, float scale_factor) {
  STD_TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  STD_TORCH_CHECK((input.scalar_type() == torch::headeronly::ScalarType::Half) ||
                      (input.scalar_type() == torch::headeronly::ScalarType::BFloat16),
                  "Only fp16 and bf16 are supported");

  const int batches = input.size(0);
  const int attn_heads = input.size(1);
  const int query_seq_len = input.size(2);
  const int key_seq_len = input.size(3);

  STD_TORCH_CHECK(key_seq_len <= 16384, "Key sequence length must be 16384 or less");
  STD_TORCH_CHECK(key_seq_len % 8 == 0, "Key sequence length must be divisible by 8");
  STD_TORCH_CHECK(query_seq_len > 1, "Query sequence length must be greater than 1");

  // Output (same dtype/device as input)
  auto softmax_results =
      torch::stable::new_empty(input, {batches, attn_heads, query_seq_len, key_seq_len});

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_softmax_forward(input_cu.data(), softmax_results_cu.data(), scale_factor,
                              getCurrentCUDAStream());

  return softmax_results;
}

torch::stable::Tensor scaled_softmax_backward(torch::stable::Tensor output_grad_,
                                              torch::stable::Tensor softmax_results_,
                                              float scale_factor) {
  auto output_grads = torch::stable::contiguous(output_grad_);
  auto softmax_results = torch::stable::contiguous(softmax_results_);

  STD_TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  STD_TORCH_CHECK(softmax_results.dim() == 4, "expected 4D tensor");

  STD_TORCH_CHECK((output_grads.scalar_type() == torch::headeronly::ScalarType::Half) ||
                      (output_grads.scalar_type() == torch::headeronly::ScalarType::BFloat16),
                  "Only fp16 and bf16 are supported");
  STD_TORCH_CHECK((softmax_results.scalar_type() == torch::headeronly::ScalarType::Half) ||
                      (softmax_results.scalar_type() == torch::headeronly::ScalarType::BFloat16),
                  "Only fp16 and bf16 are supported");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  // Produce gradients in place.
  nvte_scaled_softmax_backward(output_grads_cu.data(), softmax_results_cu.data(),
                               output_grads_cu.data(), scale_factor, getCurrentCUDAStream());

  return output_grads;
}

torch::stable::Tensor scaled_masked_softmax_forward(torch::stable::Tensor input,
                                                    torch::stable::Tensor mask, float scale_factor) {
  STD_TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  STD_TORCH_CHECK((input.scalar_type() == torch::headeronly::ScalarType::Half) ||
                      (input.scalar_type() == torch::headeronly::ScalarType::BFloat16),
                  "Only fp16 and bf16 are supported");
  STD_TORCH_CHECK(mask.dim() == 4, "expected 4D tensor");
  if (!input.is_contiguous()) input = torch::stable::contiguous(input);
  if (!mask.is_contiguous()) mask = torch::stable::contiguous(mask);

  const int batches = input.size(0);
  const int pad_batches = mask.size(0);
  const int attn_heads = input.size(1);
  const int query_seq_len = input.size(2);
  const int key_seq_len = input.size(3);

  STD_TORCH_CHECK(key_seq_len <= 16384, "Key sequence length must be 16384 or less");
  STD_TORCH_CHECK(key_seq_len % 8 == 0, "Key sequence length must be divisible by 8");
  STD_TORCH_CHECK(query_seq_len > 1, "Query sequence length must be greater than 1");
  STD_TORCH_CHECK(pad_batches == 1 || pad_batches == batches);
  STD_TORCH_CHECK(mask.size(1) == 1);
  STD_TORCH_CHECK(mask.size(2) == query_seq_len);
  STD_TORCH_CHECK(mask.size(3) == key_seq_len);

  auto softmax_results =
      torch::stable::new_empty(input, {batches, attn_heads, query_seq_len, key_seq_len});

  auto input_cu = makeTransformerEngineTensor(input);
  auto mask_cu = makeTransformerEngineTensor(mask);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_masked_softmax_forward(input_cu.data(), mask_cu.data(), softmax_results_cu.data(),
                                     scale_factor, getCurrentCUDAStream());

  return softmax_results;
}

torch::stable::Tensor scaled_masked_softmax_backward(torch::stable::Tensor output_grad_,
                                                     torch::stable::Tensor softmax_results_,
                                                     float scale_factor) {
  auto output_grads = torch::stable::contiguous(output_grad_);
  auto softmax_results = torch::stable::contiguous(softmax_results_);

  STD_TORCH_CHECK(output_grads.dim() == 4, "expected 3D tensor");
  STD_TORCH_CHECK(softmax_results.dim() == 4, "expected 3D tensor");

  STD_TORCH_CHECK((output_grads.scalar_type() == torch::headeronly::ScalarType::Half) ||
                      (output_grads.scalar_type() == torch::headeronly::ScalarType::BFloat16),
                  "Only fp16 and bf16 are supported");
  STD_TORCH_CHECK((softmax_results.scalar_type() == torch::headeronly::ScalarType::Half) ||
                      (softmax_results.scalar_type() == torch::headeronly::ScalarType::BFloat16),
                  "Only fp16 and bf16 are supported");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  // Produce gradients in place.
  nvte_scaled_softmax_backward(output_grads_cu.data(), softmax_results_cu.data(),
                               output_grads_cu.data(), scale_factor, getCurrentCUDAStream());

  return output_grads;
}

torch::stable::Tensor scaled_upper_triang_masked_softmax_forward(torch::stable::Tensor input,
                                                                 float scale_factor) {
  STD_TORCH_CHECK(input.dim() == 3, "expected 3D tensor");
  STD_TORCH_CHECK((input.scalar_type() == torch::headeronly::ScalarType::Half) ||
                      (input.scalar_type() == torch::headeronly::ScalarType::BFloat16),
                  "Only fp16 and bf16 are supported");

  const int attn_batches = input.size(0);
  const int seq_len = input.size(1);
  STD_TORCH_CHECK(seq_len <= 16384, "Sequence length must be 16384 or less");

  // Output
  auto softmax_results = torch::stable::new_empty(input, {attn_batches, seq_len, seq_len});

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_upper_triang_masked_softmax_forward(input_cu.data(), softmax_results_cu.data(),
                                                  scale_factor, getCurrentCUDAStream());

  return softmax_results;
}

torch::stable::Tensor scaled_upper_triang_masked_softmax_backward(
    torch::stable::Tensor output_grads_, torch::stable::Tensor softmax_results_,
    float scale_factor) {
  auto output_grads = torch::stable::contiguous(output_grads_);
  auto softmax_results = torch::stable::contiguous(softmax_results_);

  STD_TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  STD_TORCH_CHECK(softmax_results.dim() == 3, "expected 3D tensor");

  STD_TORCH_CHECK((output_grads.scalar_type() == torch::headeronly::ScalarType::Half) ||
                      (output_grads.scalar_type() == torch::headeronly::ScalarType::BFloat16),
                  "Only fp16 and bf16 are supported");
  STD_TORCH_CHECK((softmax_results.scalar_type() == torch::headeronly::ScalarType::Half) ||
                      (softmax_results.scalar_type() == torch::headeronly::ScalarType::BFloat16),
                  "Only fp16 and bf16 are supported");

  STD_TORCH_CHECK(output_grads.size(1) == output_grads.size(2));

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  // Produce gradients in place.
  nvte_scaled_upper_triang_masked_softmax_backward(output_grads_cu.data(), softmax_results_cu.data(),
                                                   output_grads_cu.data(), scale_factor,
                                                   getCurrentCUDAStream());

  return output_grads;
}

torch::stable::Tensor scaled_aligned_causal_masked_softmax_forward(torch::stable::Tensor input,
                                                                   float scale_factor) {
  STD_TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  STD_TORCH_CHECK((input.scalar_type() == torch::headeronly::ScalarType::Half) ||
                      (input.scalar_type() == torch::headeronly::ScalarType::BFloat16),
                  "Only fp16 and bf16 are supported");

  const int batches = input.size(0);
  const int attn_heads = input.size(1);
  const int query_seq_len = input.size(2);
  const int key_seq_len = input.size(3);

  STD_TORCH_CHECK(key_seq_len <= 16384, "Key sequence length must be 16384 or less");
  STD_TORCH_CHECK(key_seq_len % 8 == 0, "Key sequence length must be divisible by 8");
  STD_TORCH_CHECK(query_seq_len >= 1, "Query sequence length must be greater or equal to 1");

  // Output
  auto softmax_results =
      torch::stable::new_empty(input, {batches, attn_heads, query_seq_len, key_seq_len});

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_aligned_causal_masked_softmax_forward(input_cu.data(), softmax_results_cu.data(),
                                                    scale_factor, getCurrentCUDAStream());

  return softmax_results;
}

torch::stable::Tensor scaled_aligned_causal_masked_softmax_backward(
    torch::stable::Tensor output_grad_, torch::stable::Tensor softmax_results_, float scale_factor) {
  auto output_grads = torch::stable::contiguous(output_grad_);
  auto softmax_results = torch::stable::contiguous(softmax_results_);

  STD_TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  STD_TORCH_CHECK(softmax_results.dim() == 4, "expected 4D tensor");

  STD_TORCH_CHECK((output_grads.scalar_type() == torch::headeronly::ScalarType::Half) ||
                      (output_grads.scalar_type() == torch::headeronly::ScalarType::BFloat16),
                  "Only fp16 and bf16 are supported");
  STD_TORCH_CHECK((softmax_results.scalar_type() == torch::headeronly::ScalarType::Half) ||
                      (softmax_results.scalar_type() == torch::headeronly::ScalarType::BFloat16),
                  "Only fp16 and bf16 are supported");

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  // Produce gradients in place.
  nvte_scaled_aligned_causal_masked_softmax_backward(
      output_grads_cu.data(), softmax_results_cu.data(), output_grads_cu.data(), scale_factor,
      getCurrentCUDAStream());

  return output_grads;
}

}  // namespace transformer_engine::pytorch
