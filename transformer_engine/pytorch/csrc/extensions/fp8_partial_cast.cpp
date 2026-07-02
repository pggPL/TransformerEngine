/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <torch/csrc/stable/accelerator.h>
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

void fp8_block_scaling_compute_partial_amax(const torch::stable::Tensor &tensor,
                                            torch::stable::Tensor amax, size_t h, size_t w,
                                            size_t start_offset, size_t block_len) {
  STD_TORCH_CHECK(block_len == 128, "Currently only block_len = 128 is supported");
  STD_TORCH_CHECK(amax.dim() == 2, "amax must be a 2D tensor");
  STD_TORCH_CHECK(amax.scalar_type() == torch::headeronly::ScalarType::Float,
                  "amax must be a float tensor");
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Float ||
                      tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                  "tensor must be a float or bfloat16 tensor");

  const TensorWrapper tensor_cu = makeTransformerEngineTensor(tensor);
  TensorWrapper amax_cu = makeTransformerEngineTensor(amax);

  nvte_fp8_block_scaling_compute_partial_amax(tensor_cu.data(), amax_cu.data(), h, w,
                                              amax.stride(0), amax.stride(1), start_offset,
                                              block_len, current_cuda_stream());
}

void fp8_block_scaling_partial_cast(const torch::stable::Tensor &inp, torch::stable::Tensor out,
                                    const torch::stable::Tensor &scale, size_t h, size_t w,
                                    size_t start_offset, size_t block_len,
                                    const transformer_engine::DType out_dtype) {
  STD_TORCH_CHECK(block_len == 128, "Currently only block_len = 128 is supported");
  STD_TORCH_CHECK(scale.dim() == 2, "scale must be a 2D tensor");
  STD_TORCH_CHECK(scale.scalar_type() == torch::headeronly::ScalarType::Float,
                  "scale must be a float tensor");
  STD_TORCH_CHECK(inp.scalar_type() == torch::headeronly::ScalarType::Float ||
                      inp.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                  "input must be a float or bfloat16 tensor");
  STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Byte,
                  "output must be a uint8 tensor");
  STD_TORCH_CHECK(out_dtype == transformer_engine::DType::kFloat8E4M3 ||
                      out_dtype == transformer_engine::DType::kFloat8E5M2,
                  "out_dtype must be kFloat8E4M3 or kFloat8E5M2");

  const TensorWrapper inp_cu = makeTransformerEngineTensor(inp);
  TensorWrapper out_cu = makeTransformerEngineTensor(out);
  const TensorWrapper scale_cu = makeTransformerEngineTensor(scale);

  nvte_fp8_block_scaling_partial_cast(inp_cu.data(), out_cu.data(), scale_cu.data(), h, w,
                                      scale.stride(0), scale.stride(1), start_offset, block_len,
                                      static_cast<NVTEDType>(out_dtype), current_cuda_stream());
}

void mxfp8_scaling_compute_partial_amax(const torch::stable::Tensor &input,
                                        torch::stable::Tensor amax_rowwise,
                                        torch::stable::Tensor amax_colwise, int rows, int cols,
                                        size_t start_offset) {
  // TODO(stable-abi): needs torch::stable::Tensor::is_contiguous().
  STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  STD_TORCH_CHECK(amax_rowwise.is_contiguous(), "amax_rowwise must be contiguous");
  STD_TORCH_CHECK(amax_colwise.is_contiguous(), "amax_colwise must be contiguous");

  const TensorWrapper input_cu = makeTransformerEngineTensor(input);
  TensorWrapper amax_rowwise_cu = makeTransformerEngineTensor(amax_rowwise);
  TensorWrapper amax_colwise_cu = makeTransformerEngineTensor(amax_colwise);

  nvte_mxfp8_scaling_compute_partial_amax(input_cu.data(), amax_rowwise_cu.data(),
                                          amax_colwise_cu.data(), rows, cols, start_offset,
                                          current_cuda_stream());
}

void mxfp8_scaling_partial_cast(const torch::stable::Tensor &input,
                                torch::stable::Tensor output_rowwise,
                                torch::stable::Tensor output_colwise,
                                const torch::stable::Tensor &scale_inv_rowwise,
                                const torch::stable::Tensor &scale_inv_colwise, int rows, int cols,
                                size_t start_offset) {
  // TODO(stable-abi): needs torch::stable::Tensor::is_contiguous().
  STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  STD_TORCH_CHECK(output_rowwise.is_contiguous(), "output_rowwise must be contiguous");
  STD_TORCH_CHECK(output_colwise.is_contiguous(), "output_colwise must be contiguous");
  STD_TORCH_CHECK(scale_inv_rowwise.is_contiguous(), "scale_inv_rowwise must be contiguous");
  STD_TORCH_CHECK(scale_inv_colwise.is_contiguous(), "scale_inv_colwise must be contiguous");

  const TensorWrapper input_cu = makeTransformerEngineTensor(input);
  TensorWrapper output_rowwise_cu = makeTransformerEngineTensor(output_rowwise);
  TensorWrapper output_colwise_cu = makeTransformerEngineTensor(output_colwise);
  const TensorWrapper scale_inv_rowwise_cu = makeTransformerEngineTensor(scale_inv_rowwise);
  const TensorWrapper scale_inv_colwise_cu = makeTransformerEngineTensor(scale_inv_colwise);

  nvte_mxfp8_scaling_partial_cast(input_cu.data(), output_rowwise_cu.data(),
                                  output_colwise_cu.data(), scale_inv_rowwise_cu.data(),
                                  scale_inv_colwise_cu.data(), rows, cols, start_offset,
                                  current_cuda_stream());
}

}  // namespace transformer_engine::pytorch
