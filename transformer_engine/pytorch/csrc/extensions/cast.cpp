/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "extensions.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

py::object quantize(const at::Tensor& tensor, py::handle quantizer) {
  init_extension();
  auto my_quantizer = convert_quantizer(quantizer);
  auto input_tensor = tensor.contiguous();

  const TensorWrapper& te_input = makeTransformerEngineTensor(input_tensor);
  const auto& te_input_shape = te_input.shape();
  std::vector<size_t> input_shape(te_input_shape.data,
                                  te_input_shape.data + te_input_shape.ndim);
  auto fake_tensor_type = tensor.scalar_type();
  if (!detail::IsFloatingPointType(fake_tensor_type)) {
    fake_tensor_type = at::kFloat;
  }

  auto [te_output, out] = my_quantizer->create_tensor(input_shape,
                                                      GetTransformerEngineDType(fake_tensor_type));

  nvte_quantize(te_input.data(), te_output.data(), at::cuda::getCurrentCUDAStream());

  return out;
}

py::object dequantize(const py::handle& input, transformer_engine::DType otype) {
  using namespace transformer_engine::pytorch;

  const auto none = py::none();

  const auto& input_tensor = makeTransformerEngineTensor(input, none);

  NoneQuantizer q(none);

  const auto& shape = convertShape(input_tensor.shape());

  auto [out_tensor, out] = q.create_tensor(shape, otype);

  nvte_dequantize(input_tensor.data(), out_tensor.data(), at::cuda::getCurrentCUDAStream());

  return out;
}

}  // transformer_engine::pytorch

// TODO: remove once the MXFP8 quantizer and tensor are there
#if 0
std::vector<at::Tensor> cast_to_fp8_x2(const at::Tensor& input, const at::Tensor& scale_inv_rowwise,
                         at::Tensor scale_inv_colwise, transformer_engine::DType otype) {
  using namespace transformer_engine::pytorch;
  auto input_shape = input.sizes().vec();
  std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

  auto output_rowwise = at::empty_like(input, at::CUDA(GetATenDType(otype)));
  auto output_colwise = at::empty_like(input, at::CUDA(GetATenDType(otype)));

  if (input.numel() == 0) return {output_rowwise, output_colwise};

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_inv_rowwise_dptr = getDataPtr(scale_inv_rowwise, 0);
  void* scale_inv_colwise_dptr = getDataPtr(scale_inv_colwise, 0);

  auto input_cu = makeTransformerEngineTensor(input);
  auto output_rowwise_cu =
      makeTransformerEngineTensor(output_rowwise.data_ptr(), shape, otype, nullptr,
                                  nullptr, scale_inv_rowwise_dptr, getTensorShape(scale_inv_rowwise),
                                  {1, 32, 0});
  auto output_colwise_cu =
      makeTransformerEngineTensor(output_colwise.data_ptr(), shape, otype, nullptr,
                                  nullptr, scale_inv_colwise_dptr, getTensorShape(scale_inv_colwise),
                                  {32, 1, 0});

  nvte_quantize_x2(input_cu.data(), output_rowwise_cu.data(), output_colwise_cu.data(),
                       at::cuda::getCurrentCUDAStream());

  return {output_rowwise, output_colwise};
}
#endif

// TODO: REMOVE
at::Tensor cast_to_fp8(const at::Tensor& input, const at::Tensor& scale, at::Tensor amax,
                       at::Tensor scale_inv, transformer_engine::DType otype,
                       std::vector<int64_t> scaling_mode, const int scale_offset,
                       const int amax_offset, const int scale_inv_offset) {
  using namespace transformer_engine::pytorch;
  auto input_shape = input.sizes().vec();
  std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

  auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));

  if (input.numel() == 0) return output;

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  NVTEScalingMode nvte_scaling_mode = {scaling_mode[0], scaling_mode[1], scaling_mode[2]};

  auto input_cu = makeTransformerEngineTensor(input);
  auto output_cu =
      makeTransformerEngineTensor(output.data_ptr(), shape, otype, amax_dptr, scale_dptr,
                                  scale_inv_dptr, getTensorShape(scale_inv), nvte_scaling_mode);

  nvte_quantize(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

// TODO: REMOVE
void cast_to_fp8_noalloc(const at::Tensor& input, const at::Tensor& scale, at::Tensor output,
                         at::Tensor amax, at::Tensor scale_inv, transformer_engine::DType otype,
                         std::vector<int64_t> scaling_mode, const int scale_offset,
                         const int amax_offset, const int scale_inv_offset) {
  using namespace transformer_engine::pytorch;
  auto input_shape = input.sizes().vec();
  std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  NVTEScalingMode nvte_scaling_mode = {scaling_mode[0], scaling_mode[1], scaling_mode[2]};

  auto input_cu = makeTransformerEngineTensor(input);
  auto output_cu =
      makeTransformerEngineTensor(output.data_ptr(), shape, otype, amax_dptr, scale_dptr,
                                  scale_inv_dptr, getTensorShape(scale_inv), nvte_scaling_mode);

  nvte_quantize(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return;
}


std::vector<at::Tensor> fp8_cast_dbias(const at::Tensor& input, const at::Tensor& scale,
                                       at::Tensor amax, at::Tensor scale_inv,
                                       transformer_engine::DType otype,
                                       std::vector<int64_t> scaling_mode, const int scale_offset,
                                       const int amax_offset, const int scale_inv_offset) {
  using namespace transformer_engine::pytorch;
  auto input_shape = input.sizes().vec();
  std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

  DType grad_output_type = GetTransformerEngineDType(input.scalar_type());
  auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));
  auto grad_bias = allocateTorchTensor(input.size(-1), grad_output_type);

  if (input.numel() == 0) return {grad_bias, output};

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  NVTEScalingMode nvte_scaling_mode = {static_cast<int>(scaling_mode[0]),
                                       static_cast<int>(scaling_mode[1]),
                                       static_cast<int>(scaling_mode[2])};

  auto input_cu = makeTransformerEngineTensor(input);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);
  auto output_cu =
      makeTransformerEngineTensor(output.data_ptr(), shape, otype, amax_dptr, scale_dptr,
                                  scale_inv_dptr, getTensorShape(scale_inv), nvte_scaling_mode);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_quantize_dbias(input_cu.data(), output_cu.data(), dbias_cu.data(), workspace.data(),
                      at::cuda::getCurrentCUDAStream());

  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  nvte_quantize_dbias(input_cu.data(), output_cu.data(), dbias_cu.data(), workspace.data(),
                      at::cuda::getCurrentCUDAStream());

  return {grad_bias, output};
}

std::vector<at::Tensor> fp8_cast_dbias_dgelu(at::Tensor grad_output, at::Tensor act_input,
                                             at::Tensor scale, at::Tensor amax,
                                             at::Tensor scale_inv, transformer_engine::DType otype,
                                             std::vector<int64_t> scaling_mode, int scale_offset,
                                             int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  NVTEScalingMode nvte_scaling_mode = {static_cast<int>(scaling_mode[0]),
                                       static_cast<int>(scaling_mode[1]),
                                       static_cast<int>(scaling_mode[2])};

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto cast_output_cu =
      makeTransformerEngineTensor(dact.data_ptr(), {M, N}, otype, amax_dptr, scale_dptr,
                                  scale_inv_dptr, getTensorShape(scale_inv), nvte_scaling_mode);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_quantize_dbias_dgelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                            dbias_cu.data(), workspace.data(),
                            at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_quantize_dbias_dgelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                            dbias_cu.data(), workspace.data(),
                            at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact};
}

std::vector<at::Tensor> fp8_cast_dbias_dsilu(at::Tensor grad_output, at::Tensor act_input,
                                             at::Tensor scale, at::Tensor amax,
                                             at::Tensor scale_inv, transformer_engine::DType otype,
                                             std::vector<int64_t> scaling_mode, int scale_offset,
                                             int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  NVTEScalingMode nvte_scaling_mode = {static_cast<int>(scaling_mode[0]),
                                       static_cast<int>(scaling_mode[1]),
                                       static_cast<int>(scaling_mode[2])};

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto cast_output_cu =
      makeTransformerEngineTensor(dact.data_ptr(), {M, N}, otype, amax_dptr, scale_dptr,
                                  scale_inv_dptr, getTensorShape(scale_inv), nvte_scaling_mode);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_quantize_dbias_dsilu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                dbias_cu.data(), workspace.data(),
                                at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_quantize_dbias_dsilu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                dbias_cu.data(), workspace.data(),
                                at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact};
}

std::vector<at::Tensor> fp8_cast_dbias_drelu(at::Tensor grad_output, at::Tensor act_input,
                                             at::Tensor scale, at::Tensor amax,
                                             at::Tensor scale_inv, transformer_engine::DType otype,
                                             std::vector<int64_t> scaling_mode, int scale_offset,
                                             int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  NVTEScalingMode nvte_scaling_mode = {static_cast<int>(scaling_mode[0]),
                                       static_cast<int>(scaling_mode[1]),
                                       static_cast<int>(scaling_mode[2])};

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto cast_output_cu =
      makeTransformerEngineTensor(dact.data_ptr(), {M, N}, otype, amax_dptr, scale_dptr,
                                  scale_inv_dptr, getTensorShape(scale_inv), nvte_scaling_mode);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_quantize_dbias_drelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                dbias_cu.data(), workspace.data(),
                                at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_quantize_dbias_drelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                dbias_cu.data(), workspace.data(),
                                at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact};
}

std::vector<at::Tensor> fp8_cast_dbias_dqgelu(at::Tensor grad_output, at::Tensor act_input,
                                              at::Tensor scale, at::Tensor amax,
                                              at::Tensor scale_inv, transformer_engine::DType otype,
                                              std::vector<int64_t> scaling_mode, int scale_offset,
                                              int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  NVTEScalingMode nvte_scaling_mode = {static_cast<int>(scaling_mode[0]),
                                       static_cast<int>(scaling_mode[1]),
                                       static_cast<int>(scaling_mode[2])};

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto cast_output_cu =
      makeTransformerEngineTensor(dact.data_ptr(), {M, N}, otype, amax_dptr, scale_dptr,
                                  scale_inv_dptr, getTensorShape(scale_inv), nvte_scaling_mode);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_quantize_dbias_dqgelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                 dbias_cu.data(), workspace.data(),
                                 at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_quantize_dbias_dqgelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                 dbias_cu.data(), workspace.data(),
                                 at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact};
}

std::vector<at::Tensor> fp8_cast_dbias_dsrelu(at::Tensor grad_output, at::Tensor act_input,
                                              at::Tensor scale, at::Tensor amax,
                                              at::Tensor scale_inv, transformer_engine::DType otype,
                                              std::vector<int64_t> scaling_mode, int scale_offset,
                                              int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  NVTEScalingMode nvte_scaling_mode = {static_cast<int>(scaling_mode[0]),
                                       static_cast<int>(scaling_mode[1]),
                                       static_cast<int>(scaling_mode[2])};

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto cast_output_cu =
      makeTransformerEngineTensor(dact.data_ptr(), {M, N}, otype, amax_dptr, scale_dptr,
                                  scale_inv_dptr, getTensorShape(scale_inv), nvte_scaling_mode);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_quantize_dbias_dsrelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                 dbias_cu.data(), workspace.data(),
                                 at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_quantize_dbias_dsrelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                 dbias_cu.data(), workspace.data(),
                                 at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact};
}

// TODO: remove
#if 0
std::vector<at::Tensor> fp8_cast_dbias_x2(const at::Tensor& input, at::Tensor scale_inv_rowwise,
                                          at::Tensor scale_inv_colwise, transformer_engine::DType otype) {
  using namespace transformer_engine::pytorch;
  auto input_shape = input.sizes().vec();
  std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

  DType grad_output_type = GetTransformerEngineDType(input.scalar_type());
  auto output_rowwise = at::empty_like(input, at::CUDA(GetATenDType(otype)));
  auto output_columnwise = at::empty_like(input, at::CUDA(GetATenDType(otype)));
  auto grad_bias = allocateTorchTensor(input.size(-1), grad_output_type);

  if (input.numel() == 0) return {grad_bias, output_rowwise, output_columnwise};

  // Get pointers for FP8 scale, amax, scale-inverse
  void* rowwise_scale_inv_dptr = getDataPtr(scale_inv_rowwise, 0);
  void* colwise_scale_inv_dptr = getDataPtr(scale_inv_colwise, 0);

  auto input_cu = makeTransformerEngineTensor(input);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);
  auto rowwise_output_cu =
      makeTransformerEngineTensor(output_rowwise.data_ptr(), shape, otype, nullptr,
                                  nullptr, rowwise_scale_inv_dptr, getTensorShape(scale_inv_rowwise),
                                  {1, 32, 0});
  auto columnwise_output_cu =
      makeTransformerEngineTensor(output_columnwise.data_ptr(), shape, otype, nullptr,
                                  nullptr, colwise_scale_inv_dptr, getTensorShape(scale_inv_colwise),
                                  {32, 1, 0});

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_quantize_dbias_x2(input_cu.data(), rowwise_output_cu.data(), columnwise_output_cu.data(),
                             dbias_cu.data(), workspace.data(), at::cuda::getCurrentCUDAStream());

  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  nvte_quantize_dbias_x2(input_cu.data(), rowwise_output_cu.data(), columnwise_output_cu.data(),
                             dbias_cu.data(), workspace.data(), at::cuda::getCurrentCUDAStream());

  return {grad_bias, output_rowwise, output_columnwise};
}

std::vector<at::Tensor> fp8_cast_dbias_dgelu_x2(at::Tensor grad_output, at::Tensor act_input,
                                                at::Tensor scale_inv_rowwise, at::Tensor scale_inv_colwise,
                                                transformer_engine::DType otype) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* rowwise_scale_inv_dptr = getDataPtr(scale_inv_rowwise, 0);
  void* colwise_scale_inv_dptr = getDataPtr(scale_inv_colwise, 0);

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact_rowwise = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto dact_columnwise =
      allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto rowwise_output_cu =
      makeTransformerEngineTensor(dact_rowwise.data_ptr(), {M, N}, otype, nullptr,
                                  nullptr, rowwise_scale_inv_dptr, getTensorShape(scale_inv_rowwise),
                                  {1, 32, 0});
  auto columnwise_output_cu =
      makeTransformerEngineTensor(dact_columnwise.data_ptr(), {M, N}, otype, nullptr,
                                  nullptr, colwise_scale_inv_dptr, getTensorShape(scale_inv_colwise),
                                  {32, 1, 0});
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_quantize_dbias_dgelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                   columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                   at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_quantize_dbias_dgelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                   columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                   at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact_rowwise, dact_columnwise};
}

std::vector<at::Tensor> fp8_cast_dbias_dsilu_x2(at::Tensor grad_output, at::Tensor act_input,
                                                at::Tensor scale_inv_rowwise, at::Tensor scale_inv_colwise,
                                                transformer_engine::DType otype) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* rowwise_scale_inv_dptr = getDataPtr(scale_inv_rowwise, 0);
  void* colwise_scale_inv_dptr = getDataPtr(scale_inv_colwise, 0);

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact_rowwise = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto dact_columnwise =
      allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto rowwise_output_cu =
      makeTransformerEngineTensor(dact_rowwise.data_ptr(), {M, N}, otype, nullptr,
                                  nullptr, rowwise_scale_inv_dptr, getTensorShape(scale_inv_rowwise),
                                  {1, 32, 0});
  auto columnwise_output_cu =
      makeTransformerEngineTensor(dact_columnwise.data_ptr(), {M, N}, otype, nullptr,
                                  nullptr, colwise_scale_inv_dptr, getTensorShape(scale_inv_colwise),
                                  {32, 1, 0});
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_quantize_dbias_dsilu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                   columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                   at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_quantize_dbias_dsilu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                   columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                   at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact_rowwise, dact_columnwise};
}

std::vector<at::Tensor> fp8_cast_dbias_drelu_x2(at::Tensor grad_output, at::Tensor act_input,
                                                at::Tensor scale_inv_rowwise, at::Tensor scale_inv_colwise,
                                                transformer_engine::DType otype) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* rowwise_scale_inv_dptr = getDataPtr(scale_inv_rowwise, 0);
  void* colwise_scale_inv_dptr = getDataPtr(scale_inv_colwise, 0);

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact_rowwise = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto dact_columnwise =
      allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto rowwise_output_cu =
      makeTransformerEngineTensor(dact_rowwise.data_ptr(), {M, N}, otype, nullptr,
                                  nullptr, rowwise_scale_inv_dptr, getTensorShape(scale_inv_rowwise),
                                  {1, 32, 0});
  auto columnwise_output_cu =
      makeTransformerEngineTensor(dact_columnwise.data_ptr(), {M, N}, otype, nullptr,
                                  nullptr, colwise_scale_inv_dptr, getTensorShape(scale_inv_colwise),
                                  {32, 1, 0});
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_quantize_dbias_drelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                   columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                   at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_quantize_dbias_drelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                   columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                   at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact_rowwise, dact_columnwise};
}

std::vector<at::Tensor> fp8_cast_dbias_dqgelu_x2(at::Tensor grad_output, at::Tensor act_input,
                                                at::Tensor scale_inv_rowwise, at::Tensor scale_inv_colwise,
                                                transformer_engine::DType otype) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* rowwise_scale_inv_dptr = getDataPtr(scale_inv_rowwise, 0);
  void* colwise_scale_inv_dptr = getDataPtr(scale_inv_colwise, 0);

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact_rowwise = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto dact_columnwise =
      allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto rowwise_output_cu =
      makeTransformerEngineTensor(dact_rowwise.data_ptr(), {M, N}, otype, nullptr,
                                  nullptr, rowwise_scale_inv_dptr, getTensorShape(scale_inv_rowwise),
                                  {1, 32, 0});
  auto columnwise_output_cu =
      makeTransformerEngineTensor(dact_columnwise.data_ptr(), {M, N}, otype, nullptr,
                                  nullptr, colwise_scale_inv_dptr, getTensorShape(scale_inv_colwise),
                                  {32, 1, 0});
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_quantize_dbias_dqgelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                    columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                    at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_quantize_dbias_dqgelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                    columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                    at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact_rowwise, dact_columnwise};
}

std::vector<at::Tensor> fp8_cast_dbias_dsrelu_x2(at::Tensor grad_output, at::Tensor act_input,
                                                at::Tensor scale_inv_rowwise, at::Tensor scale_inv_colwise,
                                                transformer_engine::DType otype) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* rowwise_scale_inv_dptr = getDataPtr(scale_inv_rowwise, 0);
  void* colwise_scale_inv_dptr = getDataPtr(scale_inv_colwise, 0);

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact_rowwise = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto dact_columnwise =
      allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto rowwise_output_cu =
      makeTransformerEngineTensor(dact_rowwise.data_ptr(), {M, N}, otype, nullptr,
                                  nullptr, rowwise_scale_inv_dptr, getTensorShape(scale_inv_rowwise),
                                  {1, 32, 0});
  auto columnwise_output_cu =
      makeTransformerEngineTensor(dact_columnwise.data_ptr(), {M, N}, otype, nullptr,
                                  nullptr, colwise_scale_inv_dptr, getTensorShape(scale_inv_colwise),
                                  {32, 1, 0});
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_quantize_dbias_dsrelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                    columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                    at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_quantize_dbias_dsrelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                    columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                    at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact_rowwise, dact_columnwise};
}
#endif
