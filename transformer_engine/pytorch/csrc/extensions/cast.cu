/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "c10/core/ScalarType.h"
#include "extensions.h"
#include "pybind.h"
#include "object.h"
#include "torch/types.h"

namespace transformer_engine::pytorch {

namespace detail {

bool IsFloat8QParamsType(PyObject *obj) {
  return Py_TYPE(obj) == Float8QParamsClass;
}

bool IsFloatingPointType(at::ScalarType type) {
  return type == at::kFloat ||
         type == at::kHalf  ||
         type == at::kBFloat16;
}

}  // namespace detail

py::handle cast(const at::Tensor& tensor,
                py::handle quantization_params,
                bool rowwise_usage,
                bool columnwise_usage,
                py::handle proxy
                ) {
  using namespace pybind11::literals;
  init_extension();
  auto input_tensor = tensor.contiguous();
  NVTE_CHECK(rowwise_usage || columnwise_usage,
             "Could not create a QuantizedTensor with no usage.");
  if (detail::IsFloat8QParamsType(quantization_params.ptr())) {
    auto py_scale = quantization_params.attr("scale");
    auto py_amax = quantization_params.attr("amax");
    DType type = quantization_params.attr("dtype").cast<DType>();
    const at::Tensor& scale = py_scale.cast<at::Tensor>();
    auto opts = input_tensor.options().dtype(torch::kFloat32);
    at::Tensor scale_inv = at::empty({1}, opts);
    at::Tensor data, data_transpose;
    if (columnwise_usage) {
      const auto dim = tensor.dim();
      NVTE_CHECK(dim >= 2, "Tensor needs to be at least 2D for columnwise usage");
      auto reshaped_input = input_tensor.view({-1, tensor.size(dim - 1)});
      auto data_opts = input_tensor.options().dtype(torch::kUInt8);
      data = at::empty_like(input_tensor, data_opts);
      data_transpose = at::empty({reshaped_input.size(1),
                                  reshaped_input.size(0)},
                                 data_opts);
      fused_cast_transpose(reshaped_input,
                           py_scale.cast<at::Tensor>(),
                           py_amax.cast<at::Tensor>(),
                           scale_inv,
                           data,
                           data_transpose,
                           type);
    } else {
      data = cast_to_fp8(input_tensor,
                         py_scale.cast<at::Tensor>(),
                         py_amax.cast<at::Tensor>(),
                         scale_inv,
                         type);
    }
    auto fake_tensor_type = tensor.scalar_type();
    if (!detail::IsFloatingPointType(fake_tensor_type)) {
      fake_tensor_type = at::kFloat;
    }
    py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorPythonClass));
    if (columnwise_usage) {
      auto ret = Float8TensorClass("data"_a=data,
                                   "data_transpose"_a=data_transpose,
                                   "fp8_scale_inv"_a=scale_inv,
                                   "fp8_dtype"_a=type,
                                   "dtype"_a=fake_tensor_type,
                                   "proxy"_a=proxy);
      return ret.release();
    } else {
      auto ret = Float8TensorClass("data"_a=data,
                                   "fp8_scale_inv"_a=scale_inv,
                                   "fp8_dtype"_a=type,
                                   "dtype"_a=fake_tensor_type,
                                   "proxy"_a=proxy);
      return ret.release();
    }
  }
  NVTE_ERROR("Invalid type of the quantization params");
}

}  // transformer_engine::pytorch

at::Tensor cast_to_fp8(const at::Tensor& input, const at::Tensor& scale, at::Tensor amax,
                       at::Tensor scale_inv, transformer_engine::DType otype,
                       const int scale_offset, const int amax_offset, const int scale_inv_offset) {
  using namespace transformer_engine;
  auto input_shape = input.sizes().vec();
  std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

  auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));

  if (input.numel() == 0) return output;

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  auto input_cu = makeTransformerEngineTensor(input);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), shape, otype, amax_dptr,
                                               scale_dptr, scale_inv_dptr);

  nvte_fp8_quantize(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

void cast_to_fp8_noalloc(const at::Tensor& input, const at::Tensor& scale, at::Tensor output,
                         at::Tensor amax, at::Tensor scale_inv, transformer_engine::DType otype,
                         const int scale_offset, const int amax_offset,
                         const int scale_inv_offset) {
  using namespace transformer_engine;
  size_t N = static_cast<size_t>(input.size(0));
  size_t H = static_cast<size_t>(input.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  auto input_cu = makeTransformerEngineTensor(input);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {N, H}, otype, amax_dptr,
                                               scale_dptr, scale_inv_dptr);

  nvte_fp8_quantize(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return;
}

at::Tensor cast_from_fp8(const at::Tensor& input, const at::Tensor& scale_inv,
                         transformer_engine::DType itype, transformer_engine::DType otype,
                         const int scale_inv_offset) {
  using namespace transformer_engine;
  auto input_shape = input.sizes().vec();
  std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

  auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), shape, itype, nullptr, nullptr,
                                              getDataPtr(scale_inv, scale_inv_offset));
  auto output_cu = makeTransformerEngineTensor(output);

  nvte_fp8_dequantize(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

std::vector<at::Tensor> fp8_cast_dbias(const at::Tensor& input, const at::Tensor& scale,
                                       at::Tensor amax, at::Tensor scale_inv,
                                       transformer_engine::DType otype,
                                       std::vector<int64_t> scaling_mode, const int scale_offset,
                                       const int amax_offset, const int scale_inv_offset) {
  using namespace transformer_engine;
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
  NVTEScalingMode nvte_scaling_mode = {scaling_mode[0], scaling_mode[1], scaling_mode[2]};

  auto input_cu = makeTransformerEngineTensor(input);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);
  auto output_cu =
      makeTransformerEngineTensor(output.data_ptr(), shape, otype, amax_dptr, scale_dptr,
                                  scale_inv_dptr, getTensorShape(scale_inv), nvte_scaling_mode);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_fp8_quantize_dbias(input_cu.data(), output_cu.data(), dbias_cu.data(), workspace.data(),
                          at::cuda::getCurrentCUDAStream());

  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  nvte_fp8_quantize_dbias(input_cu.data(), output_cu.data(), dbias_cu.data(), workspace.data(),
                          at::cuda::getCurrentCUDAStream());

  return {grad_bias, output};
}

std::vector<at::Tensor> fp8_cast_dbias_dgelu(at::Tensor grad_output, at::Tensor act_input,
                                             at::Tensor scale, at::Tensor amax,
                                             at::Tensor scale_inv, transformer_engine::DType otype,
                                             std::vector<int64_t> scaling_mode, int scale_offset,
                                             int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  NVTEScalingMode nvte_scaling_mode = {scaling_mode[0], scaling_mode[1], scaling_mode[2]};

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
  nvte_fp8_quantize_dbias_dgelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                dbias_cu.data(), workspace.data(),
                                at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_fp8_quantize_dbias_dgelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                dbias_cu.data(), workspace.data(),
                                at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact};
}

std::vector<at::Tensor> fp8_cast_dbias_dsilu(at::Tensor grad_output, at::Tensor act_input,
                                             at::Tensor scale, at::Tensor amax,
                                             at::Tensor scale_inv, transformer_engine::DType otype,
                                             std::vector<int64_t> scaling_mode, int scale_offset,
                                             int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  NVTEScalingMode nvte_scaling_mode = {scaling_mode[0], scaling_mode[1], scaling_mode[2]};

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
  nvte_fp8_quantize_dbias_dsilu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                dbias_cu.data(), workspace.data(),
                                at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_fp8_quantize_dbias_dsilu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                dbias_cu.data(), workspace.data(),
                                at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact};
}

std::vector<at::Tensor> fp8_cast_dbias_drelu(at::Tensor grad_output, at::Tensor act_input,
                                             at::Tensor scale, at::Tensor amax,
                                             at::Tensor scale_inv, transformer_engine::DType otype,
                                             std::vector<int64_t> scaling_mode, int scale_offset,
                                             int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  NVTEScalingMode nvte_scaling_mode = {scaling_mode[0], scaling_mode[1], scaling_mode[2]};

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
  nvte_fp8_quantize_dbias_drelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                dbias_cu.data(), workspace.data(),
                                at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_fp8_quantize_dbias_drelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                dbias_cu.data(), workspace.data(),
                                at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact};
}

std::vector<at::Tensor> fp8_cast_dbias_dqgelu(at::Tensor grad_output, at::Tensor act_input,
                                              at::Tensor scale, at::Tensor amax,
                                              at::Tensor scale_inv, transformer_engine::DType otype,
                                              std::vector<int64_t> scaling_mode, int scale_offset,
                                              int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  NVTEScalingMode nvte_scaling_mode = {scaling_mode[0], scaling_mode[1], scaling_mode[2]};

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
  nvte_fp8_quantize_dbias_dqgelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                 dbias_cu.data(), workspace.data(),
                                 at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_fp8_quantize_dbias_dqgelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                 dbias_cu.data(), workspace.data(),
                                 at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact};
}

std::vector<at::Tensor> fp8_cast_dbias_dsrelu(at::Tensor grad_output, at::Tensor act_input,
                                              at::Tensor scale, at::Tensor amax,
                                              at::Tensor scale_inv, transformer_engine::DType otype,
                                              std::vector<int64_t> scaling_mode, int scale_offset,
                                              int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  NVTEScalingMode nvte_scaling_mode = {scaling_mode[0], scaling_mode[1], scaling_mode[2]};

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
  nvte_fp8_quantize_dbias_dsrelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                 dbias_cu.data(), workspace.data(),
                                 at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_fp8_quantize_dbias_dsrelu(input_cu.data(), act_input_cu.data(), cast_output_cu.data(),
                                 dbias_cu.data(), workspace.data(),
                                 at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact};
}

std::vector<at::Tensor> fp8_cast_dbias_x2(const at::Tensor& input, const at::Tensor& scale,
                                          at::Tensor amax, at::Tensor scale_inv,
                                          transformer_engine::DType otype, const int scale_offset,
                                          const int amax_offset, const int scale_inv_offset) {
  using namespace transformer_engine;
  auto input_shape = input.sizes().vec();
  std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

  DType grad_output_type = GetTransformerEngineDType(input.scalar_type());
  auto output_rowwise = at::empty_like(input, at::CUDA(GetATenDType(otype)));
  auto output_columnwise = at::empty_like(input, at::CUDA(GetATenDType(otype)));
  auto grad_bias = allocateTorchTensor(input.size(-1), grad_output_type);

  if (input.numel() == 0) return {grad_bias, output_rowwise, output_columnwise};

  // Get pointers for FP8 scale, amax, scale-inverse
  void* rowwise_scale_dptr = getDataPtr(scale, scale_offset);
  void* rowwise_amax_dptr = getDataPtr(amax, amax_offset);
  void* rowwise_scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  auto columnwise_scale = scale.detach().clone();
  auto columnwise_scale_inv = scale_inv.detach().clone();
  auto columnwise_amax = amax.detach().clone();
  void* columnwise_scale_dptr = getDataPtr(columnwise_scale, scale_offset);
  void* columnwise_amax_dptr = getDataPtr(columnwise_amax, amax_offset);
  void* columnwise_scale_inv_dptr = getDataPtr(columnwise_scale_inv, scale_inv_offset);

  auto input_cu = makeTransformerEngineTensor(input);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);
  auto rowwise_output_cu =
      makeTransformerEngineTensor(output_rowwise.data_ptr(), shape, otype, rowwise_amax_dptr,
                                  rowwise_scale_dptr, rowwise_scale_inv_dptr);
  auto columnwise_output_cu =
      makeTransformerEngineTensor(output_columnwise.data_ptr(), shape, otype, columnwise_amax_dptr,
                                  columnwise_scale_dptr, columnwise_scale_inv_dptr);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_fp8_quantize_dbias_x2(input_cu.data(), rowwise_output_cu.data(), columnwise_output_cu.data(),
                             dbias_cu.data(), workspace.data(), at::cuda::getCurrentCUDAStream());

  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  nvte_fp8_quantize_dbias_x2(input_cu.data(), rowwise_output_cu.data(), columnwise_output_cu.data(),
                             dbias_cu.data(), workspace.data(), at::cuda::getCurrentCUDAStream());

  return {grad_bias, output_rowwise, output_columnwise};
}

std::vector<at::Tensor> fp8_cast_dbias_dgelu_x2(at::Tensor grad_output, at::Tensor act_input,
                                                at::Tensor scale, at::Tensor amax,
                                                at::Tensor scale_inv,
                                                transformer_engine::DType otype, int scale_offset,
                                                int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* rowwise_scale_dptr = getDataPtr(scale, scale_offset);
  void* rowwise_amax_dptr = getDataPtr(amax, amax_offset);
  void* rowwise_scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  auto columnwise_scale = scale.detach().clone();
  auto columnwise_scale_inv = scale_inv.detach().clone();
  auto columnwise_amax = amax.detach().clone();
  void* columnwise_scale_dptr = getDataPtr(columnwise_scale, scale_offset);
  void* columnwise_amax_dptr = getDataPtr(columnwise_amax, amax_offset);
  void* columnwise_scale_inv_dptr = getDataPtr(columnwise_scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact_rowwise = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto dact_columnwise =
      allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto rowwise_output_cu =
      makeTransformerEngineTensor(dact_rowwise.data_ptr(), {M, N}, otype, rowwise_amax_dptr,
                                  rowwise_scale_dptr, rowwise_scale_inv_dptr);
  auto columnwise_output_cu =
      makeTransformerEngineTensor(dact_columnwise.data_ptr(), {M, N}, otype, columnwise_amax_dptr,
                                  columnwise_scale_dptr, columnwise_scale_inv_dptr);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_fp8_quantize_dbias_dgelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                   columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                   at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_fp8_quantize_dbias_dgelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                   columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                   at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact_rowwise, dact_columnwise};
}

std::vector<at::Tensor> fp8_cast_dbias_dsilu_x2(at::Tensor grad_output, at::Tensor act_input,
                                                at::Tensor scale, at::Tensor amax,
                                                at::Tensor scale_inv,
                                                transformer_engine::DType otype, int scale_offset,
                                                int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* rowwise_scale_dptr = getDataPtr(scale, scale_offset);
  void* rowwise_amax_dptr = getDataPtr(amax, amax_offset);
  void* rowwise_scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  auto columnwise_scale = scale.detach().clone();
  auto columnwise_scale_inv = scale_inv.detach().clone();
  auto columnwise_amax = amax.detach().clone();
  void* columnwise_scale_dptr = getDataPtr(columnwise_scale, scale_offset);
  void* columnwise_amax_dptr = getDataPtr(columnwise_amax, amax_offset);
  void* columnwise_scale_inv_dptr = getDataPtr(columnwise_scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact_rowwise = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto dact_columnwise =
      allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto rowwise_output_cu =
      makeTransformerEngineTensor(dact_rowwise.data_ptr(), {M, N}, otype, rowwise_amax_dptr,
                                  rowwise_scale_dptr, rowwise_scale_inv_dptr);
  auto columnwise_output_cu =
      makeTransformerEngineTensor(dact_columnwise.data_ptr(), {M, N}, otype, columnwise_amax_dptr,
                                  columnwise_scale_dptr, columnwise_scale_inv_dptr);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_fp8_quantize_dbias_dsilu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                   columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                   at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_fp8_quantize_dbias_dsilu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                   columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                   at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact_rowwise, dact_columnwise};
}

std::vector<at::Tensor> fp8_cast_dbias_drelu_x2(at::Tensor grad_output, at::Tensor act_input,
                                                at::Tensor scale, at::Tensor amax,
                                                at::Tensor scale_inv,
                                                transformer_engine::DType otype, int scale_offset,
                                                int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* rowwise_scale_dptr = getDataPtr(scale, scale_offset);
  void* rowwise_amax_dptr = getDataPtr(amax, amax_offset);
  void* rowwise_scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  auto columnwise_scale = scale.detach().clone();
  auto columnwise_scale_inv = scale_inv.detach().clone();
  auto columnwise_amax = amax.detach().clone();
  void* columnwise_scale_dptr = getDataPtr(columnwise_scale, scale_offset);
  void* columnwise_amax_dptr = getDataPtr(columnwise_amax, amax_offset);
  void* columnwise_scale_inv_dptr = getDataPtr(columnwise_scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact_rowwise = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto dact_columnwise =
      allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto rowwise_output_cu =
      makeTransformerEngineTensor(dact_rowwise.data_ptr(), {M, N}, otype, rowwise_amax_dptr,
                                  rowwise_scale_dptr, rowwise_scale_inv_dptr);
  auto columnwise_output_cu =
      makeTransformerEngineTensor(dact_columnwise.data_ptr(), {M, N}, otype, columnwise_amax_dptr,
                                  columnwise_scale_dptr, columnwise_scale_inv_dptr);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_fp8_quantize_dbias_drelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                   columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                   at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_fp8_quantize_dbias_drelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                   columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                   at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact_rowwise, dact_columnwise};
}

std::vector<at::Tensor> fp8_cast_dbias_dqgelu_x2(at::Tensor grad_output, at::Tensor act_input,
                                                 at::Tensor scale, at::Tensor amax,
                                                 at::Tensor scale_inv,
                                                 transformer_engine::DType otype, int scale_offset,
                                                 int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* rowwise_scale_dptr = getDataPtr(scale, scale_offset);
  void* rowwise_amax_dptr = getDataPtr(amax, amax_offset);
  void* rowwise_scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  auto columnwise_scale = scale.detach().clone();
  auto columnwise_scale_inv = scale_inv.detach().clone();
  auto columnwise_amax = amax.detach().clone();
  void* columnwise_scale_dptr = getDataPtr(columnwise_scale, scale_offset);
  void* columnwise_amax_dptr = getDataPtr(columnwise_amax, amax_offset);
  void* columnwise_scale_inv_dptr = getDataPtr(columnwise_scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact_rowwise = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto dact_columnwise =
      allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto rowwise_output_cu =
      makeTransformerEngineTensor(dact_rowwise.data_ptr(), {M, N}, otype, rowwise_amax_dptr,
                                  rowwise_scale_dptr, rowwise_scale_inv_dptr);
  auto columnwise_output_cu =
      makeTransformerEngineTensor(dact_columnwise.data_ptr(), {M, N}, otype, columnwise_amax_dptr,
                                  columnwise_scale_dptr, columnwise_scale_inv_dptr);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_fp8_quantize_dbias_dqgelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                    columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                    at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_fp8_quantize_dbias_dqgelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                    columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                    at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact_rowwise, dact_columnwise};
}

std::vector<at::Tensor> fp8_cast_dbias_dsrelu_x2(at::Tensor grad_output, at::Tensor act_input,
                                                 at::Tensor scale, at::Tensor amax,
                                                 at::Tensor scale_inv,
                                                 transformer_engine::DType otype, int scale_offset,
                                                 int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* rowwise_scale_dptr = getDataPtr(scale, scale_offset);
  void* rowwise_amax_dptr = getDataPtr(amax, amax_offset);
  void* rowwise_scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);
  auto columnwise_scale = scale.detach().clone();
  auto columnwise_scale_inv = scale_inv.detach().clone();
  auto columnwise_amax = amax.detach().clone();
  void* columnwise_scale_dptr = getDataPtr(columnwise_scale, scale_offset);
  void* columnwise_amax_dptr = getDataPtr(columnwise_amax, amax_offset);
  void* columnwise_scale_inv_dptr = getDataPtr(columnwise_scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dact_rowwise = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto dact_columnwise =
      allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto act_input_cu = makeTransformerEngineTensor(act_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto rowwise_output_cu =
      makeTransformerEngineTensor(dact_rowwise.data_ptr(), {M, N}, otype, rowwise_amax_dptr,
                                  rowwise_scale_dptr, rowwise_scale_inv_dptr);
  auto columnwise_output_cu =
      makeTransformerEngineTensor(dact_columnwise.data_ptr(), {M, N}, otype, columnwise_amax_dptr,
                                  columnwise_scale_dptr, columnwise_scale_inv_dptr);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_fp8_quantize_dbias_dsrelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                    columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                    at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_fp8_quantize_dbias_dsrelu_x2(input_cu.data(), act_input_cu.data(), rowwise_output_cu.data(),
                                    columnwise_output_cu.data(), dbias_cu.data(), workspace.data(),
                                    at::cuda::getCurrentCUDAStream());

  return {grad_bias, dact_rowwise, dact_columnwise};
}
