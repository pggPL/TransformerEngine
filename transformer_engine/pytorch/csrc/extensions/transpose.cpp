/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <optional>
#include "ATen/core/TensorBody.h"
#include "extensions.h"

void fused_cast_transpose_noop(at::Tensor input, at::Tensor noop, at::Tensor scale, at::Tensor amax,
                               at::Tensor scale_inv, at::Tensor input_cast,
                               at::Tensor input_transpose, transformer_engine::DType otype,
                               int scale_offset, int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  auto input_cu = makeTransformerEngineTensor(input);
  auto noop_cu = makeTransformerEngineTensor(noop);
  auto output_cu = makeTransformerEngineTensor(input_cast.data_ptr(),
                                               input_transpose.data_ptr(),
                                               {M, N},
                                               {N, M},
                                               otype,
                                               amax_dptr,
                                               scale_dptr,
                                               scale_inv_dptr,
                                               scale_inv_dptr);

  // Launch kernel
  nvte_cast_transpose_with_noop(input_cu.data(), noop_cu.data(), output_cu.data(),
                                at::cuda::getCurrentCUDAStream());
}

// TODO: remove
std::vector<at::Tensor> fused_cast_transpose_bgrad(at::Tensor grad_output, at::Tensor scale,
                                                   at::Tensor amax, at::Tensor scale_inv,
                                                   transformer_engine::DType otype,
                                                   int scale_offset, int amax_offset,
                                                   int scale_inv_offset) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Allocate output tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto grad_output_cast =
      allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto grad_output_transpose =
      allocateTorchTensor(grad_output.size(1), grad_output.size(0), DType::kByte);

  // Return immediately if tensors are empty
  if (M == 0 || N == 0) {
    return {grad_bias.zero_(), grad_output_cast, grad_output_transpose};
  }

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  auto input_cu = makeTransformerEngineTensor(grad_output);

  auto output_cu = makeTransformerEngineTensor(grad_output_cast.data_ptr(),
                                               grad_output_transpose.data_ptr(),
                                               {M, N},
                                               {N, M},
                                               otype,
                                               amax_dptr,
                                               scale_dptr,
                                               scale_inv_dptr,
                                               scale_inv_dptr);

  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_cast_transpose_dbias(input_cu.data(), output_cu.data(),
                            dbias_cu.data(), workspace.data(), at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_cast_transpose_dbias(input_cu.data(), output_cu.data(),
                            dbias_cu.data(), workspace.data(), at::cuda::getCurrentCUDAStream());

  return {grad_bias, grad_output_cast, grad_output_transpose};
}

std::vector<at::Tensor> fused_fp8_transpose_bgrad(at::Tensor grad_output, at::Tensor scale,
                                                  at::Tensor amax, at::Tensor scale_inv,
                                                  transformer_engine::DType otype,
                                                  transformer_engine::DType grad_bias_type,
                                                  int scale_offset, int amax_offset,
                                                  int scale_inv_offset) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_bias_type);
  auto grad_output_transpose =
      allocateTorchTensor(grad_output.size(1), grad_output.size(0), DType::kByte);
  auto input_cu = makeTransformerEngineTensor(grad_output.data_ptr(), {M, N}, otype, amax_dptr,
                                              scale_dptr, scale_inv_dptr);
  auto transposed_output_cu = makeTransformerEngineTensor(
      grad_output_transpose.data_ptr(), {N, M}, otype, amax_dptr, scale_dptr, scale_inv_dptr);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_fp8_transpose_dbias(input_cu.data(), transposed_output_cu.data(), dbias_cu.data(),
                           workspace.data(), at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_fp8_transpose_dbias(input_cu.data(), transposed_output_cu.data(), dbias_cu.data(),
                           workspace.data(), at::cuda::getCurrentCUDAStream());

  return {grad_bias, grad_output_transpose};
}

std::vector<at::Tensor> fused_cast_transpose_bgrad_dgelu(at::Tensor grad_output,
                                                         at::Tensor gelu_input, at::Tensor scale,
                                                         at::Tensor amax, at::Tensor scale_inv,
                                                         transformer_engine::DType otype,
                                                         int scale_offset, int amax_offset,
                                                         int scale_inv_offset) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dgelu = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto dgelu_transpose =
      allocateTorchTensor(grad_output.size(1), grad_output.size(0), DType::kByte);
  auto gelu_input_cu = makeTransformerEngineTensor(gelu_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto output_cu = makeTransformerEngineTensor(dgelu.data_ptr(),
                                               dgelu_transpose.data_ptr(),
                                               {M, N},
                                               {N, M},
                                               otype,
                                               amax_dptr,
                                               scale_dptr,
                                               scale_inv_dptr,
                                               scale_inv_dptr);

  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_cast_transpose_dbias_dgelu(input_cu.data(), gelu_input_cu.data(), output_cu.data(),
                                  dbias_cu.data(), workspace.data(),
                                  at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_cast_transpose_dbias_dgelu(input_cu.data(), gelu_input_cu.data(), output_cu.data(),
                                  dbias_cu.data(), workspace.data(),
                                  at::cuda::getCurrentCUDAStream());

  return {grad_bias, dgelu, dgelu_transpose};
}

void fused_dswiglu_cast_transpose(at::Tensor grad_output, at::Tensor input, at::Tensor grad_input,
                                  at::Tensor grad_input_transpose, at::Tensor scale,
                                  at::Tensor amax, at::Tensor scale_inv,
                                  transformer_engine::DType otype, int scale_offset,
                                  int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine::pytorch;

  // Tensor dimensions
  auto outer_dim = [](const at::Tensor& tensor) -> size_t {
    return tensor.numel() / tensor.size(-1);
  };
  const auto M = outer_dim(grad_output);
  const auto N = static_cast<size_t>(grad_output.size(-1));

  // Check tensor dims
  NVTE_CHECK(grad_output.dim() == 2, "Expected grad output tensor to have 2 dims, but found ",
             grad_output.dim());
  NVTE_CHECK(input.dim() == 2, "Expected input tensor to have 2 dims, but found ", input.dim());
  NVTE_CHECK(outer_dim(input) == M, "Expected input tensor to have outer dimension of ", M,
             ", but found ", outer_dim(input));
  NVTE_CHECK(input.size(-1) == 2 * N, "Expected input tensor to have inner dimension of ", 2 * N,
             ", but found ", input.size(-1));
  NVTE_CHECK(grad_input.dim() == 2, "Expected grad input tensor to have 2 dims, but found ",
             grad_input.dim());
  NVTE_CHECK(outer_dim(grad_input) == M, "Expected grad input tensor to have outer dimension of ",
             M, ", but found ", outer_dim(grad_input));
  NVTE_CHECK(grad_input.size(-1) == 2 * N, "Expected grad input tensor to have inner dimension of ",
             2 * N, ", but found ", grad_input.size(-1));
  NVTE_CHECK(grad_input_transpose.dim() == 2,
             "Expected grad input transpose tensor to have 2 dims, but found ",
             grad_input_transpose.dim());
  NVTE_CHECK(grad_input_transpose.size(0) == 2 * N,
             "Expected grad input tensor to have outer dimension of ", 2 * N, ", but found ",
             grad_input_transpose.size(0));
  NVTE_CHECK(grad_input_transpose.size(1) == M,
             "Expected grad input tensor to have outer dimension of ", M, ", but found ",
             grad_input_transpose.size(1));

  // Check tensor format
  NVTE_CHECK(grad_output.is_contiguous(), "Expected grad output tensor to be contiguous");
  NVTE_CHECK(input.is_contiguous(), "Expected input tensor to be contiguous");
  NVTE_CHECK(grad_input.is_contiguous(), "Expected grad input tensor to be contiguous");
  NVTE_CHECK(grad_input_transpose.is_contiguous(),
             "Expected grad input transpose tensor to be contiguous");
  NVTE_CHECK(grad_output.scalar_type() == input.scalar_type(),
             "Expected grad output tensor and input tensor to have same dtype");
  NVTE_CHECK(grad_input.scalar_type() == at::ScalarType::Byte,
             "Expected grad input tensor to be uint8 buffer");
  NVTE_CHECK(grad_input_transpose.scalar_type() == at::ScalarType::Byte,
             "Expected grad input transpose tensor to be uint8 buffer");

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  auto dy_cu = makeTransformerEngineTensor(grad_output);
  auto x_cu = makeTransformerEngineTensor(input);
  auto dx_cu = makeTransformerEngineTensor(grad_input.data_ptr(),
                                           grad_input_transpose.data_ptr(),
                                           {M, 2 * N},
                                           {2 * N, M},
                                           otype,
                                           amax_dptr,
                                           scale_dptr,
                                           scale_inv_dptr,
                                           scale_inv_dptr);

  // Launch kernel
  nvte_dswiglu_cast_transpose(dy_cu.data(), x_cu.data(), dx_cu.data(),
                              at::cuda::getCurrentCUDAStream());
}

std::vector<py::object> fused_multi_quantize(std::vector<py::handle> input_list,
                          std::optional<std::vector<py::handle>> output_list,
                          std::vector<py::handle> quantizer_list,
                          transformer_engine::DType otype) {
  using namespace transformer_engine::pytorch;
  std::vector<NVTETensor> nvte_tensor_input_list;
  std::vector<NVTETensor> nvte_tensor_output_list;
  std::vector<py::object> py_output_objects_list;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;
  auto none = py::none();
  // create TE tensors from input
  for(int i = 0; i < input_list.size(); i++) {
    
    auto input_tensor = makeTransformerEngineTensor(input_list[i], none);
    const NVTEShape input_shape = input_tensor.shape();

    transformer_engine::TensorWrapper output_tensor;
    
    if (output_list == std::nullopt) {
      std::unique_ptr<Quantizer> quantizer = convert_quantizer(quantizer_list[i]);
      std::vector<size_t> output_shape(input_shape.data, input_shape.data + input_shape.ndim);
      py::object o;
      std::tie(output_tensor, o) = quantizer->create_tensor(output_shape, otype);
      py_output_objects_list.push_back(o);
    }
    else {
      output_tensor = makeTransformerEngineTensor((*output_list)[i], quantizer_list[i]);
    }
    if(input_tensor.numel() == 0) continue;
    
    
    nvte_tensor_output_list.emplace_back(output_tensor.data());
    nvte_tensor_input_list.emplace_back(input_tensor.data());
    tensor_wrappers.emplace_back(std::move(input_tensor));
    tensor_wrappers.emplace_back(std::move(output_tensor));
  }

  // Check tensor lists
  NVTE_CHECK(nvte_tensor_output_list.size() == nvte_tensor_input_list.size(),
             "Number of input and output tensors must match");

  // Launch TE kernel
 
  nvte_multi_cast_transpose(nvte_tensor_input_list.size(), nvte_tensor_input_list.data(),
                            nvte_tensor_output_list.data(), at::cuda::getCurrentCUDAStream());
  return py_output_objects_list;
}


at::Tensor fp8_transpose(at::Tensor input, transformer_engine::DType otype, std::optional<at::Tensor> output) {
  using namespace transformer_engine::pytorch;

  const auto dim = input.dim();
  NVTE_CHECK(dim >= 2, "Need at least 2D tensor to transpose.");

  if (input.dim() > 2) {
    input = input.view({-1, input.size(dim - 1)});
  }

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  at::Tensor out;
  if (output.has_value()) {
    out = *output;
  } else {
    out = allocateTorchTensor(input.size(1), input.size(0), DType::kByte);
  }
  if (M == 0 || N == 0) return out;

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, otype);
  auto output_cu = makeTransformerEngineTensor(out.data_ptr(), {N, M}, otype);

  nvte_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return out;
}

void fp8_transpose_noalloc_noop(at::Tensor input, at::Tensor output, at::Tensor noop,
                                transformer_engine::DType otype) {
  using namespace transformer_engine::pytorch;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, otype);
  auto noop_cu = makeTransformerEngineTensor(noop);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {N, M}, otype);

  nvte_transpose_with_noop(input_cu.data(), noop_cu.data(), output_cu.data(),
                           at::cuda::getCurrentCUDAStream());
}
