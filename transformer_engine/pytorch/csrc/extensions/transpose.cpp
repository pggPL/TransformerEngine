/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <optional>

#include "ATen/core/TensorBody.h"
#include "extensions.h"

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

  auto output_cu = makeTransformerEngineTensor(
      grad_output_cast.data_ptr(), grad_output_transpose.data_ptr(), {M, N}, {N, M}, otype,
      amax_dptr, scale_dptr, scale_inv_dptr, scale_inv_dptr);

  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_cast_transpose_dbias(input_cu.data(), output_cu.data(), dbias_cu.data(), workspace.data(),
                            at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_cast_transpose_dbias(input_cu.data(), output_cu.data(), dbias_cu.data(), workspace.data(),
                            at::cuda::getCurrentCUDAStream());

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
  for (int i = 0; i < input_list.size(); i++) {
    auto input_tensor = makeTransformerEngineTensor(input_list[i], none);
    const NVTEShape input_shape = input_tensor.shape();

    transformer_engine::TensorWrapper output_tensor;

    if (output_list == std::nullopt) {
      std::unique_ptr<Quantizer> quantizer = convert_quantizer(quantizer_list[i]);
      std::vector<size_t> output_shape(input_shape.data, input_shape.data + input_shape.ndim);
      py::object o;
      std::tie(output_tensor, o) = quantizer->create_tensor(output_shape, otype);
      py_output_objects_list.push_back(o);
    } else {
      output_tensor = makeTransformerEngineTensor((*output_list)[i], quantizer_list[i]);
    }
    if (input_tensor.numel() == 0) continue;

    nvte_tensor_output_list.emplace_back(output_tensor.data());
    nvte_tensor_input_list.emplace_back(input_tensor.data());
    tensor_wrappers.emplace_back(std::move(input_tensor));
    tensor_wrappers.emplace_back(std::move(output_tensor));
  }

  // Check tensor lists
  NVTE_CHECK(nvte_tensor_output_list.size() == nvte_tensor_input_list.size(),
             "Number of input and output tensors must match");

  // Choose implementation
  bool with_fused_kernel = true;
  for (size_t i = 0; i < nvte_tensor_output_list.size(); i++) {
    if (nvte_tensor_columnwise_data(nvte_tensor_output_list[i]) == nullptr) {
      with_fused_kernel = false;
      break;
    }
  }

  // Launch TE kernel
  if (with_fused_kernel) {
    nvte_multi_cast_transpose(nvte_tensor_input_list.size(),
                              nvte_tensor_input_list.data(),
                              nvte_tensor_output_list.data(),
                              at::cuda::getCurrentCUDAStream());
  } else {
    for (size_t i = 0; i < nvte_tensor_output_list.size(); i++) {
      nvte_quantize(nvte_tensor_input_list[i],
                    nvte_tensor_output_list[i],
                    at::cuda::getCurrentCUDAStream());
    }
  }
  return py_output_objects_list;
}

at::Tensor fp8_transpose(at::Tensor input, transformer_engine::DType otype,
                         std::optional<at::Tensor> output) {
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
