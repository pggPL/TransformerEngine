/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <optional>
#include <string>

#include "common.h"
#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "extensions.h"
#include "object.h"
#include "pytorch/csrc/common.h"
#include "transformer_engine/transformer_engine.h"
#include "pybind.h"

namespace {

void* get_data_ptr(MaybeTensor tensor) {
  if (tensor.has_value()) return tensor->data_ptr();
  return nullptr;
}

size_t get_size(MaybeTensor tensor, int dim) {
  if (tensor.has_value()) return static_cast<size_t>(tensor->size(dim));
  return 0;
}

}  // namespace

namespace transformer_engine::pytorch {

namespace detail {

std::vector<size_t> getGemmOutputShape(const NVTEShape& A_shape,
                                       const bool transa,
                                       const NVTEShape& B_shape,
                                       const bool transb) {
  // Flatten outer dims to get 2D matrices
  const size_t A0 = product(A_shape, 0, A_shape.ndim - 1);
  const size_t A1 = A_shape.data[A_shape.ndim - 1];
  const size_t B0 = product(B_shape, 0, B_shape.ndim - 1);
  const size_t B1 = B_shape.data[B_shape.ndim - 1];

  // Check matrix dims
  NVTE_CHECK((transa ? A1 : A0) == (transb ? B0 : B1),
             "Invalid matrix dimensions for GEMM (A=(", A0, ",", A1, "), transa=",
             transa, ", B=(", B0, ",", B1, "), transb=", transb, ")");

  // Construct output dims
  std::vector<size_t> ret;
  if (transb) {
    ret.emplace_back(B1);
  } else {
    // Unflatten B0
    for (size_t i = 0; i < B_shape.ndim - 1; ++i) {
      ret.emplace_back(B_shape.data[i]);
    }
  }
  if (transa) {
    ret.emplace_back(A0);
  } else {
    ret.emplace_back(A1);
  }
  return ret;
}

bool checkGemmShape(const std::vector<size_t>& expected,
                    const NVTEShape& actual) {
  if (expected.size() != actual.ndim) return false;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual.data[i]) return false;
  }
  return true;
}

}  // namespace detail

std::pair<TensorWrapper, py::object> createOutputTensor(const std::vector<size_t>& shape,
                                                        DType dtype,
                                                        py::handle quantizer) {
  std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
  return my_quantizer->create_tensor(shape, dtype);
}

std::vector<py::object> gemm(py::handle A, bool transa, py::handle B, bool transb,
                             py::object D, py::handle quantizer,
                             std::optional<DType> out_dtype,
                             MaybeTensor bias, DType bias_type, bool gelu,
                             bool grad, at::Tensor workspace, size_t workspaceSize,
                             bool accumulate, bool use_split_accumulator) {
  // Input tensors
  NVTE_CHECK(!A.is_none(), "Tensor A has not been provided");
  NVTE_CHECK(!B.is_none(), "Tensor B has not been provided");
  auto none = py::none();
  const TensorWrapper& A_tensor = makeTransformerEngineTensor(A, none);
  const TensorWrapper& B_tensor = makeTransformerEngineTensor(B, none);

  // Check tensor dimensions
  const auto& A_shape = A_tensor.shape();
  const auto& B_shape = B_tensor.shape();
  const auto& D_shape = detail::getGemmOutputShape(A_shape, transa, B_shape, transb);
  NVTE_CHECK(A_shape.ndim >= 1, "Tensor A needs to have at least 1 dimension");
  NVTE_CHECK(B_shape.ndim >= 1, "Tensor B needs to have at least 1 dimension");

  // Output tensor
  TensorWrapper D_tensor;
  if (D.is_none()) {
    DType output_dtype = out_dtype ? *out_dtype : A_tensor.dtype();
    std::tie(D_tensor, D) = createOutputTensor(D_shape, output_dtype, quantizer);
  } else {
    D_tensor = makeTransformerEngineTensor(D, quantizer);
    NVTE_CHECK(detail::checkGemmShape(D_shape, D_tensor.shape()),
               "GEMM output has invalid dims (expected ",
               std::to_string(D_shape), ", got ",
               std::to_string(D_tensor.shape()), ")");
    if (out_dtype) {
      NVTE_CHECK(*out_dtype == D_tensor.dtype(),
                 "GEMM output has invalid dtype (expected ", int(*out_dtype),
                 ", found ", int(D_tensor.dtype()), ")");
    }
  }

  // Bias tensor
  TensorWrapper bias_tensor;
  MaybeTensor bias_grad = std::nullopt;
  if (bias.has_value()) {
    if (!bias->is_contiguous()) {
      bias = bias->contiguous();
    }
    if (!grad) {
      bias_tensor = makeTransformerEngineTensor(*bias);
    } else {
      bias_grad = at::empty_like(*bias);
      bias_tensor = makeTransformerEngineTensor(*bias_grad);
    }
  }

  // Activation input tensor
  MaybeTensor pre_gelu_out = std::nullopt;
  DType gelu_type = bias_type;
  if (gelu && !grad) {
    auto dtype = GetATenDType(bias_type);
    auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
    std::vector<int64_t> torch_shape;
    for (auto v : D_shape) {
      torch_shape.push_back(v);
    }
    *pre_gelu_out = at::empty(torch_shape, opts);
  }
  const auto gelu_shape = gelu ? D_shape : std::vector<size_t>{0};
  auto te_pre_gelu_out =
      makeTransformerEngineTensor(get_data_ptr(pre_gelu_out), gelu_shape, gelu_type);

  // Workspace
  auto te_workspace =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, DType::kByte);

  // Set an external SM Margin to all the GEMMs.
  // This comes in handy when DP is overlapped with GEMMs
  const int device_id = at::cuda::current_device();
  const int sm_count = transformer_engine::cuda::sm_count(device_id);
  int num_math_sms = sm_count - transformer_engine::getenv<int>("NVTE_EXT_MARGIN_SM", sm_count);

  // Launch GEMM
  nvte_cublas_gemm(A_tensor.data(), B_tensor.data(), D_tensor.data(), bias_tensor.data(),
                   te_pre_gelu_out.data(), transa, transb, grad, te_workspace.data(), accumulate,
                   use_split_accumulator, num_math_sms, at::cuda::getCurrentCUDAStream());

  // Pack outputs
  std::vector<py::object> out;
  out.emplace_back(std::move(D));
  out.emplace_back(py::cast(bias_grad));
  out.emplace_back(py::cast(pre_gelu_out));
  return out;
}

}  // namespace transformer_engine::pytorch


void te_atomic_gemm(at::Tensor A, at::Tensor A_scale_inverse, transformer_engine::DType A_type,
                    std::vector<int64_t> A_scaling_mode, bool transa, at::Tensor B,
                    at::Tensor B_scale_inverse, transformer_engine::DType B_type,
                    std::vector<int64_t> B_scaling_mode, bool transb, at::Tensor D,
                    at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
                    at::Tensor bias, transformer_engine::DType bias_type, at::Tensor pre_gelu_out,
                    bool grad, at::Tensor workspace, size_t workspaceSize, bool accumulate,
                    bool use_split_accumulator, int math_sm_count, int m_split, int n_split,
                    bool gemm_producer, at::Tensor counter) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  // TODO: Handle scaling modes
  NVTEScalingMode nvte_scaling_modeA = NVTE_DELAYED_TENSOR_SCALING;
  NVTEScalingMode nvte_scaling_modeB = NVTE_DELAYED_TENSOR_SCALING;

  auto te_A = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,
      nullptr, nullptr, A_scale_inverse.data_ptr(), getTensorShape(A_scale_inverse),
      nvte_scaling_modeA);
  auto te_B = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,
      nullptr, nullptr, B_scale_inverse.data_ptr(), getTensorShape(B_scale_inverse),
      nvte_scaling_modeB);
  // TODO: D_scale_inv cannot be nullptr when D_type is FP8.
  auto te_D = makeTransformerEngineTensor(
      D.data_ptr(), {static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,
      D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
  auto te_bias =
      makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))}, bias_type);
  auto te_counter = makeTransformerEngineTensor(
      counter.data_ptr(), {static_cast<size_t>(counter.size(0))}, DType::kInt32);

  const auto gelu_shape = pre_gelu_out.data_ptr() == nullptr
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto te_pre_gelu_out = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));
  auto te_workspace =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, DType::kByte);

  nvte_cublas_atomic_gemm(te_A.data(), te_B.data(), te_D.data(), te_bias.data(),
                          te_pre_gelu_out.data(), transa, transb, grad, te_workspace.data(),
                          accumulate, use_split_accumulator, math_sm_count, m_split, n_split,
                          gemm_producer, te_counter.data(), at::cuda::getCurrentCUDAStream());
}

void te_grouped_gemm(std::vector<at::Tensor> A, at::Tensor A_scale_inverse, int A_offset,
                     transformer_engine::DType A_type, std::vector<int64_t> A_scaling_mode,
                     bool transa, std::vector<at::Tensor> B, at::Tensor B_scale_inverse,
                     int B_offset, transformer_engine::DType B_type,
                     std::vector<int64_t> B_scaling_mode, bool transb, std::vector<at::Tensor> D,
                     int D_offset, at::Tensor D_scale, transformer_engine::DType D_type,
                     at::Tensor D_amax, std::vector<at::Tensor> bias,
                     transformer_engine::DType bias_type, std::vector<at::Tensor> pre_gelu_out,
                     bool grad, std::vector<at::Tensor> workspace, size_t workspaceSize,
                     bool accumulate, bool use_split_accumulator, int math_sm_count) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;
  std::vector<NVTETensor> te_A, te_B, te_D, te_bias, te_pre_gelu_out, te_workspace;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;

  // TODO: Handle scaling modes
  NVTEScalingMode nvte_scaling_modeA = NVTE_DELAYED_TENSOR_SCALING;
  NVTEScalingMode nvte_scaling_modeB = NVTE_DELAYED_TENSOR_SCALING;

  auto make_tensor = [&tensor_wrappers](void* dptr, const std::vector<size_t>& shape,
                                        transformer_engine::DType dtype, void* amax_dptr,
                                        void* scale_dptr, void* scale_inv_dptr,
                                        NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING)
    -> NVTETensor {
    // TODO(ksivamani): check scaling factor shapes for mxfp8.
    tensor_wrappers.emplace_back(makeTransformerEngineTensor(
        dptr, shape, dtype, amax_dptr, scale_dptr, scale_inv_dptr, {1}, scaling_mode));
    return tensor_wrappers.back().data();
  };
  for (size_t i = 0; i < A.size(); i++) {
    if (A[i].numel() == 0 || B[i].numel() == 0) {
      if (D[i].numel() != 0 && !accumulate) D[i].zero_();
      if (bias[i].numel() != 0 && grad) {
        if (B[i].numel() == 0) {
          bias[i].zero_();
        } else {
          bias[i].copy_(B[i].sum(0));
        }
      }
      if (pre_gelu_out[i].numel() != 0) pre_gelu_out[i].zero_();
      continue;
    }

    NVTE_CHECK(A[i].is_contiguous(), "A[", i, "] must be contiguous.");
    NVTE_CHECK(B[i].is_contiguous(), "B[", i, "] must be contiguous.");
    NVTE_CHECK(D[i].is_contiguous(), "D[", i, "] must be contiguous.");

    te_A.emplace_back(make_tensor(
        A[i].data_ptr(), {static_cast<size_t>(A[i].size(0)), static_cast<size_t>(A[i].size(1))},
        A_type, nullptr, nullptr, getDataPtr(A_scale_inverse, A_offset + i), nvte_scaling_modeA));
    te_B.emplace_back(make_tensor(
        B[i].data_ptr(), {static_cast<size_t>(B[i].size(0)), static_cast<size_t>(B[i].size(1))},
        B_type, nullptr, nullptr, getDataPtr(B_scale_inverse, B_offset + i), nvte_scaling_modeB));
    // TODO: D_scale_inv cannot be nullptr when D_type is FP8.
    te_D.emplace_back(make_tensor(
        D[i].data_ptr(), {static_cast<size_t>(D[i].size(0)), static_cast<size_t>(D[i].size(1))},
        D_type, getDataPtr(D_amax, D_offset + i), getDataPtr(D_scale, D_offset + i), nullptr));
    te_bias.emplace_back(make_tensor(bias[i].data_ptr(), {static_cast<size_t>(bias[i].size(0))},
                                     bias_type, nullptr, nullptr, nullptr));

    const auto gelu_shape = pre_gelu_out[i].data_ptr() == nullptr
                                ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out[i].size(0))}
                                : std::vector<size_t>{static_cast<size_t>(pre_gelu_out[i].size(0)),
                                                      static_cast<size_t>(pre_gelu_out[i].size(1))};
    te_pre_gelu_out.emplace_back(make_tensor(
        pre_gelu_out[i].data_ptr(), gelu_shape,
        GetTransformerEngineDType(pre_gelu_out[i].scalar_type()), nullptr, nullptr, nullptr));
  }
  for (size_t i = 0; i < workspace.size(); i++) {
    te_workspace.emplace_back(make_tensor(workspace[i].data_ptr(), {workspaceSize}, DType::kByte,
                                          nullptr, nullptr, nullptr));
  }

  // For now, we only have multi-stream cublas backend.
  nvte_multi_stream_cublas_gemm(te_A.data(), te_B.data(), te_D.data(), te_bias.data(),
                                te_pre_gelu_out.data(), te_A.size(), transa, transb, grad,
                                te_workspace.data(), accumulate, use_split_accumulator,
                                math_sm_count, at::cuda::getCurrentCUDAStream());
}

void te_grouped_gemm_single_output(
    std::vector<at::Tensor> A, std::vector<at::Tensor> A_scale_inverse, int A_offset,
    transformer_engine::DType A_type, bool transa, std::vector<at::Tensor> B,
    at::Tensor B_scale_inverse, int B_offset, transformer_engine::DType B_type, bool transb,
    std::vector<int64_t> m_splits, at::Tensor D, int D_offset, at::Tensor D_scale,
    transformer_engine::DType D_type, at::Tensor D_amax, std::vector<at::Tensor> bias,
    transformer_engine::DType bias_type, std::vector<at::Tensor> pre_gelu_out, bool grad,
    std::vector<at::Tensor> workspace, size_t workspaceSize, bool accumulate,
    bool use_split_accumulator, int math_sm_count) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;
  std::vector<NVTETensor> te_A, te_B, te_D, te_bias, te_pre_gelu_out, te_workspace;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr, const std::vector<size_t>& shape,
                                        transformer_engine::DType dtype, void* amax_dptr,
                                        void* scale_dptr, void* scale_inv_dptr) -> NVTETensor {
    tensor_wrappers.emplace_back(
        makeTransformerEngineTensor(dptr, shape, dtype, amax_dptr, scale_dptr, scale_inv_dptr));
    return tensor_wrappers.back().data();
  };
  NVTE_CHECK(D.is_contiguous(), "D must be contiguous.");
  void* d_i_ptr = reinterpret_cast<void*>(D.data_ptr());
  for (size_t i = 0; i < A.size(); i++) {
    if (m_splits[i] == 0) continue;
    NVTE_CHECK(A[i].data_ptr() != nullptr, "A[", i, "] must not be nullptr.");
    NVTE_CHECK(B[i].data_ptr() != nullptr, "B[", i, "] must not be nullptr.");
    NVTE_CHECK(A[i].is_contiguous(), "A[", i, "] must be contiguous.");
    NVTE_CHECK(B[i].is_contiguous(), "B[", i, "] must be contiguous.");
    te_A.emplace_back(make_tensor(
        A[i].data_ptr(), {static_cast<size_t>(A[i].size(0)), static_cast<size_t>(A[i].size(1))},
        A_type, nullptr, nullptr, getDataPtr(A_scale_inverse[i], A_offset)));
    te_B.emplace_back(make_tensor(
        B[i].data_ptr(), {static_cast<size_t>(B[i].size(0)), static_cast<size_t>(B[i].size(1))},
        B_type, nullptr, nullptr, getDataPtr(B_scale_inverse, B_offset + i)));
    // TODO: D_scale_inv cannot be nullptr when D_type is FP8.
    te_D.emplace_back(make_tensor(
        d_i_ptr, {static_cast<size_t>(m_splits[i]), static_cast<size_t>(A[i].size(0))}, D_type,
        getDataPtr(D_amax, D_offset + i), getDataPtr(D_scale, D_offset + i), nullptr));
    te_bias.emplace_back(make_tensor(bias[i].data_ptr(), {static_cast<size_t>(bias[i].size(0))},
                                     bias_type, nullptr, nullptr, nullptr));

    const auto gelu_shape = pre_gelu_out[i].data_ptr() == nullptr
                                ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out[i].size(0))}
                                : std::vector<size_t>{static_cast<size_t>(pre_gelu_out[i].size(0)),
                                                      static_cast<size_t>(pre_gelu_out[i].size(1))};
    te_pre_gelu_out.emplace_back(make_tensor(
        pre_gelu_out[i].data_ptr(), gelu_shape,
        GetTransformerEngineDType(pre_gelu_out[i].scalar_type()), nullptr, nullptr, nullptr));
    // Move the D pointer to the next split.
    char* char_ptr = reinterpret_cast<char*>(d_i_ptr);
    char_ptr += m_splits[i] * A[i].size(0) * D.element_size();
    d_i_ptr = reinterpret_cast<void*>(char_ptr);
  }
  for (size_t i = 0; i < workspace.size(); i++) {
    te_workspace.emplace_back(make_tensor(workspace[i].data_ptr(), {workspaceSize}, DType::kByte,
                                          nullptr, nullptr, nullptr));
  }

  // For now, we only have multi-stream cublas backend.
  nvte_multi_stream_cublas_gemm(te_A.data(), te_B.data(), te_D.data(), te_bias.data(),
                                te_pre_gelu_out.data(), te_A.size(), transa, transb, grad,
                                te_workspace.data(), accumulate, use_split_accumulator,
                                math_sm_count, at::cuda::getCurrentCUDAStream());
}
