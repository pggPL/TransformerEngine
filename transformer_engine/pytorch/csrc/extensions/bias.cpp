/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/python/interop.h>
#include <torch/csrc/stable/tensor.h>

#include <utility>
#include <vector>

#include "common.h"
#include "extensions.h"
#include "pybind.h"
#include "transformer_engine/cast.h"
#include "transformer_engine/transformer_engine.h"

namespace nb = nanobind;

namespace transformer_engine {
namespace pytorch {

namespace {
// stable Tensor -> Python object (new ref).
inline nb::object wrap_tensor(const torch::stable::Tensor &t) {
  return nb::steal<nb::object>(nb::handle(static_cast<PyObject *>(torch::stable::to_pyobject(t))));
}
}  // namespace

std::vector<nb::object> bgrad_quantize(const torch::stable::Tensor &grad_output,
                                       nb::handle quantizer) {
  using namespace transformer_engine::pytorch::detail;
  init_extension();

  // Grad output tensor
  auto grad_output_torch = torch::stable::contiguous(grad_output);
  const TensorWrapper &grad_output_nvte = makeTransformerEngineTensor(grad_output_torch);
  const auto shape = getTensorShape(grad_output_torch);
  auto grad_output_dtype = GetTransformerEngineDType(grad_output_torch.scalar_type());

  // Construct grad bias tensor
  const int64_t bias_size = static_cast<int64_t>(shape.back());
  auto grad_bias_torch = allocateTorchTensor(bias_size, grad_output_dtype);
  auto grad_bias_nvte = makeTransformerEngineTensor(grad_bias_torch);

  // Unquantized impl only requires computing grad bias
  if (quantizer.is_none()) {
    if (product(shape) == 0) {
      torch::stable::fill_(grad_bias_torch, 0);
    } else {
      torch::stable::sum_out(grad_bias_torch, torch::stable::reshape(grad_output_torch,
                                                                     {-1, bias_size}),
                             {0});
    }
    return {wrap_tensor(grad_bias_torch), wrap_tensor(grad_output_torch)};
  }

  // Construct grad input tensor
  auto quantizer_cpp = convert_quantizer(quantizer);
  auto [grad_input_nvte, grad_input_py] = quantizer_cpp->create_tensor(shape, grad_output_dtype);

  // Trivial impl if tensors are empty
  if (product(shape) == 0) {
    torch::stable::fill_(grad_bias_torch, 0);
    return {wrap_tensor(grad_bias_torch), std::move(grad_input_py)};
  }

  // Check if fused kernel is supported
  bool with_fused_kernel = false;
  if (detail::IsFloat8Quantizers(quantizer.ptr())) {
    // TODO(stable-abi): needs device compute-capability query (was
    // at::cuda::getCurrentDeviceProperties()->major/minor).
    const size_t sm_arch = getDeviceComputeCapability();
    if (sm_arch >= 100) {
      // Fused kernel for dbias + FP8 cast on SM arch 10.0+
      with_fused_kernel = true;
    } else if (quantizer_cpp->rowwise_usage && quantizer_cpp->columnwise_usage) {
      // Fused kernel for dbias + FP8 cast + FP8 transpose
      with_fused_kernel = true;
    }
  } else if (detail::IsMXFP8Quantizers(quantizer.ptr())) {
    // Fused kernel for dbias + MXFP8 quantize
    with_fused_kernel = true;
  }

  // Apply unfused impl if fused kernel is not supported
  if (!with_fused_kernel) {
    torch::stable::sum_out(grad_bias_torch,
                           torch::stable::reshape(grad_output_torch, {-1, bias_size}), {0});
    quantizer_cpp->quantize(grad_output_nvte, grad_input_nvte);
    return {wrap_tensor(grad_bias_torch), std::move(grad_input_py)};
  }

  // Query workspace size
  TensorWrapper workspace_nvte;
  torch::stable::Tensor workspace_torch;
  auto stream = getCurrentCUDAStream();
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_dbias(grad_output_nvte.data(), grad_input_nvte.data(), grad_bias_nvte.data(),
                        workspace_nvte.data(), stream);
  });

  // Allocate workspace
  if (workspace_nvte.ndim() > 0 && workspace_nvte.numel() > 0) {
    workspace_torch = allocateSpace(workspace_nvte.shape(), workspace_nvte.dtype());
    workspace_nvte = makeTransformerEngineTensor(workspace_torch.data_ptr(), workspace_nvte.shape(),
                                                 workspace_nvte.dtype());
  }

  // Launch fused kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_dbias(grad_output_nvte.data(), grad_input_nvte.data(), grad_bias_nvte.data(),
                        workspace_nvte.data(), stream);
  });

  return {wrap_tensor(grad_bias_torch), std::move(grad_input_py)};
}

namespace {

std::vector<nb::object> dact_dbias(
    void (*dact_dbias_func)(const NVTETensor, const NVTETensor, NVTETensor, NVTETensor, NVTETensor,
                            cudaStream_t),
    void (*dact_func)(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t),
    torch::stable::Tensor grad_output_torch, torch::stable::Tensor act_input_torch,
    nb::handle quantizer_py) {
  using namespace transformer_engine::pytorch::detail;
  init_extension();

  // Grad output and activation input tensors
  grad_output_torch = torch::stable::contiguous(grad_output_torch);
  const TensorWrapper &grad_output_nvte = makeTransformerEngineTensor(grad_output_torch);
  const auto output_shape = getTensorShape(grad_output_torch);
  auto grad_output_dtype = GetTransformerEngineDType(grad_output_torch.scalar_type());
  act_input_torch = torch::stable::contiguous(act_input_torch);
  const TensorWrapper &act_input_nvte = makeTransformerEngineTensor(act_input_torch);
  const auto input_shape = getTensorShape(act_input_torch);

  // Construct tensors
  auto quantizer_cpp = convert_quantizer(quantizer_py);
  auto [grad_input_nvte, grad_input_py] =
      quantizer_cpp->create_tensor(input_shape, grad_output_dtype);
  const int64_t bias_size = static_cast<int64_t>(input_shape.back());
  auto grad_bias_torch = allocateTorchTensor(bias_size, grad_output_dtype);
  auto grad_bias_nvte = makeTransformerEngineTensor(grad_bias_torch);

  // Return immediately if tensors are empty
  if (product(output_shape) == 0) {
    torch::stable::fill_(grad_bias_torch, 0);
    return {wrap_tensor(grad_bias_torch), std::move(grad_input_py)};
  }

  // Choose implementation
  enum class Impl {
    UNFUSED,
    FUSED_DACT_DBIAS_QUANTIZE,
    FUSED_DACT_AMAX_FP8,
    FUSED_DACT_AMAX_NVFP4
  };
  Impl impl = Impl::UNFUSED;
  if (detail::IsFloat8Quantizers(quantizer_py.ptr()) ||
      detail::IsMXFP8Quantizers(quantizer_py.ptr())) {
    impl = Impl::FUSED_DACT_DBIAS_QUANTIZE;
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer_py.ptr())) {
    impl = Impl::FUSED_DACT_AMAX_FP8;
  } else if (detail::IsNVFP4Quantizers(quantizer_py.ptr())) {
    auto nvfp4_quantizer_cpp = dynamic_cast<NVFP4Quantizer *>(quantizer_cpp.get());
    NVTE_CHECK(nvfp4_quantizer_cpp != nullptr, "Could not cast to NVFP4 quantizer");
    if (nvfp4_quantizer_cpp->row_scaled_nvfp4 ||
        (nvfp4_quantizer_cpp->with_rht && nvfp4_quantizer_cpp->with_post_rht_amax)) {
      // Amax is handled within NVFP4 quantizer
      impl = Impl::UNFUSED;
    } else {
      impl = Impl::FUSED_DACT_AMAX_NVFP4;
    }
  }

  // Perform compute
  auto stream = getCurrentCUDAStream();
  switch (impl) {
    case Impl::UNFUSED:
      // Unfused dact, dbias, quantize
      {
        auto [temp_nvte, temp_py] =
            NoneQuantizer(nb::none()).create_tensor(input_shape, grad_output_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          dact_func(grad_output_nvte.data(), act_input_nvte.data(), temp_nvte.data(), stream);
        });
        const auto temp_torch = torch::stable::from_pyobject(temp_py.ptr());
        torch::stable::sum_out(grad_bias_torch,
                               torch::stable::reshape(temp_torch, {-1, bias_size}), {0});
        quantizer_cpp->quantize(temp_nvte, grad_input_nvte);
        break;
      }
    case Impl::FUSED_DACT_DBIAS_QUANTIZE:
      // Fused dact-dbias-quantize kernel
      {
        // Query workspace size
        TensorWrapper workspace_nvte;
        NVTE_SCOPED_GIL_RELEASE({
          dact_dbias_func(grad_output_nvte.data(), act_input_nvte.data(), grad_input_nvte.data(),
                          grad_bias_nvte.data(), workspace_nvte.data(), stream);
        });

        // Allocate workspace
        torch::stable::Tensor workspace_torch;
        if (workspace_nvte.ndim() > 0 && workspace_nvte.numel() > 0) {
          workspace_torch = allocateSpace(workspace_nvte.shape(), workspace_nvte.dtype());
          workspace_nvte = makeTransformerEngineTensor(
              workspace_torch.data_ptr(), workspace_nvte.shape(), workspace_nvte.dtype());
        }

        // Launch kernel
        NVTE_SCOPED_GIL_RELEASE({
          dact_dbias_func(grad_output_nvte.data(), act_input_nvte.data(), grad_input_nvte.data(),
                          grad_bias_nvte.data(), workspace_nvte.data(), stream);
        });
        break;
      }
    case Impl::FUSED_DACT_AMAX_FP8:
      // Fused dact-amax kernel, unfused dbias and FP8 quantize
      {
        auto *fp8_quantizer_cpp =
            dynamic_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
        NVTE_CHECK(fp8_quantizer_cpp != nullptr,
                   "Invalid quantizer for fused dact-amax kernel impl");
        auto [temp_nvte, temp_py, amax_buf] =
            fp8_quantizer_cpp->create_unquantized_tensor_with_amax(input_shape, grad_output_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          dact_func(grad_output_nvte.data(), act_input_nvte.data(), temp_nvte.data(), stream);
        });
        const auto temp_torch = torch::stable::from_pyobject(temp_py.ptr());
        torch::stable::sum_out(grad_bias_torch,
                               torch::stable::reshape(temp_torch, {-1, bias_size}), {0});
        fp8_quantizer_cpp->quantize_with_amax(temp_nvte, grad_input_nvte, amax_buf);
        break;
      }
    case Impl::FUSED_DACT_AMAX_NVFP4:
      // Fused dact-amax kernel, unfused dbias and NVFP4 quantize
      {
        auto *nvfp4_quantizer_cpp =
            static_cast<NVFP4Quantizer *>(quantizer_cpp.get());  // Already checked cast is valid
        NVTE_CHECK(nvfp4_quantizer_cpp != nullptr,
                   "Invalid quantizer for fused dact-amax kernel impl");
        auto [temp_nvte, temp_py] = nvfp4_quantizer_cpp->create_unquantized_tensor_with_amax(
            grad_input_nvte, grad_output_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          dact_func(grad_output_nvte.data(), act_input_nvte.data(), temp_nvte.data(), stream);
        });
        const auto temp_torch = torch::stable::from_pyobject(temp_py.ptr());
        torch::stable::sum_out(grad_bias_torch,
                               torch::stable::reshape(temp_torch, {-1, bias_size}), {0});
        nvfp4_quantizer_cpp->quantize_with_amax(temp_nvte, grad_input_nvte);
        break;
      }
    default:
      NVTE_ERROR("Invalid implementation");
  }

  return {wrap_tensor(grad_bias_torch), std::move(grad_input_py)};
}

}  // namespace

std::vector<nb::object> dbias_dgelu(const torch::stable::Tensor &grad_output,
                                    const torch::stable::Tensor &act_input, nb::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_dgelu, nvte_dgelu, grad_output, act_input, quantizer);
}

std::vector<nb::object> dbias_dsilu(const torch::stable::Tensor &grad_output,
                                    const torch::stable::Tensor &act_input, nb::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_dsilu, nvte_dsilu, grad_output, act_input, quantizer);
}

std::vector<nb::object> dbias_drelu(const torch::stable::Tensor &grad_output,
                                    const torch::stable::Tensor &act_input, nb::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_drelu, nvte_drelu, grad_output, act_input, quantizer);
}

std::vector<nb::object> dbias_dqgelu(const torch::stable::Tensor &grad_output,
                                     const torch::stable::Tensor &act_input, nb::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_dqgelu, nvte_dqgelu, grad_output, act_input, quantizer);
}

std::vector<nb::object> dbias_dsrelu(const torch::stable::Tensor &grad_output,
                                     const torch::stable::Tensor &act_input, nb::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_dsrelu, nvte_dsrelu, grad_output, act_input, quantizer);
}

}  // namespace pytorch
}  // namespace transformer_engine
