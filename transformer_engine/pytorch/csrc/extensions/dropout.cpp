/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/dropout.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/python/interop.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "../common.h"
#include "../extensions.h"
#include "../pybind.h"
#include "transformer_engine/transformer_engine.h"

namespace nb = nanobind;

namespace transformer_engine {
namespace pytorch {

namespace {
// TODO(stable-abi): torch::stable::accelerator lacks a native cudaStream_t handle.
inline cudaStream_t current_cuda_stream() {
  return static_cast<cudaStream_t>(
      torch::stable::accelerator::getCurrentStream(
          torch::stable::accelerator::getCurrentDeviceIndex())
          .stream());
}

// stable Tensor -> Python object.
inline nb::object to_py(const torch::stable::Tensor& t) {
  return nb::steal<nb::object>(
      nb::handle(static_cast<PyObject*>(torch::stable::to_pyobject(t))));
}
}  // namespace

std::vector<nb::object> dropout_fwd(const nb::handle& input, float dropout_probability,
                                    std::optional<torch::stable::Tensor> out) {
  using namespace transformer_engine::pytorch::detail;

  // Input tensor
  const TensorWrapper input_nvte = makeTransformerEngineTensor(input, nb::none());

  // Allocate output tensor if needed
  if (!out) {
    torch::headeronly::ScalarType dtype = GetATenDType(input_nvte.dtype());
    if (dtype == torch::headeronly::ScalarType::Float8_e4m3fn ||
        dtype == torch::headeronly::ScalarType::Float8_e5m2) {
      // TODO(stable-abi): need a nanobind type caster / helper to turn a Python
      // torch.dtype object into torch::headeronly::ScalarType.
      dtype = nb::cast<torch::headeronly::ScalarType>(input.attr("dtype"));
    }
    const auto shape_uint64 = convertShape(input_nvte.shape());
    const std::vector<int64_t> shape_int64(shape_uint64.begin(), shape_uint64.end());
    // TODO(stable-abi): needs torch::stable::empty(IntArrayRef, ScalarType, DeviceType).
    out = torch::stable::empty(shape_int64, dtype, torch::headeronly::DeviceType::CUDA);
  }
  TensorWrapper out_nvte = makeTransformerEngineTensor(*out);

  // Mask tensor
  auto mask_pyt = allocateTorchTensor(input_nvte.numel() / 8, DType::kByte);
  auto mask_nvte = makeTransformerEngineTensor(mask_pyt);

  // RNG state tensor (default CUDA generator).
  constexpr int64_t rng_elts_per_thread = 4;
  auto philox_args = init_philox_state(get_cuda_generator(std::nullopt), rng_elts_per_thread);
  auto rng_state_pyt = allocateTorchTensor(2, DType::kInt64);
  philox_unpack(philox_args, reinterpret_cast<int64_t*>(rng_state_pyt.data_ptr()));
  auto rng_state_nvte = makeTransformerEngineTensor(rng_state_pyt);

  // Launch kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_dropout_fwd(input_nvte.data(), out_nvte.data(), mask_nvte.data(), rng_state_nvte.data(),
                     dropout_probability, current_cuda_stream());
  });

  return {to_py(*out), to_py(mask_pyt)};
}

nb::object dropout_bwd(const torch::stable::Tensor& grad_output, const torch::stable::Tensor& mask,
                       const float dropout_probability,
                       std::optional<torch::stable::Tensor> grad_input) {
  const auto grad_output_nvte = makeTransformerEngineTensor(grad_output);
  const auto mask_nvte = makeTransformerEngineTensor(mask);
  if (!grad_input) {
    grad_input = torch::stable::empty_like(grad_output);
  }
  auto grad_input_nvte = makeTransformerEngineTensor(*grad_input);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_dropout_bwd(grad_output_nvte.data(), mask_nvte.data(), grad_input_nvte.data(),
                     dropout_probability, current_cuda_stream());
  });
  return to_py(*grad_input);
}

}  // namespace pytorch
}  // namespace transformer_engine
