/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <tuple>
#include <utility>
#include <vector>

#include "../extensions.h"
#include "common/common.h"
#include "pybind.h"

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

size_t get_cublasLt_version() { return cublasLtGetVersion(); }

size_t get_cudnn_version() { return cudnnGetVersion(); }

torch::stable::Tensor splits_to_offsets(const torch::stable::Tensor &first_dims,
                                        int64_t logical_last_dim) {
  NVTE_CHECK(first_dims.is_cuda(), "first_dims must be on CUDA.");
  NVTE_CHECK(first_dims.scalar_type() == torch::headeronly::ScalarType::Long,
             "first_dims must have dtype int64.");
  NVTE_CHECK(first_dims.dim() == 1, "first_dims must be a 1D tensor.");
  NVTE_CHECK(logical_last_dim > 0, "logical_last_dim must be greater than 0.");

  auto first_dims_contiguous = torch::stable::contiguous(first_dims);
  const auto num_tensors = static_cast<size_t>(first_dims_contiguous.numel());
  // TODO(stable-abi): needs torch::stable::new_empty(self, size, ScalarType) with
  // a dtype override (here Long, keeping the source device).
  auto output = torch::stable::new_empty(first_dims_contiguous,
                                         {static_cast<int64_t>(num_tensors) + 1},
                                         torch::headeronly::ScalarType::Long);

  nvte_splits_to_offsets(static_cast<const int64_t *>(first_dims_contiguous.data_ptr()),
                         static_cast<int64_t *>(output.data_ptr()), num_tensors, logical_last_dim,
                         current_cuda_stream());

  return output;
}

std::tuple<torch::stable::Tensor, std::vector<torch::stable::Tensor>> splits_to_offsets_multi(
    const torch::stable::Tensor &split_sizes, const torch::stable::Device &device,
    const std::vector<int64_t> &strides, const std::vector<bool> &include_leading_zero,
    const std::vector<torch::headeronly::ScalarType> &dtypes, bool bulk_allocate_outputs) {
  const size_t num_outputs = strides.size();
  const size_t num_splits = static_cast<size_t>(split_sizes.numel());

  // Check inputs.
  NVTE_CHECK(include_leading_zero.size() == num_outputs && dtypes.size() == num_outputs,
             "strides, include_leading_zero, and dtypes must have matching lengths, but got ",
             strides.size(), ", ", include_leading_zero.size(), ", and ", dtypes.size(), ".");
  // TODO(stable-abi): needs torch::stable::Device::is_cuda().
  NVTE_CHECK(device.is_cuda(), "device must be CUDA.");

  // Convert split sizes to int64 GPU tensor.
  // TODO(stable-abi): needs torch::stable::to(Tensor, ScalarType) and
  // torch::stable::to(Tensor, Device) plus torch::stable::Tensor::device().
  const torch::stable::Tensor split_sizes_i64 =
      split_sizes.scalar_type() == torch::headeronly::ScalarType::Long
          ? split_sizes
          : torch::stable::to(split_sizes, torch::headeronly::ScalarType::Long);
  const torch::stable::Tensor split_sizes_out =
      split_sizes_i64.device() == device ? split_sizes_i64
                                         : torch::stable::to(split_sizes_i64, device);

  // Allocate outputs.
  std::vector<torch::stable::Tensor> outputs;
  outputs.reserve(num_outputs);
  if (bulk_allocate_outputs) {
    std::vector<std::vector<size_t>> shapes;
    shapes.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
      const size_t length = num_splits + (include_leading_zero[i] ? 1 : 0);
      shapes.emplace_back(std::vector<size_t>{length});
    }
    // cuDNN CuTe DSL grouped GEMM kernels require padded_offsets
    // aligned to 16 bytes.
    const std::vector<size_t> alignments(num_outputs, 16);
    outputs = bulk_allocate(shapes, dtypes, device, alignments);
  } else {
    for (size_t i = 0; i < num_outputs; ++i) {
      const int64_t length = static_cast<int64_t>(num_splits) + (include_leading_zero[i] ? 1 : 0);
      // TODO(stable-abi): needs torch::stable::empty(IntArrayRef, ScalarType, Device).
      outputs.emplace_back(torch::stable::empty({length}, dtypes[i], std::nullopt, device));
    }
  }

  // Construct NVTETensors.
  MultiTensorWrapper outputs_nvte(num_outputs);
  std::vector<int> include_leading_zero_int(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    const size_t length = num_splits + (include_leading_zero[i] ? 1 : 0);
    NVTEShape shape = nvte_make_shape(&length, 1);
    NVTEBasicTensor data = {outputs[i].data_ptr(),
                            static_cast<NVTEDType>(GetTransformerEngineDType(dtypes[i])), shape};
    nvte_set_tensor_param_v2(outputs_nvte[i], kNVTERowwiseData, &data, sizeof(data));
    include_leading_zero_int[i] = include_leading_zero[i] ? 1 : 0;
  }

  auto split_sizes_nvte = makeTransformerEngineTensor(split_sizes_out);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_splits_to_offsets_multi(split_sizes_nvte.data(), outputs_nvte.data(), strides.data(),
                                 include_leading_zero_int.data(), num_outputs,
                                 current_cuda_stream());
  });

  return {split_sizes_out, std::move(outputs)};
}

torch::stable::Tensor copy_data_ptrs_to_device(const std::vector<torch::stable::Tensor> &tensors,
                                               const torch::stable::Device &device) {
  // Collect data pointers
  std::vector<uint64_t> ptrs_host;
  ptrs_host.reserve(tensors.size());
  for (const auto &tensor : tensors) {
    ptrs_host.push_back(reinterpret_cast<uintptr_t>(tensor.data_ptr()));
  }

  // Allocate device buffer
  // TODO(stable-abi): needs torch::stable::empty(IntArrayRef, ScalarType, Device).
  auto ptrs_device = torch::stable::empty({static_cast<int64_t>(tensors.size())},
                                          torch::headeronly::ScalarType::Long, std::nullopt, device);

  // Load pointers on device
  nvte_copy_host_to_device_via_kernel(ptrs_host.data(), ptrs_device.data_ptr(),
                                      tensors.size() * sizeof(uint64_t), current_cuda_stream());

  return ptrs_device;
}

}  // namespace transformer_engine::pytorch
