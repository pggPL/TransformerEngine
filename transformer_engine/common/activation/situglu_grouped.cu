/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <cmath>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/math.h"
#include "../utils.cuh"
#include "transformer_engine/activation.h"

namespace transformer_engine {
namespace {

constexpr int kThreads = 256;

template <typename T, bool Backward>
__global__ void grouped_situglu_kernel(const T *input, const T *grad, T *output,
                                       const int64_t *group_sizes, size_t groups, size_t capacity,
                                       size_t hidden, SiTUGLUParam params) {
  __shared__ int64_t counts[kThreads];
  int64_t count = 0;
  if (group_sizes != nullptr) {
    for (size_t group = threadIdx.x; group < groups; group += blockDim.x) {
      count += group_sizes[group];
    }
  }
  counts[threadIdx.x] = count;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) counts[threadIdx.x] += counts[threadIdx.x + stride];
    __syncthreads();
  }
  const size_t active_rows = groups == 0 ? 0 : (group_sizes == nullptr ? capacity : counts[0]);
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < capacity * hidden;
       index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t row = index / hidden;
    const size_t col = index % hidden;
    const size_t gate_index = row * 2 * hidden + col;
    const size_t up_index = gate_index + hidden;
    float gate_value = 0.0f;
    float up_value = 0.0f;
    if (row < active_rows) {
      const float gate = static_cast<float>(input[gate_index]);
      const float up = static_cast<float>(input[up_index]);
      if constexpr (Backward) {
        const float dy = static_cast<float>(grad[index]);
        gate_value =
            dy * dsitu_gate<float, float>(gate, params) * situ_up<float, float>(up, params);
        up_value = dy * situ_gate<float, float>(gate, params) * dsitu_up<float, float>(up, params);
      } else {
        gate_value = situ_gate<float, float>(gate, params) * situ_up<float, float>(up, params);
      }
    }
    if constexpr (Backward) {
      output[gate_index] = static_cast<T>(gate_value);
      output[up_index] = static_cast<T>(up_value);
    } else {
      output[index] = static_cast<T>(gate_value);
    }
  }
}

template <bool Backward>
void grouped_situglu(const NVTEGroupedTensor input_handle, const NVTEGroupedTensor grad_handle,
                     NVTEGroupedTensor output_handle, float beta1, float beta2,
                     cudaStream_t stream) {
  const auto *input = convertNVTEGroupedTensorCheck(input_handle);
  auto *output = convertNVTEGroupedTensorCheck(output_handle);
  NVTE_CHECK(std::isfinite(beta1) && beta1 > 0 && std::isfinite(beta2) && beta2 > 0,
             "SiTU soft-cap parameters must be finite and positive.");
  NVTE_CHECK(input->logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "Grouped SiTU requires rank-2 logical shapes.");
  NVTE_CHECK(input->all_same_last_dim() && output->all_same_last_dim(),
             "Grouped SiTU requires a uniform feature dimension.");
  const size_t capacity = input->logical_shape.data[0];
  const size_t input_width = input->logical_shape.data[1];
  NVTE_CHECK(input_width > 0 && input_width % 2 == 0,
             "Grouped SiTU input width must be positive and even.");
  const size_t hidden = input_width / 2;
  NVTE_CHECK(output->num_tensors == input->num_tensors &&
                 output->logical_shape.data[0] == capacity &&
                 output->logical_shape.data[1] == (Backward ? input_width : hidden),
             "Grouped SiTU output shape or group count is inconsistent with input.");
  const DType dtype = input->data.dtype;
  NVTE_CHECK(dtype == DType::kFloat32 || dtype == DType::kFloat16 || dtype == DType::kBFloat16,
             "Grouped SiTU supports float32, float16 and bfloat16.");
  NVTE_CHECK(output->data.dtype == dtype, "Grouped SiTU input/output dtypes must match.");
  NVTE_CHECK(input->data.numel() >= capacity * input_width &&
                 output->data.numel() >= capacity * (Backward ? input_width : hidden),
             "Grouped SiTU input/output storage is smaller than its logical shape.");
  if (input->first_dims.has_data()) {
    NVTE_CHECK(
        input->first_dims.dtype == DType::kInt64 && input->first_dims.numel() == input->num_tensors,
        "Grouped SiTU first_dims must contain one int64 count per group.");
  }
  const auto *grad = Backward ? convertNVTEGroupedTensorCheck(grad_handle) : nullptr;
  if constexpr (Backward) {
    NVTE_CHECK(grad->logical_shape.ndim == 2 && grad->all_same_last_dim() &&
                   grad->logical_shape.data[0] == capacity &&
                   grad->logical_shape.data[1] == hidden &&
                   grad->num_tensors == input->num_tensors && grad->data.dtype == dtype,
               "Grouped SiTU gradient shape, dtype or group count is inconsistent with input.");
    NVTE_CHECK(grad->data.numel() >= capacity * hidden,
               "Grouped SiTU gradient storage is smaller than its logical shape.");
  }
  if (capacity == 0) return;
  NVTE_CHECK(input->data.has_data() && output->data.has_data(),
             "Grouped SiTU input/output data must be allocated.");
  if constexpr (Backward) NVTE_CHECK(grad->data.has_data(), "Grouped SiTU grad is not allocated.");
  const auto *sizes = static_cast<const int64_t *>(input->first_dims.dptr);
  const int blocks = std::min<size_t>((capacity * hidden + kThreads - 1) / kThreads, 1024);
  const SiTUGLUParam params{beta1, beta2};
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      dtype, T,
      grouped_situglu_kernel<T, Backward><<<blocks, kThreads, 0, stream>>>(
          static_cast<const T *>(input->data.dptr),
          Backward ? static_cast<const T *>(grad->data.dptr) : nullptr,
          static_cast<T *>(output->data.dptr), sizes, input->num_tensors, capacity, hidden,
          params););
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace
}  // namespace transformer_engine

void nvte_group_situglu(const NVTEGroupedTensor input, NVTEGroupedTensor output, float beta1,
                        float beta2, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_situglu);
  transformer_engine::grouped_situglu<false>(input, nullptr, output, beta1, beta2, stream);
}

void nvte_group_dsituglu(const NVTEGroupedTensor grad, const NVTEGroupedTensor input,
                         NVTEGroupedTensor output, float beta1, float beta2, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_dsituglu);
  transformer_engine::grouped_situglu<true>(input, grad, output, beta1, beta2, stream);
}
