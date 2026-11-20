/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/* Vectorization model (gated):
 *
 * With no GLU interleave, the row is laid out as:
 *   [ act[0:H] | gate[0:H] ]
 * With GLU interleave (e.g. 32):
 *   [ act[0:32] | gate[0:32] | act[32:64] | gate[32:64] | ... ]
 *
 * Backward uses fused SiLU / ClampedSiLU specializations (one sigmoid) matching
 * gated_fp8.cuh; other ActOP/DActOP pairs call the ops directly.
 */

#include <algorithm>
#include <type_traits>

#include "../common.h"
#include "../util/math.h"
#include "./scaled_activation.h"

namespace transformer_engine {
namespace {

using namespace detail::scaled_activation;

using WarpReducer = Reducer<float, 1, 1, 1>;

// blockDim.x must be a multiple of warp size and <= kReductionThreads.
__device__ __forceinline__ float block_reduce_sum(float value, float *smem) {
  const int lane = threadIdx.x % THREADS_PER_WARP;
  const int warp = threadIdx.x / THREADS_PER_WARP;
  const int num_warps = blockDim.x / THREADS_PER_WARP;
  Empty params = {};
  WarpReducer reducer(params, /*bidm=*/0, /*bidn=*/0, /*warp_m=*/0, /*warp_n=*/0, lane,
                      /*smem=*/nullptr);
  Sum<float> sum;

  value = reducer.reduce(value, sum);
  if (lane == 0) {
    smem[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < num_warps ? smem[lane] : 0.0f;
  return warp == 0 ? reducer.reduce(value, sum) : value;
}

const int32_t *num_rows_pointer(const NVTETensor nvte_num_rows, const char *api_name) {
  if (nvte_num_rows == nullptr) return nullptr;
  const Tensor *num_rows = convertNVTETensorCheck(nvte_num_rows);
  NVTE_CHECK(num_rows->data.dtype == DType::kInt32, api_name,
             ": num_rows must be an int32 device scalar.");
  NVTE_CHECK(num_rows->data.numel() == 1, api_name, ": num_rows must hold one element.");
  return reinterpret_cast<const int32_t *>(num_rows->data.dptr);
}

// ---------------------------------------------------------------------------
// Device helpers: fused Act / DAct (IEEE sigmoid / expf)
// ---------------------------------------------------------------------------

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
__device__ __forceinline__ float gated_forward_value(const float act_in, const float gate_in,
                                                     const ParamOP &param) {
  const float act_elt = ActOP(act_in, param);
  float gate_elt = gate_in;
  if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
    gate_elt = fminf(fmaxf(-param.limit, gate_in), param.limit) + param.glu_linear_offset;
  } else if constexpr (std::is_same<ParamOP, SiTUGLUParam>::value) {
    gate_elt = situ_up<float, float>(gate_in, param);
  }
  return act_elt * gate_elt;
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
__device__ __forceinline__ void gated_backward_values(const float act_in, const float gate_in,
                                                      const ParamOP &param, float *dact,
                                                      float *dgate, float *unscaled) {
  Empty empty = {};
  float act_elt = 0.0f;
  float gate_elt = gate_in;
  float dact_elt = 0.0f;
  float dgate_elt = 1.0f;

  if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
    if (gate_in > param.limit || gate_in < -param.limit) {
      dgate_elt = 0.f;
    }
    gate_elt = fminf(fmaxf(-param.limit, gate_in), param.limit) + param.glu_linear_offset;
  } else if constexpr (std::is_same<ParamOP, SiTUGLUParam>::value) {
    const float t = fast_tanhf(gate_in / param.beta2);
    gate_elt = param.beta2 * t;
    dgate_elt = 1.0f - t * t;
  }

  if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
    const bool dact_mask = act_in <= param.limit;
    const float clamped_act_in = fminf(act_in, param.limit);
    const float s = sigmoid<float, float>(param.alpha * clamped_act_in, empty);
    act_elt = clamped_act_in * s;
    dact_elt = dact_mask ? s + param.alpha * clamped_act_in * s * (1.0f - s) : 0.0f;
  } else if constexpr (std::is_same<ParamOP, Empty>::value) {
    if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
      const float s = sigmoid<float, float>(act_in, empty);
      act_elt = act_in * s;
      dact_elt = s + act_in * s * (1.0f - s);
    } else {
      act_elt = ActOP(act_in, param);
      dact_elt = DActOP(act_in, param);
    }
  } else if constexpr (std::is_same<ParamOP, SiTUGLUParam>::value) {
    const float t = fast_tanhf(act_in / param.beta1);
    const float s = sigmoidf(act_in);
    act_elt = param.beta1 * t * s;
    dact_elt = (1.0f - t * t) * s + param.beta1 * t * s * (1.0f - s);
  } else {
    act_elt = ActOP(act_in, param);
    dact_elt = DActOP(act_in, param);
  }

  *unscaled = act_elt * gate_elt;
  *dact = dact_elt * gate_elt;
  *dgate = act_elt * dgate_elt;
}

// ---------------------------------------------------------------------------
// Gated kernels
// ---------------------------------------------------------------------------

template <int nvec, typename InputT, typename ScaleT, typename OutputT, typename ParamOP,
          float (*ActOP)(float, const ParamOP &)>
__global__ void __launch_bounds__(kThreads, 4)
    scaled_gated_forward_kernel(const InputT *__restrict__ input,
                                const ScaleT *__restrict__ act_scales, OutputT *__restrict__ output,
                                const int32_t *__restrict__ num_rows, const size_t rows,
                                const size_t hidden, const size_t segment_size,
                                const size_t num_segments, const size_t num_vectors_per_segment,
                                const ParamOP param) {
  size_t active_rows = rows;
  if (num_rows != nullptr) {
    const int32_t n = *num_rows;
    active_rows = n < 0 ? 0 : std::min(rows, static_cast<size_t>(n));
  }
  const size_t total_vectors = active_rows * num_segments * num_vectors_per_segment;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total_vectors;
       tid += gridDim.x * blockDim.x) {
    const size_t vector_idx = tid % num_vectors_per_segment;
    const size_t segment = (tid / num_vectors_per_segment) % num_segments;
    const size_t row = tid / (num_vectors_per_segment * num_segments);
    const size_t input_segment_offset = row * hidden * 2 + segment * segment_size * 2;
    const size_t output_segment_offset = row * hidden + segment * segment_size;

    VectorizedLoader<InputT, nvec, true> act_loader(input + input_segment_offset, segment_size);
    VectorizedLoader<InputT, nvec, true> gate_loader(input + input_segment_offset + segment_size,
                                                     segment_size);
    VectorizedStorer<OutputT, nvec, true> output_storer(output + output_segment_offset,
                                                        segment_size);
    act_loader.load(vector_idx, segment_size);
    gate_loader.load(vector_idx, segment_size);
    const float scale = static_cast<float>(act_scales[row]);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      const float unscaled = gated_forward_value<ParamOP, ActOP>(
          static_cast<float>(act_loader.separate()[lane]),
          static_cast<float>(gate_loader.separate()[lane]), param);
      output_storer.separate()[lane] = static_cast<OutputT>(unscaled * scale);
    }
    output_storer.store(vector_idx, segment_size);
  }
}

template <int nvec, bool ComputeScaleGrad, typename GradT, typename InputT, typename ScaleT,
          typename OutputT, typename GradScaleT, typename ParamOP,
          float (*ActOP)(float, const ParamOP &), float (*DActOP)(float, const ParamOP &)>
__global__ void __launch_bounds__(kReductionThreads, 4)
    scaled_gated_backward_kernel(const GradT *__restrict__ grad_output,
                                 const InputT *__restrict__ input,
                                 const ScaleT *__restrict__ act_scales,
                                 OutputT *__restrict__ grad_input,
                                 GradScaleT *__restrict__ grad_act_scales,
                                 const int32_t *__restrict__ num_rows, const size_t rows,
                                 const size_t hidden, const size_t segment_size,
                                 const size_t num_segments, const size_t num_vectors_per_segment,
                                 const ParamOP param) {
  __shared__ float smem[kReductionWarps];
  const size_t row = blockIdx.x;
  (void)rows;
  if (num_rows != nullptr && static_cast<int64_t>(row) >= static_cast<int64_t>(*num_rows)) {
    // Padding row: nothing reads its grad_input, the scale grad must not be garbage.
    if constexpr (ComputeScaleGrad) {
      if (threadIdx.x == 0) grad_act_scales[row] = static_cast<GradScaleT>(0.0f);
    }
    return;
  }
  float scale_grad = 0.0f;
  const float scale = static_cast<float>(act_scales[row]);

  const size_t row_vectors = num_segments * num_vectors_per_segment;
  for (size_t row_vector_idx = threadIdx.x; row_vector_idx < row_vectors;
       row_vector_idx += blockDim.x) {
    const size_t segment = row_vector_idx / num_vectors_per_segment;
    const size_t vector_idx = row_vector_idx % num_vectors_per_segment;
    const size_t input_segment_offset = row * hidden * 2 + segment * segment_size * 2;
    const size_t output_segment_offset = row * hidden + segment * segment_size;
    VectorizedLoader<GradT, nvec, true> grad_loader(grad_output + output_segment_offset,
                                                    segment_size);
    VectorizedLoader<InputT, nvec, true> act_loader(input + input_segment_offset, segment_size);
    VectorizedLoader<InputT, nvec, true> gate_loader(input + input_segment_offset + segment_size,
                                                     segment_size);
    VectorizedStorer<OutputT, nvec, true> act_storer(grad_input + input_segment_offset,
                                                     segment_size);
    VectorizedStorer<OutputT, nvec, true> gate_storer(
        grad_input + input_segment_offset + segment_size, segment_size);

    grad_loader.load(vector_idx, segment_size);
    act_loader.load(vector_idx, segment_size);
    gate_loader.load(vector_idx, segment_size);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      float dact = 0.0f;
      float dgate = 0.0f;
      float unscaled = 0.0f;
      gated_backward_values<ParamOP, ActOP, DActOP>(
          static_cast<float>(act_loader.separate()[lane]),
          static_cast<float>(gate_loader.separate()[lane]), param, &dact, &dgate, &unscaled);
      const float grad = static_cast<float>(grad_loader.separate()[lane]);
      if constexpr (ComputeScaleGrad) {
        scale_grad += grad * unscaled;
      }

      const float scaled_grad = grad * scale;
      act_storer.separate()[lane] = static_cast<OutputT>(scaled_grad * dact);
      gate_storer.separate()[lane] = static_cast<OutputT>(scaled_grad * dgate);
    }
    act_storer.store(vector_idx, segment_size);
    gate_storer.store(vector_idx, segment_size);
  }

  if constexpr (ComputeScaleGrad) {
    scale_grad = block_reduce_sum(scale_grad, smem);
    if (threadIdx.x == 0) {
      grad_act_scales[row] = static_cast<GradScaleT>(scale_grad);
    }
  }
}

// ---------------------------------------------------------------------------
// Unary kernels
// ---------------------------------------------------------------------------

template <int nvec, typename InputT, typename ScaleT, typename OutputT, typename ParamOP,
          float (*ActOP)(float, const ParamOP &)>
__global__ void __launch_bounds__(kThreads, 4)
    scaled_unary_forward_kernel(const InputT *__restrict__ input,
                                const ScaleT *__restrict__ act_scales, OutputT *__restrict__ output,
                                const size_t rows, const size_t hidden,
                                const size_t num_vectors_per_row, const ParamOP param) {
  const size_t total_vectors = rows * num_vectors_per_row;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total_vectors;
       tid += gridDim.x * blockDim.x) {
    const size_t vector_idx = tid % num_vectors_per_row;
    const size_t row = tid / num_vectors_per_row;
    VectorizedLoader<InputT, nvec, true> input_loader(input + row * hidden, hidden);
    VectorizedStorer<OutputT, nvec, true> output_storer(output + row * hidden, hidden);
    input_loader.load(vector_idx, hidden);
    const float scale = static_cast<float>(act_scales[row]);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      const float unscaled = ActOP(static_cast<float>(input_loader.separate()[lane]), param);
      output_storer.separate()[lane] = static_cast<OutputT>(unscaled * scale);
    }
    output_storer.store(vector_idx, hidden);
  }
}

template <int nvec, bool ComputeScaleGrad, typename GradT, typename InputT, typename ScaleT,
          typename OutputT, typename GradScaleT, typename ParamOP,
          float (*ActOP)(float, const ParamOP &), float (*DActOP)(float, const ParamOP &)>
__global__ void __launch_bounds__(kReductionThreads, 4)
    scaled_unary_backward_kernel(const GradT *__restrict__ grad_output,
                                 const InputT *__restrict__ input,
                                 const ScaleT *__restrict__ act_scales,
                                 OutputT *__restrict__ grad_input,
                                 GradScaleT *__restrict__ grad_act_scales, const size_t rows,
                                 const size_t hidden, const size_t num_vectors_per_row,
                                 const ParamOP param) {
  __shared__ float smem[kReductionWarps];
  const size_t row = blockIdx.x;
  (void)rows;
  float scale_grad = 0.0f;
  const float scale = static_cast<float>(act_scales[row]);

  VectorizedLoader<GradT, nvec, true> grad_loader(grad_output + row * hidden, hidden);
  VectorizedLoader<InputT, nvec, true> input_loader(input + row * hidden, hidden);
  VectorizedStorer<OutputT, nvec, true> grad_input_storer(grad_input + row * hidden, hidden);
  for (size_t vector_idx = threadIdx.x; vector_idx < num_vectors_per_row;
       vector_idx += blockDim.x) {
    grad_loader.load(vector_idx, hidden);
    input_loader.load(vector_idx, hidden);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      const float x = static_cast<float>(input_loader.separate()[lane]);
      const float grad = static_cast<float>(grad_loader.separate()[lane]);
      if constexpr (ComputeScaleGrad) {
        scale_grad += grad * ActOP(x, param);
      }
      grad_input_storer.separate()[lane] = static_cast<OutputT>(grad * scale * DActOP(x, param));
    }
    grad_input_storer.store(vector_idx, hidden);
  }

  if constexpr (ComputeScaleGrad) {
    scale_grad = block_reduce_sum(scale_grad, smem);
    if (threadIdx.x == 0) {
      grad_act_scales[row] = static_cast<GradScaleT>(scale_grad);
    }
  }
}

// ---------------------------------------------------------------------------
// Tensor checks
// ---------------------------------------------------------------------------

void check_gated_forward_tensors(const Tensor *input, const Tensor *act_scales,
                                 const Tensor *output, const int64_t glu_interleave_size,
                                 const char *api_name, size_t *rows, size_t *hidden) {
  const auto input_dims = input->flat_2d_dims();
  const auto output_dims = output->flat_2d_dims();
  NVTE_CHECK(input_dims[0] == output_dims[0], api_name, ": input/output row mismatch.");
  NVTE_CHECK(input_dims[1] == output_dims[1] * 2, api_name,
             ": gated input last dimension must be twice output last dimension.");
  NVTE_CHECK(glu_interleave_size >= 0, api_name, ": glu_interleave_size must be non-negative.");
  if (glu_interleave_size > 0) {
    NVTE_CHECK(glu_interleave_size % 32 == 0, api_name,
               ": nonzero glu_interleave_size must be a multiple of 32.");
    NVTE_CHECK(output_dims[1] % static_cast<size_t>(glu_interleave_size) == 0, api_name,
               ": output last dimension must be divisible by glu_interleave_size.");
  }
  NVTE_CHECK(act_scales->numel() == input_dims[0], api_name,
             ": act_scales must have one value per row.");
  *rows = input_dims[0];
  *hidden = output_dims[1];
}

void check_gated_backward_tensors(const Tensor *grad_output, const Tensor *input,
                                  const Tensor *act_scales, const Tensor *grad_input,
                                  const Tensor *grad_act_scales, const int64_t glu_interleave_size,
                                  const char *api_name, size_t *rows, size_t *hidden) {
  const auto grad_dims = grad_output->flat_2d_dims();
  const auto input_dims = input->flat_2d_dims();
  const auto grad_input_dims = grad_input->flat_2d_dims();
  NVTE_CHECK(grad_dims[0] == input_dims[0] && input_dims[0] == grad_input_dims[0], api_name,
             ": input/grad row mismatch.");
  NVTE_CHECK(input_dims[1] == grad_dims[1] * 2 && grad_input_dims[1] == input_dims[1], api_name,
             ": gated backward dimensions are inconsistent.");
  NVTE_CHECK(glu_interleave_size >= 0, api_name, ": glu_interleave_size must be non-negative.");
  if (glu_interleave_size > 0) {
    NVTE_CHECK(glu_interleave_size % 32 == 0, api_name,
               ": nonzero glu_interleave_size must be a multiple of 32.");
    NVTE_CHECK(grad_dims[1] % static_cast<size_t>(glu_interleave_size) == 0, api_name,
               ": grad last dimension must be divisible by glu_interleave_size.");
  }
  NVTE_CHECK(act_scales->numel() == input_dims[0], api_name,
             ": act_scales must have one value per row.");
  if (grad_act_scales != nullptr) {
    NVTE_CHECK(grad_act_scales->numel() == input_dims[0], api_name,
               ": grad_act_scales must have one value per row.");
  }
  *rows = input_dims[0];
  *hidden = grad_dims[1];
}

void check_unary_forward_tensors(const Tensor *input, const Tensor *act_scales,
                                 const Tensor *output, const char *api_name, size_t *rows,
                                 size_t *hidden) {
  const auto input_dims = input->flat_2d_dims();
  const auto output_dims = output->flat_2d_dims();
  NVTE_CHECK(input_dims[0] == output_dims[0] && input_dims[1] == output_dims[1], api_name,
             ": input/output shapes must match.");
  NVTE_CHECK(act_scales->numel() == input_dims[0], api_name,
             ": act_scales must have one value per row.");
  *rows = input_dims[0];
  *hidden = output_dims[1];
}

void check_unary_backward_tensors(const Tensor *grad_output, const Tensor *input,
                                  const Tensor *act_scales, const Tensor *grad_input,
                                  const Tensor *grad_act_scales, const char *api_name, size_t *rows,
                                  size_t *hidden) {
  const auto grad_dims = grad_output->flat_2d_dims();
  const auto input_dims = input->flat_2d_dims();
  const auto grad_input_dims = grad_input->flat_2d_dims();
  NVTE_CHECK(grad_dims[0] == input_dims[0] && input_dims[0] == grad_input_dims[0], api_name,
             ": input/grad row mismatch.");
  NVTE_CHECK(grad_dims[1] == input_dims[1] && input_dims[1] == grad_input_dims[1], api_name,
             ": unary backward dimensions are inconsistent.");
  NVTE_CHECK(act_scales->numel() == input_dims[0], api_name,
             ": act_scales must have one value per row.");
  if (grad_act_scales != nullptr) {
    NVTE_CHECK(grad_act_scales->numel() == input_dims[0], api_name,
               ": grad_act_scales must have one value per row.");
  }
  *rows = input_dims[0];
  *hidden = grad_dims[1];
}

}  // namespace

// ---------------------------------------------------------------------------
// Launch implementations
// ---------------------------------------------------------------------------

using namespace detail::scaled_activation;

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
void launch_scaled_gated_forward(const NVTETensor nvte_input, const NVTETensor nvte_act_scales,
                                 NVTETensor nvte_output, ParamOP param, int64_t glu_interleave_size,
                                 cudaStream_t stream, const char *api_name,
                                 const NVTETensor nvte_num_rows) {
  const Tensor *input = convertNVTETensorCheck(nvte_input);
  const Tensor *act_scales = convertNVTETensorCheck(nvte_act_scales);
  Tensor *output = convertNVTETensorCheck(nvte_output);
  size_t rows = 0;
  size_t hidden = 0;
  check_gated_forward_tensors(input, act_scales, output, glu_interleave_size, api_name, &rows,
                              &hidden);
  if (rows == 0 || hidden == 0) return;
  const int32_t *num_rows_ptr = num_rows_pointer(nvte_num_rows, api_name);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input->data.dtype, InputT, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(act_scales->data.dtype, ScaleT, {
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(output->data.dtype, OutputT, {
        constexpr int nvec = 32 / static_cast<int>(std::max(sizeof(InputT), sizeof(OutputT)));
        const auto input_ptr = reinterpret_cast<const InputT *>(input->data.dptr);
        const auto scale_ptr = reinterpret_cast<const ScaleT *>(act_scales->data.dptr);
        auto output_ptr = reinterpret_cast<OutputT *>(output->data.dptr);
        const size_t segment_size =
            glu_interleave_size > 0 ? static_cast<size_t>(glu_interleave_size) : hidden;
        const size_t num_segments = glu_interleave_size > 0 ? hidden / segment_size : 1;
        const auto align = row_vector_alignment(segment_size, nvec, input_ptr,
                                                input_ptr + segment_size, output_ptr);
        const bool use_vector = align == Alignment::SAME_ALIGNED;
        const size_t num_vectors =
            use_vector ? get_num_aligned_elements(input_ptr, segment_size, nvec, sizeof(InputT))
                       : segment_size;
        const int blocks = static_cast<int>(std::min<size_t>(
            DIVUP(rows * num_segments * num_vectors, static_cast<size_t>(kThreads)), 65535));
        if (use_vector) {
          scaled_gated_forward_kernel<nvec, InputT, ScaleT, OutputT, ParamOP, ActOP>
              <<<blocks, kThreads, 0, stream>>>(input_ptr, scale_ptr, output_ptr, num_rows_ptr,
                                                rows, hidden, segment_size, num_segments,
                                                num_vectors, param);
        } else {
          scaled_gated_forward_kernel<1, InputT, ScaleT, OutputT, ParamOP, ActOP>
              <<<blocks, kThreads, 0, stream>>>(input_ptr, scale_ptr, output_ptr, num_rows_ptr,
                                                rows, hidden, segment_size, num_segments,
                                                segment_size, param);
        }
      });
    });
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void launch_scaled_gated_backward(const NVTETensor nvte_grad_output, const NVTETensor nvte_input,
                                  const NVTETensor nvte_act_scales, NVTETensor nvte_grad_input,
                                  NVTETensor nvte_grad_act_scales, ParamOP param,
                                  int64_t glu_interleave_size, cudaStream_t stream,
                                  const char *api_name, const NVTETensor nvte_num_rows) {
  const Tensor *grad_output = convertNVTETensorCheck(nvte_grad_output);
  const Tensor *input = convertNVTETensorCheck(nvte_input);
  const Tensor *act_scales = convertNVTETensorCheck(nvte_act_scales);
  Tensor *grad_input = convertNVTETensorCheck(nvte_grad_input);
  Tensor *grad_act_scales =
      nvte_grad_act_scales == nullptr ? nullptr : convertNVTETensorCheck(nvte_grad_act_scales);
  size_t rows = 0;
  size_t hidden = 0;
  check_gated_backward_tensors(grad_output, input, act_scales, grad_input, grad_act_scales,
                               glu_interleave_size, api_name, &rows, &hidden);
  if (rows == 0 || hidden == 0) return;
  const int32_t *num_rows_ptr = num_rows_pointer(nvte_num_rows, api_name);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_output->data.dtype, GradT, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input->data.dtype, InputT, {
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(act_scales->data.dtype, ScaleT, {
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_input->data.dtype, OutputT, {
          constexpr int nvec =
              32 / static_cast<int>(std::max({sizeof(GradT), sizeof(InputT), sizeof(OutputT)}));
          const auto grad_ptr = reinterpret_cast<const GradT *>(grad_output->data.dptr);
          const auto input_ptr = reinterpret_cast<const InputT *>(input->data.dptr);
          const auto scale_ptr = reinterpret_cast<const ScaleT *>(act_scales->data.dptr);
          auto grad_input_ptr = reinterpret_cast<OutputT *>(grad_input->data.dptr);
          const size_t segment_size =
              glu_interleave_size > 0 ? static_cast<size_t>(glu_interleave_size) : hidden;
          const size_t num_segments = glu_interleave_size > 0 ? hidden / segment_size : 1;
          const auto align = row_vector_alignment(segment_size, nvec, grad_ptr, input_ptr,
                                                  input_ptr + segment_size, grad_input_ptr,
                                                  grad_input_ptr + segment_size);
          const bool use_vector = align == Alignment::SAME_ALIGNED;
          const size_t num_vectors =
              use_vector ? get_num_aligned_elements(input_ptr, segment_size, nvec, sizeof(InputT))
                         : segment_size;
          const int reduction_threads = choose_reduction_threads(num_segments * num_vectors);
          if (grad_act_scales == nullptr) {
            if (use_vector) {
              scaled_gated_backward_kernel<nvec, false, GradT, InputT, ScaleT, OutputT, OutputT,
                                           ParamOP, ActOP, DActOP>
                  <<<static_cast<int>(rows), reduction_threads, 0, stream>>>(
                      grad_ptr, input_ptr, scale_ptr, grad_input_ptr, nullptr, num_rows_ptr, rows,
                      hidden, segment_size, num_segments, num_vectors, param);
            } else {
              scaled_gated_backward_kernel<1, false, GradT, InputT, ScaleT, OutputT, OutputT,
                                           ParamOP, ActOP, DActOP>
                  <<<static_cast<int>(rows), reduction_threads, 0, stream>>>(
                      grad_ptr, input_ptr, scale_ptr, grad_input_ptr, nullptr, num_rows_ptr, rows,
                      hidden, segment_size, num_segments, segment_size, param);
            }
          } else {
            TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_act_scales->data.dtype, GradScaleT, {
              auto grad_act_scales_ptr = reinterpret_cast<GradScaleT *>(grad_act_scales->data.dptr);
              if (use_vector) {
                scaled_gated_backward_kernel<nvec, true, GradT, InputT, ScaleT, OutputT, GradScaleT,
                                             ParamOP, ActOP, DActOP>
                    <<<static_cast<int>(rows), reduction_threads, 0, stream>>>(
                        grad_ptr, input_ptr, scale_ptr, grad_input_ptr, grad_act_scales_ptr,
                        num_rows_ptr, rows, hidden, segment_size, num_segments, num_vectors, param);
              } else {
                scaled_gated_backward_kernel<1, true, GradT, InputT, ScaleT, OutputT, GradScaleT,
                                             ParamOP, ActOP, DActOP>
                    <<<static_cast<int>(rows), reduction_threads, 0, stream>>>(
                        grad_ptr, input_ptr, scale_ptr, grad_input_ptr, grad_act_scales_ptr,
                        num_rows_ptr, rows, hidden, segment_size, num_segments, segment_size,
                        param);
              }
            });
          }
        });
      });
    });
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
void launch_scaled_unary_forward(const NVTETensor nvte_input, const NVTETensor nvte_act_scales,
                                 NVTETensor nvte_output, ParamOP param, cudaStream_t stream,
                                 const char *api_name) {
  const Tensor *input = convertNVTETensorCheck(nvte_input);
  const Tensor *act_scales = convertNVTETensorCheck(nvte_act_scales);
  Tensor *output = convertNVTETensorCheck(nvte_output);
  size_t rows = 0;
  size_t hidden = 0;
  check_unary_forward_tensors(input, act_scales, output, api_name, &rows, &hidden);
  if (rows == 0 || hidden == 0) return;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input->data.dtype, InputT, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(act_scales->data.dtype, ScaleT, {
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(output->data.dtype, OutputT, {
        constexpr int nvec = 32 / static_cast<int>(std::max(sizeof(InputT), sizeof(OutputT)));
        const auto input_ptr = reinterpret_cast<const InputT *>(input->data.dptr);
        const auto scale_ptr = reinterpret_cast<const ScaleT *>(act_scales->data.dptr);
        auto output_ptr = reinterpret_cast<OutputT *>(output->data.dptr);
        const auto align = row_vector_alignment(hidden, nvec, input_ptr, output_ptr);
        const bool use_vector = align == Alignment::SAME_ALIGNED;
        const size_t num_vectors =
            use_vector ? get_num_aligned_elements(input_ptr, hidden, nvec, sizeof(InputT)) : hidden;
        const int blocks = static_cast<int>(
            std::min<size_t>(DIVUP(rows * num_vectors, static_cast<size_t>(kThreads)), 65535));
        if (use_vector) {
          scaled_unary_forward_kernel<nvec, InputT, ScaleT, OutputT, ParamOP, ActOP>
              <<<blocks, kThreads, 0, stream>>>(input_ptr, scale_ptr, output_ptr, rows, hidden,
                                                num_vectors, param);
        } else {
          scaled_unary_forward_kernel<1, InputT, ScaleT, OutputT, ParamOP, ActOP>
              <<<blocks, kThreads, 0, stream>>>(input_ptr, scale_ptr, output_ptr, rows, hidden,
                                                hidden, param);
        }
      });
    });
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void launch_scaled_unary_backward(const NVTETensor nvte_grad_output, const NVTETensor nvte_input,
                                  const NVTETensor nvte_act_scales, NVTETensor nvte_grad_input,
                                  NVTETensor nvte_grad_act_scales, ParamOP param,
                                  cudaStream_t stream, const char *api_name) {
  const Tensor *grad_output = convertNVTETensorCheck(nvte_grad_output);
  const Tensor *input = convertNVTETensorCheck(nvte_input);
  const Tensor *act_scales = convertNVTETensorCheck(nvte_act_scales);
  Tensor *grad_input = convertNVTETensorCheck(nvte_grad_input);
  Tensor *grad_act_scales =
      nvte_grad_act_scales == nullptr ? nullptr : convertNVTETensorCheck(nvte_grad_act_scales);
  size_t rows = 0;
  size_t hidden = 0;
  check_unary_backward_tensors(grad_output, input, act_scales, grad_input, grad_act_scales,
                               api_name, &rows, &hidden);
  if (rows == 0 || hidden == 0) return;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_output->data.dtype, GradT, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input->data.dtype, InputT, {
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(act_scales->data.dtype, ScaleT, {
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_input->data.dtype, OutputT, {
          constexpr int nvec =
              32 / static_cast<int>(std::max({sizeof(GradT), sizeof(InputT), sizeof(OutputT)}));
          const auto grad_ptr = reinterpret_cast<const GradT *>(grad_output->data.dptr);
          const auto input_ptr = reinterpret_cast<const InputT *>(input->data.dptr);
          const auto scale_ptr = reinterpret_cast<const ScaleT *>(act_scales->data.dptr);
          auto grad_input_ptr = reinterpret_cast<OutputT *>(grad_input->data.dptr);
          const auto align =
              row_vector_alignment(hidden, nvec, grad_ptr, input_ptr, grad_input_ptr);
          const bool use_vector = align == Alignment::SAME_ALIGNED;
          const size_t num_vectors =
              use_vector ? get_num_aligned_elements(input_ptr, hidden, nvec, sizeof(InputT))
                         : hidden;
          const int reduction_threads = choose_reduction_threads(num_vectors);
          if (grad_act_scales == nullptr) {
            if (use_vector) {
              scaled_unary_backward_kernel<nvec, false, GradT, InputT, ScaleT, OutputT, OutputT,
                                           ParamOP, ActOP, DActOP>
                  <<<static_cast<int>(rows), reduction_threads, 0, stream>>>(
                      grad_ptr, input_ptr, scale_ptr, grad_input_ptr, nullptr, rows, hidden,
                      num_vectors, param);
            } else {
              scaled_unary_backward_kernel<1, false, GradT, InputT, ScaleT, OutputT, OutputT,
                                           ParamOP, ActOP, DActOP>
                  <<<static_cast<int>(rows), reduction_threads, 0, stream>>>(
                      grad_ptr, input_ptr, scale_ptr, grad_input_ptr, nullptr, rows, hidden, hidden,
                      param);
            }
          } else {
            TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_act_scales->data.dtype, GradScaleT, {
              auto grad_act_scales_ptr = reinterpret_cast<GradScaleT *>(grad_act_scales->data.dptr);
              if (use_vector) {
                scaled_unary_backward_kernel<nvec, true, GradT, InputT, ScaleT, OutputT, GradScaleT,
                                             ParamOP, ActOP, DActOP>
                    <<<static_cast<int>(rows), reduction_threads, 0, stream>>>(
                        grad_ptr, input_ptr, scale_ptr, grad_input_ptr, grad_act_scales_ptr, rows,
                        hidden, num_vectors, param);
              } else {
                scaled_unary_backward_kernel<1, true, GradT, InputT, ScaleT, OutputT, GradScaleT,
                                             ParamOP, ActOP, DActOP>
                    <<<static_cast<int>(rows), reduction_threads, 0, stream>>>(
                        grad_ptr, input_ptr, scale_ptr, grad_input_ptr, grad_act_scales_ptr, rows,
                        hidden, hidden, param);
              }
            });
          }
        });
      });
    });
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

template void launch_scaled_gated_forward<Empty, silu<fp32, fp32>>(
    const NVTETensor, const NVTETensor, NVTETensor, Empty, int64_t, cudaStream_t, const char *,
    const NVTETensor);
template void launch_scaled_gated_backward<Empty, silu<fp32, fp32>, dsilu<fp32, fp32>>(
    const NVTETensor, const NVTETensor, const NVTETensor, NVTETensor, NVTETensor, Empty, int64_t,
    cudaStream_t, const char *, const NVTETensor);

template void launch_scaled_gated_forward<ClampedSwiGLUParam, clamped_silu<fp32, fp32>>(
    const NVTETensor, const NVTETensor, NVTETensor, ClampedSwiGLUParam, int64_t, cudaStream_t,
    const char *, const NVTETensor);
template void launch_scaled_gated_backward<ClampedSwiGLUParam, clamped_silu<fp32, fp32>,
                                           clamped_dsilu<fp32, fp32>>(
    const NVTETensor, const NVTETensor, const NVTETensor, NVTETensor, NVTETensor,
    ClampedSwiGLUParam, int64_t, cudaStream_t, const char *, const NVTETensor);

template void launch_scaled_gated_forward<SiTUGLUParam, situ_gate<fp32, fp32>>(
    const NVTETensor, const NVTETensor, NVTETensor, SiTUGLUParam, int64_t, cudaStream_t,
    const char *, const NVTETensor);
template void
launch_scaled_gated_backward<SiTUGLUParam, situ_gate<fp32, fp32>, dsitu_gate<fp32, fp32>>(
    const NVTETensor, const NVTETensor, const NVTETensor, NVTETensor, NVTETensor, SiTUGLUParam,
    int64_t, cudaStream_t, const char *, const NVTETensor);

template void launch_scaled_unary_forward<Empty, srelu<fp32, fp32>>(const NVTETensor,
                                                                    const NVTETensor, NVTETensor,
                                                                    Empty, cudaStream_t,
                                                                    const char *);
template void launch_scaled_unary_backward<Empty, srelu<fp32, fp32>, dsrelu<fp32, fp32>>(
    const NVTETensor, const NVTETensor, const NVTETensor, NVTETensor, NVTETensor, Empty,
    cudaStream_t, const char *);

}  // namespace transformer_engine
