/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/activation.h>

#include <cfloat>

#include "../common.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "math.h"
#include "ptx.cuh"

namespace transformer_engine {

constexpr size_t ALIGNMENT_SIZE = 128;
constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 512;
constexpr size_t THREADS_PER_CHUNK_X = CHUNK_DIM_X;
constexpr size_t THREADS_PER_CHUNK_Y = THREADS_PER_CHUNK / THREADS_PER_CHUNK_X;  // 4 = 512 / 128
constexpr size_t BUFFERS_NUM = 2;
constexpr size_t BUFFER_DIM_Y = 32;
constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;  // 128
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;  // 32
constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;  // 128

constexpr size_t BUFFER_STAGES_NUM = BUFFER_DIM_Y / THREADS_PER_CHUNK_Y;  //  8 =  32 / 4
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;                 //   4 = 128 / 32
static_assert(ITERATIONS >= 1);

__device__ inline float sigmoidf(const float x) { return __frcp_rn(1.0f + __expf(-x)); }

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    cast_fp8_dgated_kernel(const __grid_constant__ CUtensorMap tensor_map_grad,
                           const __grid_constant__ CUtensorMap tensor_map_gated_input,
                           const __grid_constant__ CUtensorMap tensor_map_output,
                           float *const amax_ptr, float *const scale_inv_ptr,
                           const float *const scale_ptr, const size_t rows, const size_t cols) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  const int chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int chunk_offset_X = blockIdx.x * CHUNK_DIM_X;

  const int tid_Y = threadIdx.x / THREADS_PER_CHUNK_X;
  const int tid_X = threadIdx.x % THREADS_PER_CHUNK_X;

  const int thread_offset_Y = tid_Y;
  const int thread_offset_X = tid_X;

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  extern __shared__ char dshmem_unaligned[];
  const uint64_t dshmem_unaligned_as_uint = reinterpret_cast<uint64_t>(dshmem_unaligned);
  const uint64_t dshmem_aligned_as_uint =
      DIVUP(dshmem_unaligned_as_uint, static_cast<uint64_t>(ALIGNMENT_SIZE)) * ALIGNMENT_SIZE;
  char *dshmem = reinterpret_cast<char *>(dshmem_aligned_as_uint);

  const size_t buff_elems = SHMEM_DIM_Y * SHMEM_DIM_X;
  const size_t buff_elems_total = BUFFERS_NUM * buff_elems;
  const size_t buff_size_aligned_in =
      DIVUP(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
  const size_t buff_size_aligned_out =
      DIVUP(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;

  const size_t grad_mem = buff_size_aligned_in;

  const size_t in_act_mem = buff_size_aligned_in;
  const size_t in_gate_mem = buff_size_aligned_in;
  const size_t in_mem = in_act_mem + in_gate_mem;

  const size_t out_act_mem = buff_size_aligned_out;
  const size_t out_gate_mem = buff_size_aligned_out;
  const size_t out_mem = out_act_mem + out_gate_mem;

  // const size_t in_transaction_size = grad_mem + in_mem;
  const size_t in_transaction_size = 3 * buff_elems * sizeof(IType);

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_grad_sh = reinterpret_cast<IType *>(dshmem);
  IType *in_act_sh = reinterpret_cast<IType *>(dshmem + grad_mem);
  IType *in_gate_sh = reinterpret_cast<IType *>(dshmem + grad_mem + in_act_mem);
  OType *out_act_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem);
  OType *out_gate_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_act_mem);
  // uint64_t *mbar = reinterpret_cast<uint64_t *>(dshmem + grad_mem + in_mem + out_mem);

  const uint64_t *TMAP_grad_in = reinterpret_cast<const uint64_t *>(&tensor_map_grad);
  const uint64_t *TMAP_gate_in = reinterpret_cast<const uint64_t *>(&tensor_map_gated_input);
  const uint64_t *TMAP_output = reinterpret_cast<const uint64_t *>(&tensor_map_output);

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  if (is_master_thread) {
// Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
#pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      ptx::mbarrier_init(&mbar[it], THREADS_PER_CHUNK);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int parity = 0;

  // Prefetch data of the first stage
  if (is_master_thread) {
    // Initiate bulk tensor copy
    // Grad
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(&in_grad_sh[0]),
                                                  TMAP_grad_in, chunk_offset_X, chunk_offset_Y,
                                                  &mbar[0]);

    // Act
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(&in_act_sh[0]),
                                                  TMAP_gate_in, chunk_offset_X, chunk_offset_Y,
                                                  &mbar[0]);

    // Gate
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(&in_gate_sh[0]),
                                                  TMAP_gate_in, chunk_offset_X + cols,
                                                  chunk_offset_Y, &mbar[0]);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(&mbar[0], in_transaction_size);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(&mbar[0]);
  }

#pragma unroll
  for (int it = 0; it < ITERATIONS; ++it) {
    const int buff = it % BUFFERS_NUM;
    const int next_it = it + 1;
    if (next_it < ITERATIONS) {
      if (is_master_thread) {
        const int next_buff = next_it % BUFFERS_NUM;
        const int chunk_it_offset_y = chunk_offset_Y + next_it * BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        // Initiate bulk tensor copy
        // Grad
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_grad_sh[next_buff * buff_elems]), TMAP_grad_in,
            chunk_it_offset_x, chunk_it_offset_y, &mbar[next_it]);
        // Act
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_act_sh[next_buff * buff_elems]), TMAP_gate_in,
            chunk_it_offset_x, chunk_it_offset_y, &mbar[next_it]);
        // Gate
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_gate_sh[next_buff * buff_elems]), TMAP_gate_in,
            chunk_it_offset_x + cols, chunk_it_offset_y, &mbar[next_it]);

        // Arrive on the barrier and tell how many bytes are expected to come in.
        ptx::mbarrier_arrive_expect_tx(&mbar[next_it], in_transaction_size);
      } else {
        // Other threads just arrive
        ptx::mbarrier_arrive(&mbar[next_it]);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[it], parity);

    IType *in_grad_sh_curr = in_grad_sh + buff * buff_elems;
    IType *in_act_sh_curr = in_act_sh + buff * buff_elems;
    IType *in_gate_sh_curr = in_gate_sh + buff * buff_elems;
    OType *out_act_sh_curr = out_act_sh + buff * buff_elems;
    OType *out_gate_sh_curr = out_gate_sh + buff * buff_elems;

#pragma unroll
    for (int stage = 0; stage < BUFFER_STAGES_NUM; ++stage) {
      const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y;
      const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
      const int shmem_offset_x = thread_offset_X;
      const int shmem_idx = shmem_offset_y * SHMEM_DIM_X + shmem_offset_x;

      float grad_elt = static_cast<float>(in_grad_sh_curr[shmem_idx]);
      float act_elt = static_cast<float>(in_act_sh_curr[shmem_idx]);
      float gate_elt = static_cast<float>(in_gate_sh_curr[shmem_idx]);

      const float x = act_elt;
      float act_x;
      float dact_x;

      if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
        const float s = sigmoidf(x);
        act_x = x * s;
        dact_x = x * s * (1 - s) + s;
      } else {
        act_x = ActOP(x);
        dact_x = DActOP(x);
      }

      float after_dact = dact_x * grad_elt * gate_elt;
      float after_dgate = act_x * grad_elt;

      out_act_sh_curr[shmem_idx] = static_cast<OType>(scale * after_dact);
      out_gate_sh_curr[shmem_idx] = static_cast<OType>(scale * after_dgate);

      amax = fmaxf(amax, fabsf(after_dact));
      amax = fmaxf(amax, fabsf(after_dgate));
    }

    // Wait for shared memory writes to be visible to TMA engine (cross-proxy fence)
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const int chunk_it_offset_y = chunk_offset_Y + it * BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;

      // dGeLU
      ptx::cp_async_bulk_tensor_2d_shared_to_global(TMAP_output, chunk_it_offset_x,
                                                    chunk_it_offset_y,
                                                    reinterpret_cast<uint64_t *>(out_act_sh_curr));

      // dGate
      ptx::cp_async_bulk_tensor_2d_shared_to_global(TMAP_output, chunk_it_offset_x + cols,
                                                    chunk_it_offset_y,
                                                    reinterpret_cast<uint64_t *>(out_gate_sh_curr));

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<BUFFERS_NUM - 1>();
    }
  }
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    amax = reduce_max<THREADS_PER_CHUNK / THREADS_PER_WARP>(amax, warp_id);
    // Update the global amax
    if (is_master_thread) {
      atomicMaxFloat(amax_ptr, amax);
    }
  }

  // Update scale-inverse
  if (is_master_thread && blockIdx.x == 0 && (scale_inv_ptr != nullptr)) {
    reciprocal<float>(scale_inv_ptr, scale);
  }

  // Destroy the barriers. This invalidates the memory region of the barrier.
  // If further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      ptx::mbarrier_invalid(&mbar[it]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType,
          size_t SCALE_DIM_Y, size_t SCALE_DIM_X>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    cast_mxfp8_dgated_kernel(const __grid_constant__ CUtensorMap tensor_map_grad,
                             const __grid_constant__ CUtensorMap tensor_map_gated_input,
                             const __grid_constant__ CUtensorMap tensor_map_output_rowwise,
                             const __grid_constant__ CUtensorMap tensor_map_output_colwise,
                             e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise,
                             float *const amax_ptr,
                             const size_t rows, const size_t cols,
                             const size_t scale_stride_rowwise, const size_t scale_stride_colwise) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;
  constexpr bool COMPUTE_IN_ROWWISE_SECTION = !USE_COLWISE_SCALING;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = CHUNK_DIM_Y;                //  128
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM_X;  //    4 = 128 / 32

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM_Y;  //    4 = 128 / 32
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = CHUNK_DIM_X;                //  128

  const int scales_rowwise_chunk_offset_Y = blockIdx.y * SCALES_ROWWISE_PER_CHUNK_Y;
  const int scales_rowwise_chunk_offset_X = blockIdx.x * SCALES_ROWWISE_PER_CHUNK_X;
  const int scales_colwise_chunk_offset_Y = blockIdx.y * SCALES_COLWISE_PER_CHUNK_Y;
  const int scales_colwise_chunk_offset_X = blockIdx.x * SCALES_COLWISE_PER_CHUNK_X;

  const int chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int chunk_offset_X = blockIdx.x * CHUNK_DIM_X;

  const int tid_Y = threadIdx.x / THREADS_PER_CHUNK_X;
  const int tid_X = threadIdx.x % THREADS_PER_CHUNK_X;

  const int thread_offset_Y = tid_Y;
  const int thread_offset_X = tid_X;

  float thread_amax = 0;

  extern __shared__ char dshmem_unaligned[];
  const uint64_t dshmem_unaligned_as_uint = reinterpret_cast<uint64_t>(dshmem_unaligned);
  const uint64_t dshmem_aligned_as_uint =
      DIVUP(dshmem_unaligned_as_uint, static_cast<uint64_t>(ALIGNMENT_SIZE)) * ALIGNMENT_SIZE;
  char *dshmem = reinterpret_cast<char *>(dshmem_aligned_as_uint);

  const size_t buff_elems = SHMEM_DIM_Y * SHMEM_DIM_X;
  const size_t buff_elems_total = BUFFERS_NUM * buff_elems;
  const size_t buff_size_aligned_in =
      DIVUP(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
  const size_t buff_size_aligned_out =
      DIVUP(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;

  const size_t grad_mem = buff_size_aligned_in;

  const size_t in_act_mem = buff_size_aligned_in;
  const size_t in_gate_mem = buff_size_aligned_in;
  const size_t in_mem = in_act_mem + in_gate_mem;

  const size_t out_act_mem = buff_size_aligned_out;
  const size_t out_gate_mem = buff_size_aligned_out;
  const size_t out_mem = out_act_mem + out_gate_mem;

  // const size_t in_transaction_size = grad_mem + in_mem;
  const size_t in_transaction_size = 3 * buff_elems * sizeof(IType);

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_grad_sh = reinterpret_cast<IType *>(dshmem);
  IType *in_act_sh = reinterpret_cast<IType *>(dshmem + grad_mem);
  IType *in_gate_sh = reinterpret_cast<IType *>(dshmem + grad_mem + in_act_mem);

  OType *out_act_rowwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem);
  OType *out_gate_rowwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_act_mem);

  OType *out_act_colwise_sh = out_act_rowwise_sh;
  OType *out_gate_colwise_sh = out_gate_rowwise_sh;

  if constexpr (USE_ROWWISE_SCALING && USE_COLWISE_SCALING) {
    out_act_colwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_mem);
    out_gate_colwise_sh =
        reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_mem + out_act_mem);
  }

  const uint64_t *TMAP_grad_in = reinterpret_cast<const uint64_t *>(&tensor_map_grad);
  const uint64_t *TMAP_gate_in = reinterpret_cast<const uint64_t *>(&tensor_map_gated_input);
  const uint64_t *TMAP_output_rowwise =
      reinterpret_cast<const uint64_t *>(&tensor_map_output_rowwise);
  const uint64_t *TMAP_output_colwise =
      reinterpret_cast<const uint64_t *>(&tensor_map_output_colwise);

  __shared__ float stage_amax_sh[THREADS_PER_CHUNK_Y][CHUNK_DIM_X];

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  const bool is_master_thread = (threadIdx.x == 0);

  if (is_master_thread) {
// Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
#pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      ptx::mbarrier_init(&mbar[it], THREADS_PER_CHUNK);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int parity = 0;

  // Prefetch data of the first stage
  if (is_master_thread) {
    // Initiate bulk tensor copy
    // Grad
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(&in_grad_sh[0]),
                                                  TMAP_grad_in, chunk_offset_X, chunk_offset_Y,
                                                  &mbar[0]);

    // Act
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(&in_act_sh[0]),
                                                  TMAP_gate_in, chunk_offset_X, chunk_offset_Y,
                                                  &mbar[0]);

    // Gate
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(&in_gate_sh[0]),
                                                  TMAP_gate_in, chunk_offset_X + cols,
                                                  chunk_offset_Y, &mbar[0]);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(&mbar[0], in_transaction_size);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(&mbar[0]);
  }

#pragma unroll
  for (int it = 0; it < ITERATIONS; ++it) {
    const int buff = it % BUFFERS_NUM;
    const int next_it = it + 1;
    if (next_it < ITERATIONS) {
      if (is_master_thread) {
        const int next_buff = next_it % BUFFERS_NUM;
        const int chunk_it_offset_y = chunk_offset_Y + next_it * BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        // Initiate bulk tensor copy
        // Grad
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_grad_sh[next_buff * buff_elems]), TMAP_grad_in,
            chunk_it_offset_x, chunk_it_offset_y, &mbar[next_it]);
        // Act
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_act_sh[next_buff * buff_elems]), TMAP_gate_in,
            chunk_it_offset_x, chunk_it_offset_y, &mbar[next_it]);
        // Gate
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_gate_sh[next_buff * buff_elems]), TMAP_gate_in,
            chunk_it_offset_x + cols, chunk_it_offset_y, &mbar[next_it]);

        // Arrive on the barrier and tell how many bytes are expected to come in.
        ptx::mbarrier_arrive_expect_tx(&mbar[next_it], in_transaction_size);
      } else {
        // Other threads just arrive
        ptx::mbarrier_arrive(&mbar[next_it]);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[it], parity);

    IType *in_grad_sh_curr = in_grad_sh + buff * buff_elems;
    IType *in_act_sh_curr = in_act_sh + buff * buff_elems;
    IType *in_gate_sh_curr = in_gate_sh + buff * buff_elems;
    OType *out_act_rowwise_sh_curr = out_act_rowwise_sh + buff * buff_elems;
    OType *out_gate_rowwise_sh_curr = out_gate_rowwise_sh + buff * buff_elems;
    OType *out_act_colwise_sh_curr = out_act_colwise_sh + buff * buff_elems;
    OType *out_gate_colwise_sh_curr = out_gate_colwise_sh + buff * buff_elems;

    // Assuming one iteration covers exactly 32 rows
    const int iteration_scale_colwise_offset_Y = scales_colwise_chunk_offset_Y + it;
    const int iteration_scale_rowwise_offset_Y = scales_rowwise_chunk_offset_Y + it * BUFFER_DIM_Y;

    float after_dact_reg[BUFFER_STAGES_NUM];
    float after_dgate_reg[BUFFER_STAGES_NUM];
    float thread_Y_mx_block_amax = 0.0f;

#pragma unroll
    for (int stage = 0; stage < BUFFER_STAGES_NUM; ++stage) {
      const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y;
      const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
      const int shmem_offset_x = thread_offset_X;
      const int shmem_idx = shmem_offset_y * SHMEM_DIM_X + shmem_offset_x;

      float grad_elt = static_cast<float>(in_grad_sh_curr[shmem_idx]);
      float act_elt = static_cast<float>(in_act_sh_curr[shmem_idx]);
      float gate_elt = static_cast<float>(in_gate_sh_curr[shmem_idx]);

      const float x = act_elt;
      float act_x;
      float dact_x;

      if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
        const float s = sigmoidf(x);
        act_x = x * s;
        dact_x = x * s * (1 - s) + s;
      } else {
        act_x = ActOP(x);
        dact_x = DActOP(x);
      }
      after_dact_reg[stage] = dact_x * grad_elt * gate_elt;
      after_dgate_reg[stage] = act_x * grad_elt;

      const float amax_gated_elem =
          fmaxf(fabsf(after_dact_reg[stage]), fabsf(after_dgate_reg[stage]));

      if constexpr (USE_ROWWISE_SCALING) {
        __builtin_assume(amax_gated_elem >= 0);
        __builtin_assume(thread_amax >= 0);
        thread_amax = fmaxf(thread_amax, amax_gated_elem);

        const float mx_block_X_amax = warp_reduce_max_broadcast(amax_gated_elem);
        const e8m0_t biased_exponent_X =
            float_to_e8m0(mx_block_X_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_reciprocal_X = exp2f_rcp(biased_exponent_X);

        out_act_rowwise_sh_curr[shmem_idx] =
            static_cast<OType>(scale_reciprocal_X * after_dact_reg[stage]);
        out_gate_rowwise_sh_curr[shmem_idx] =
            static_cast<OType>(scale_reciprocal_X * after_dgate_reg[stage]);

        // Only single thread writes the computed scaling factor
        if (tid_X % SCALE_DIM_X == 0) {
          const int global_scales_offset_Y =
              iteration_scale_rowwise_offset_Y + stage_offset_Y + thread_offset_Y;
          const int global_scales_offset_X = scales_rowwise_chunk_offset_X + tid_X / SCALE_DIM_X;
          const int scale_idx =
              global_scales_offset_Y * scale_stride_rowwise + global_scales_offset_X;
          scales_rowwise[scale_idx] = biased_exponent_X;
        }
      }

      if constexpr (USE_COLWISE_SCALING) {
        __builtin_assume(amax_gated_elem >= 0);
        __builtin_assume(thread_Y_mx_block_amax >= 0);
        thread_Y_mx_block_amax = fmaxf(thread_Y_mx_block_amax, amax_gated_elem);
      }
    }

    if constexpr (USE_COLWISE_SCALING) {
      // Colwise max reduction of the amax element
      if (tid_Y > 0) {
        stage_amax_sh[tid_Y][tid_X] = thread_Y_mx_block_amax;
      }
      __syncthreads();
      if (tid_Y == 0) {
#pragma unroll
        for (int y = 1; y < THREADS_PER_CHUNK_Y; ++y) {
          thread_Y_mx_block_amax = fmaxf(thread_Y_mx_block_amax, stage_amax_sh[y][tid_X]);
        }
        stage_amax_sh[0][tid_X] = thread_Y_mx_block_amax;  // write mx column-block amax
      }
      __syncthreads();

      const float mx_block_Y_amax = stage_amax_sh[0][tid_X];  // read the mx column-block amax

      // For the scaling along both dimensions, the thread amax is already computed in ROWWISE section
      if constexpr (!USE_ROWWISE_SCALING) {
        __builtin_assume(mx_block_Y_amax >= 0);
        __builtin_assume(thread_amax >= 0);
        thread_amax = fmaxf(thread_amax, mx_block_Y_amax);
      }

      const e8m0_t biased_exponent =
          float_to_e8m0(mx_block_Y_amax * Quantized_Limits<OType>::max_norm_rcp);
      const float scale_reciprocal = exp2f_rcp(biased_exponent);

      // Only single thread writes the computed scaling factor
      // Also assuming one iteration covers exactly 32 rows
      if (tid_Y == 0) {
        const int global_scales_offset_Y = iteration_scale_colwise_offset_Y;
        const int global_scales_offset_X = scales_colwise_chunk_offset_X + tid_X;
        const int scale_idx =
            global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
        scales_colwise[scale_idx] = biased_exponent;
      }

#pragma unroll
      for (int stage = 0; stage < BUFFER_STAGES_NUM; ++stage) {
        const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y;
        const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
        const int shmem_offset_x = thread_offset_X;
        const int shmem_idx = shmem_offset_y * SHMEM_DIM_X + shmem_offset_x;

        out_act_colwise_sh_curr[shmem_idx] =
            static_cast<OType>(scale_reciprocal * after_dact_reg[stage]);
        out_gate_colwise_sh_curr[shmem_idx] =
            static_cast<OType>(scale_reciprocal * after_dgate_reg[stage]);
      }
    }  // endif USE_COLWISE_SCALING

    // Wait for shared memory writes to be visible to TMA engine (cross-proxy fence)
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const int chunk_it_offset_y = chunk_offset_Y + it * BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;

      // dGeLU
      if constexpr (USE_ROWWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            TMAP_output_rowwise, chunk_it_offset_x, chunk_it_offset_y,
            reinterpret_cast<uint64_t *>(out_act_rowwise_sh_curr));

        // dGate
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            TMAP_output_rowwise, chunk_it_offset_x + cols, chunk_it_offset_y,
            reinterpret_cast<uint64_t *>(out_gate_rowwise_sh_curr));
      }

      // dGeLU
      if constexpr (USE_COLWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            TMAP_output_colwise, chunk_it_offset_x, chunk_it_offset_y,
            reinterpret_cast<uint64_t *>(out_act_colwise_sh_curr));

        // dGate
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            TMAP_output_colwise, chunk_it_offset_x + cols, chunk_it_offset_y,
            reinterpret_cast<uint64_t *>(out_gate_colwise_sh_curr));
      }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<BUFFERS_NUM - 1>();
    }
  }
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  float block_amax;
  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    block_amax = reduce_max<THREADS_PER_CHUNK / THREADS_PER_WARP>(thread_amax, warp_id);
  }

  if (is_master_thread && amax_ptr != nullptr) {
    atomicMaxFloat(amax_ptr, block_amax);
  }

  // Destroy the barriers. This invalidates the memory region of the barrier.
  // If further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      ptx::mbarrier_invalid(&mbar[it]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void cast_fp8_dgated(const Tensor &grad, const Tensor &gated_input, Tensor *output,
                     cudaStream_t stream) {
  NVTE_CHECK(!output->has_columnwise_data(), "Only cast supported in this function.");
  const size_t rows = grad.data.shape[0];
  const size_t cols = grad.data.shape[1];

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block_dim(THREADS_PER_CHUNK);
  const dim3 grid_dim(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      grad.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,

          alignas(64) CUtensorMap tensor_map_grad{};
          alignas(64) CUtensorMap tensor_map_gated_input{};
          alignas(64) CUtensorMap tensor_map_output{};

          create_2D_tensor_map(tensor_map_grad, grad.data, rows, cols,
                               SHMEM_DIM_Y, SHMEM_DIM_X, sizeof(IType));
          create_2D_tensor_map(tensor_map_gated_input, gated_input.data, rows, cols * 2,
                               SHMEM_DIM_Y, SHMEM_DIM_X, sizeof(IType));
          create_2D_tensor_map(tensor_map_output, output->data, rows, cols * 2, SHMEM_DIM_Y,
                               SHMEM_DIM_X, sizeof(OType));

          const size_t buff_elems_total = BUFFERS_NUM * SHMEM_DIM_Y * SHMEM_DIM_X;
          const size_t buff_size_aligned_in =
              DIVUP(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
          const size_t buff_size_aligned_out =
              DIVUP(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
          const size_t grad_mem = buff_size_aligned_in;
          const size_t in_act_mem = buff_size_aligned_in;
          const size_t in_gate_mem = buff_size_aligned_in;
          const size_t out_act_mem = buff_size_aligned_out;
          const size_t out_gate_mem = buff_size_aligned_out;
          // const size_t mbar_mem = ITERATIONS * sizeof(uint64_t);
          const size_t shmem_size = ALIGNMENT_SIZE + grad_mem + (in_act_mem + in_gate_mem) +
                                    (out_act_mem + out_gate_mem);  // + mbar_mem;

          cudaFuncSetAttribute(cast_fp8_dgated_kernel<ParamOP, ActOP, DActOP, IType, OType>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

          cast_fp8_dgated_kernel<ParamOP, ActOP, DActOP, IType, OType>
          <<<grid_dim, block_dim, shmem_size, stream>>>(tensor_map_grad, tensor_map_gated_input,
                                                        tensor_map_output, amax_ptr, scale_inv_ptr,
                                                        scale_ptr, rows, cols););  // NOLINT(*)
  );                                                                               // NOLINT(*)
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void cast_mxfp8_dgated(const Tensor &grad, const Tensor &gated_input, Tensor *output,
                       cudaStream_t stream) {
  const bool USE_ROWWISE_SCALING = output->has_data();
  const bool USE_COLWISE_SCALING = output->has_columnwise_data();

  // TODO: Make more general
  const size_t scale_dim_X_rowwise = USE_ROWWISE_SCALING ? 32 : 1;
  const size_t scale_dim_Y_colwise = USE_COLWISE_SCALING ? 32 : 1;

  const size_t rows = grad.data.shape[0];
  const size_t cols = grad.data.shape[1];
  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);
  const size_t scale_stride_rowwise = DIVUP(cols, scale_dim_X_rowwise);
  const size_t scale_stride_colwise = cols;

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);

  e8m0_t *const scales_rowwise_ptr =
      USE_ROWWISE_SCALING ? reinterpret_cast<e8m0_t *>(output->scale_inv.dptr) : nullptr;
  e8m0_t *const scales_colwise_ptr =
      USE_COLWISE_SCALING ? reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr) : nullptr;

  const dim3 block_dim(THREADS_PER_CHUNK);
  const dim3 grid_dim(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
      scale_dim_Y_colwise, SCALE_DIM_Y,
      TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
          scale_dim_X_rowwise, SCALE_DIM_X,
          TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
              grad.dtype(), IType,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
                  output->dtype(), OType,

                  alignas(64) CUtensorMap tensor_map_grad{};
                  alignas(64) CUtensorMap tensor_map_gated_input{};
                  alignas(64) CUtensorMap tensor_map_output_rowwise{};
                  alignas(64) CUtensorMap tensor_map_output_colwise{};

                  create_2D_tensor_map(tensor_map_grad, grad.data, rows, cols, SHMEM_DIM_Y,
                                       SHMEM_DIM_X, sizeof(IType));
                  create_2D_tensor_map(tensor_map_gated_input, gated_input.data, rows, cols * 2,
                                       SHMEM_DIM_Y, SHMEM_DIM_X, sizeof(IType));

                  if (USE_ROWWISE_SCALING) {
                    create_2D_tensor_map(tensor_map_output_rowwise, output->data, rows,
                                         cols * 2, SHMEM_DIM_Y, SHMEM_DIM_X, sizeof(OType));
                  }

                  if (USE_COLWISE_SCALING) {
                    create_2D_tensor_map(tensor_map_output_colwise, output->columnwise_data, rows,
                                         cols * 2, SHMEM_DIM_Y, SHMEM_DIM_X, sizeof(OType));
                  }

                  const size_t buff_elems_total = BUFFERS_NUM * SHMEM_DIM_Y * SHMEM_DIM_X;
                  const size_t buff_size_aligned_in =
                      DIVUP(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
                  const size_t buff_size_aligned_out =
                      DIVUP(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;

                  const size_t grad_mem = buff_size_aligned_in;
                  const size_t in_act_mem = buff_size_aligned_in;
                  const size_t in_gate_mem = buff_size_aligned_in;
                  const size_t in_mem = grad_mem + in_act_mem + in_gate_mem;

                  const size_t out_act_mem = buff_size_aligned_out;
                  const size_t out_gate_mem = buff_size_aligned_out;
                  size_t out_mem = out_act_mem + out_gate_mem;
                  if (USE_ROWWISE_SCALING && USE_COLWISE_SCALING) { out_mem *= 2; }

                  // const size_t mbar_mem = ITERATIONS * sizeof(uint64_t);
                  // const size_t shmem_size = ALIGNMENT_SIZE + in_mem + out_mem + mbar_mem;

                  const size_t shmem_size = ALIGNMENT_SIZE + in_mem + out_mem;

                  cudaFuncSetAttribute(cast_mxfp8_dgated_kernel<ParamOP, ActOP, DActOP, IType,
                                                                OType, SCALE_DIM_Y, SCALE_DIM_X>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

                  cast_mxfp8_dgated_kernel<ParamOP, ActOP, DActOP, IType, OType, SCALE_DIM_Y,
                                           SCALE_DIM_X>
                  <<<grid_dim, block_dim, shmem_size, stream>>>(
                      tensor_map_grad, tensor_map_gated_input, tensor_map_output_rowwise,
                      tensor_map_output_colwise, scales_rowwise_ptr, scales_colwise_ptr,
                      amax_ptr, rows, cols, scale_stride_rowwise,
                      scale_stride_colwise););  // NOLINT(*)
          );                                    // NOLINT(*)
      );                                        // NOLINT(*)
  );                                            // NOLINT(*)
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void fp8_quantize_dgated(const Tensor &grad, const Tensor &gated_input, Tensor *output,
                         cudaStream_t stream) {
  NVTE_CHECK(is_supported_by_CC_100(), "Not supported by the Arch < 10.0");

  checkCuDriverContext(stream);

  CheckInputTensor(grad, "grad");
  CheckInputTensor(gated_input, "gated_input");
  CheckOutputTensor(*output, "output");

  NVTE_CHECK(!is_fp8_dtype(grad.data.dtype), "Grad input must be in higher precision.");
  NVTE_CHECK(grad.data.shape.size() == 2, "Grad input must have 2 dimensions.");
  const size_t rows = grad.data.shape[0];
  const size_t cols = grad.data.shape[1];

  NVTE_CHECK(grad.data.dtype == gated_input.data.dtype, "Types of both inputs must match.");
  NVTE_CHECK(gated_input.data.shape.size() == 2, "Gated input must have 2 dimensions.");
  NVTE_CHECK(gated_input.data.shape[0] == rows, "Wrong dimension of the gated input.");
  NVTE_CHECK(gated_input.data.shape[1] == cols * 2, "Wrong dimension of the gated input.");

  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");
  if (output->has_data()) {
    NVTE_CHECK(is_fp8_dtype(output->data.dtype), "Output must have FP8 type.");
    NVTE_CHECK(output->data.shape.size() == 2, "Output must have 2 dimensions.");
    NVTE_CHECK(output->data.shape[0] == rows, "Wrong dimension of the output.");
    NVTE_CHECK(output->data.shape[1] == cols * 2, "Wrong dimension of the output.");
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  }
  if (output->has_columnwise_data()) {
    NVTE_CHECK(is_fp8_dtype(output->columnwise_data.dtype), "Output must have FP8 type.");
    NVTE_CHECK(output->columnwise_data.shape.size() == 2, "Output must have 2 dimensions.");
    NVTE_CHECK(output->columnwise_data.shape[0] == rows, "Wrong dimension of the output.");
    NVTE_CHECK(output->columnwise_data.shape[1] == cols * 2, "Wrong dimension of the output.");
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  }

  const bool isFullTile = (rows % CHUNK_DIM_Y == 0) && (cols % CHUNK_DIM_X == 0);
  NVTE_CHECK(isFullTile, "Only full tiles are supported.");

  if (is_delayed_tensor_scaling(output->scaling_mode)) {
    cast_fp8_dgated<ParamOP, ActOP, DActOP>(grad, gated_input, output, stream);
  } else if (is_mxfp_scaling(output->scaling_mode)) {
    cast_mxfp8_dgated<ParamOP, ActOP, DActOP>(grad, gated_input, output, stream);
  } else {
    NVTE_ERROR("Not supported FP8 scaling mode");
  }
}

}  // namespace transformer_engine

void nvte_quantize_dswiglu(const NVTETensor grad, const NVTETensor gated_input,
                           NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dswiglu);
  using namespace transformer_engine;

  fp8_quantize_dgated<Empty, silu<fp32, fp32>, dsilu<fp32, fp32>>(
      *reinterpret_cast<const Tensor *>(grad),
      *reinterpret_cast<const Tensor *>(gated_input),
      reinterpret_cast<Tensor *>(output), stream);
}
