/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include <cfloat>
#include <iostream>
#include <limits>
#include <unordered_map>

#include "../common.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "math.h"
#include "ptx.cuh"
#include "cuda_driver.h"

namespace transformer_engine {

using namespace ptx;

constexpr size_t ALIGNMENT_SIZE = 128;
constexpr size_t FP8_CHUNK_DIM_Y = 128;
constexpr size_t FP8_CHUNK_DIM_X = 128;
constexpr size_t FP8_CHUNKS_PER_BLOCK_Y = 1;
constexpr size_t FP8_CHUNKS_PER_BLOCK_X = 1;
constexpr size_t FP8_CHUNKS_PER_BLOCK = FP8_CHUNKS_PER_BLOCK_Y * FP8_CHUNKS_PER_BLOCK_X;
constexpr size_t FP8_THREADS_PER_CHUNK = 512;
constexpr size_t FP8_THREADS_PER_CHUNK_X = FP8_CHUNK_DIM_X;
constexpr size_t FP8_THREADS_PER_CHUNK_Y = FP8_THREADS_PER_CHUNK / FP8_THREADS_PER_CHUNK_X;
constexpr size_t FP8_BUFFERS_NUM = 2;
constexpr size_t FP8_PREFETCH_BUFFERS_NUM = 1;
static_assert(FP8_PREFETCH_BUFFERS_NUM < FP8_BUFFERS_NUM);

constexpr size_t FP8_BUFFER_DIM_Y = 32;
constexpr size_t FP8_BUFFER_DIM_X = FP8_CHUNK_DIM_X;  // 128
constexpr size_t FP8_SHMEM_DIM_Y = FP8_BUFFER_DIM_Y;  // 32
constexpr size_t FP8_SHMEM_DIM_X = FP8_BUFFER_DIM_X;  // 128

constexpr size_t FP8_BUFF_STAGES_NUM = FP8_BUFFER_DIM_Y / FP8_THREADS_PER_CHUNK_Y;    //  16 =  32 / 2
constexpr size_t FP8_ITERATIONS = FP8_CHUNK_DIM_Y / FP8_BUFFER_DIM_Y;                 //   8 = 128 / 16
static_assert(FP8_ITERATIONS >= FP8_PREFETCH_BUFFERS_NUM);

__device__ inline float sigmoidf(const float x) {
  return __frcp_rn(1.0f + __expf(-x));
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType>
__global__ void __launch_bounds__(FP8_THREADS_PER_CHUNK)
    cast_fp8_dgated_kernel(const __grid_constant__ CUtensorMap tensor_map_grad,
                           const __grid_constant__ CUtensorMap tensor_map_gated_input,
                           const __grid_constant__ CUtensorMap tensor_map_output,
                           float *const amax_ptr, float *const scale_inv_ptr,
                           const float *const scale_ptr, const size_t rows, const size_t cols) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  const int block_offset_Y = blockIdx.y * FP8_CHUNKS_PER_BLOCK_Y * FP8_CHUNK_DIM_Y;
  const int block_offset_X = blockIdx.x * FP8_CHUNKS_PER_BLOCK_X * FP8_CHUNK_DIM_X;

  const int tid_Y = threadIdx.x / FP8_THREADS_PER_CHUNK_X;
  const int tid_X = threadIdx.x % FP8_THREADS_PER_CHUNK_X;

  const int thread_offset_Y = tid_Y;
  const int thread_offset_X = tid_X;

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  extern __shared__ char dshmem_unaligned[];
  const uint64_t dshmem_unaligned_as_uint = reinterpret_cast<uint64_t>(dshmem_unaligned);
  const uint64_t dshmem_aligned_as_uint = DIVUP(dshmem_unaligned_as_uint, static_cast<uint64_t>(ALIGNMENT_SIZE))
                                          * ALIGNMENT_SIZE;
  char* dshmem = reinterpret_cast<char*>(dshmem_aligned_as_uint);

  const size_t buff_elems = FP8_SHMEM_DIM_Y * FP8_SHMEM_DIM_X;
  const size_t buff_elems_total = FP8_BUFFERS_NUM * buff_elems;
  const size_t buff_size_aligned_in = DIVUP(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
  const size_t buff_size_aligned_out = DIVUP(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;

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
  IType * in_grad_sh  = reinterpret_cast<IType*>(dshmem);
  IType * in_act_sh   = reinterpret_cast<IType*>(dshmem + grad_mem);
  IType * in_gate_sh  = reinterpret_cast<IType*>(dshmem + grad_mem + in_act_mem);
  OType * out_act_sh  = reinterpret_cast<OType*>(dshmem + grad_mem + in_mem);
  OType * out_gate_sh = reinterpret_cast<OType*>(dshmem + grad_mem + in_mem + out_act_mem);
  uint64_t * mbar  = reinterpret_cast<uint64_t*>(dshmem + grad_mem + in_mem + out_mem);

  const uint64_t* TMAP_grad_in = reinterpret_cast<const uint64_t *>(&tensor_map_grad);
  const uint64_t* TMAP_gate_in = reinterpret_cast<const uint64_t *>(&tensor_map_gated_input);
  const uint64_t* TMAP_output = reinterpret_cast<const uint64_t *>(&tensor_map_output);

  const bool is_master_thread = (threadIdx.x == 0);

  if (is_master_thread) {
    // Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
    #pragma unroll
    for (int it = 0; it < FP8_ITERATIONS; ++it) {
      mbarrier_init(&mbar[it], FP8_THREADS_PER_CHUNK);
    }
    fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int parity = 0;
  #pragma unroll
  for (int chunk = 0; chunk < FP8_CHUNKS_PER_BLOCK; ++chunk) {
    const int chunk_Y = chunk / FP8_CHUNKS_PER_BLOCK_X;
    const int chunk_X = chunk % FP8_CHUNKS_PER_BLOCK_X;

    const int chunk_offset_Y = block_offset_Y + chunk_Y * FP8_CHUNK_DIM_Y;
    const int chunk_offset_X = block_offset_X + chunk_X * FP8_CHUNK_DIM_X;

    if (is_master_thread) {
      #pragma unroll
      for (int prefetch_buff = 0; prefetch_buff < FP8_PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
        const int chunk_stage_offset_Y = chunk_offset_Y + prefetch_buff * FP8_BUFFER_DIM_Y;
        const int chunk_stage_offset_X = chunk_offset_X;
        // Initiate bulk tensor copy
        // Grad
        cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_grad_sh[prefetch_buff * buff_elems]), TMAP_grad_in,
            chunk_stage_offset_X, chunk_stage_offset_Y, &mbar[prefetch_buff]);
        // Act
        cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_act_sh[prefetch_buff * buff_elems]), TMAP_gate_in,
            chunk_stage_offset_X, chunk_stage_offset_Y, &mbar[prefetch_buff]);
        // Gate
        cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_gate_sh[prefetch_buff * buff_elems]), TMAP_gate_in,
            chunk_stage_offset_X + cols, chunk_stage_offset_Y, &mbar[prefetch_buff]);
        // Arrive on the barrier and tell how many bytes are expected to come in.
        mbarrier_arrive_expect_tx(&mbar[prefetch_buff], in_transaction_size);
      }
    } else {
      // Other threads just arrive
      #pragma unroll
      for (int prefetch_buff = 0; prefetch_buff < FP8_PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
        mbarrier_arrive(&mbar[prefetch_buff]);
      }
    }

    #pragma unroll
    for (int it = 0; it < FP8_ITERATIONS; ++it) {
      const int buff = it % FP8_BUFFERS_NUM;
      const int next_it = it + FP8_PREFETCH_BUFFERS_NUM;
      if (next_it < FP8_ITERATIONS) {
        if (is_master_thread) {
          const int next_buff = next_it % FP8_BUFFERS_NUM;
          const int chunk_it_offset_y = chunk_offset_Y + next_it * FP8_BUFFER_DIM_Y;
          const int chunk_it_offset_x = chunk_offset_X;
          // Initiate bulk tensor copy
          // Grad
          cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&in_grad_sh[next_buff * buff_elems]), TMAP_grad_in,
              chunk_it_offset_x, chunk_it_offset_y, &mbar[next_it]);
          // Act
          cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&in_act_sh[next_buff * buff_elems]), TMAP_gate_in,
              chunk_it_offset_x, chunk_it_offset_y, &mbar[next_it]);
          // Gate
          cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&in_gate_sh[next_buff * buff_elems]), TMAP_gate_in,
              chunk_it_offset_x + cols, chunk_it_offset_y, &mbar[next_it]);

          // Arrive on the barrier and tell how many bytes are expected to come in.
          mbarrier_arrive_expect_tx(&mbar[next_it], in_transaction_size);
        } else {
          // Other threads just arrive
          mbarrier_arrive(&mbar[next_it]);
        }
      }

      // Wait for the data to have arrived
      mbarrier_wait_parity(&mbar[it], parity);

      IType * in_grad_sh_curr = in_grad_sh + buff * buff_elems;
      IType * in_act_sh_curr = in_act_sh + buff * buff_elems;
      IType * in_gate_sh_curr = in_gate_sh + buff * buff_elems;
      OType * out_act_sh_curr = out_act_sh + buff * buff_elems;
      OType * out_gate_sh_curr = out_gate_sh + buff * buff_elems;

      #pragma unroll
      for (int stage = 0; stage < FP8_BUFF_STAGES_NUM; ++stage) {
        const int stage_offset_Y = stage * FP8_THREADS_PER_CHUNK_Y;
        const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
        const int shmem_offset_x = thread_offset_X;
        const int shmem_idx = shmem_offset_y * FP8_SHMEM_DIM_X + shmem_offset_x;

        float grad_elt = static_cast<float>(in_grad_sh_curr[shmem_idx]);
        float act_elt = static_cast<float>(in_act_sh_curr[shmem_idx]);
        float gate_elt = static_cast<float>(in_gate_sh_curr[shmem_idx]);

        const float x = act_elt;
        const float s = sigmoidf(x);
        const float silu_x = x * s;
        const float dsilu_x = x * s * (1 - s) + s; 

        float after_dact = dsilu_x * grad_elt * gate_elt;
        float after_dgate = silu_x * grad_elt;

        out_act_sh_curr[shmem_idx] = static_cast<OType>(scale * after_dact);
        out_gate_sh_curr[shmem_idx] = static_cast<OType>(scale * after_dgate);

        amax = fmaxf(amax, fabsf(after_dact));
        amax = fmaxf(amax, fabsf(after_dgate));
      }

      // Wait for shared memory writes to be visible to TMA engine.
      fence_proxy_async_shared_cta();
      __syncthreads();
      // After syncthreads, writes by all threads are visible to TMA engine.

      // Initiate TMA transfer to copy shared memory to global memory
      if (is_master_thread) {
        const int chunk_it_offset_y = chunk_offset_Y + it * FP8_BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        // dGeLU
        cp_async_bulk_tensor_2d_shared_to_global(
            TMAP_output, chunk_it_offset_x, chunk_it_offset_y,
            // reinterpret_cast<uint64_t *>(out_act_sh[buff]));
            reinterpret_cast<uint64_t *>(out_act_sh_curr));
        // dGate
        cp_async_bulk_tensor_2d_shared_to_global(
            TMAP_output, chunk_it_offset_x + cols, chunk_it_offset_y,
            // reinterpret_cast<uint64_t *>(out_gate_sh[buff]));
            reinterpret_cast<uint64_t *>(out_gate_sh_curr));

        // Create a "bulk async-group" out of the previous bulk copy operation.
        cp_async_bulk_commit_group();

        // Wait for TMA transfer to have finished reading shared memory.
        cp_async_bulk_wait_group_read<FP8_PREFETCH_BUFFERS_NUM>();
      }
    }
    cp_async_bulk_wait_group_read<0>();
    __syncthreads();

    parity ^= 1;
  }

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    amax = reduce_max<FP8_THREADS_PER_CHUNK / THREADS_PER_WARP>(amax, warp_id);
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
    for (int it = 0; it < FP8_ITERATIONS; ++it) {
      mbarrier_invalid(&mbar[it]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// Get a function pointer to the cuTensorMapEncodeTiled driver API
static PFN_cuTensorMapEncodeTiled cuDriverTensorMapEncodeTiled = [](){
  const void *driver_ptr = cuda_driver::get_symbol("cuTensorMapEncodeTiled");
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(driver_ptr);
}();

static CUtensorMapDataType get_CUtensorMapDataType(DType dtype) {
  static const std::unordered_map<DType, CUtensorMapDataType> dtypeMapping = {
      {DType::kByte, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
      {DType::kFloat32, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32},
      {DType::kFloat16, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16},
      {DType::kBFloat16, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16},
      {DType::kFloat8E4M3, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
      {DType::kFloat8E5M2, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8}};
  return dtypeMapping.at(dtype);
}

static inline bool isPointerAligned(const void *const ptr, const int alignment) {
  const uint64_t ptr_as_uint = reinterpret_cast<uint64_t>(ptr);
  return ptr_as_uint % alignment == 0;
}

// Set up parameters to create TMA descriptor.
template <typename T>
static void create_tensor_map(CUtensorMap &tensorMap, const Tensor *tensor_ptr,
                              const uint64_t globalY, const uint64_t globalX, const uint32_t shmemY,
                              const uint32_t shmemX) {
  const Tensor &tensor = *tensor_ptr;
  // rank is the number of dimensions of the array
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {globalX, globalY};

  // The stride is the number of bytes to traverse from the first element of one row to the next
  uint64_t stride[rank - 1] = {globalX * sizeof(T)};

  // The boxSize is the size of the shared memory buffer that is used as the
  // source/destination of a TMA transfer
  uint32_t boxSize[rank] = {shmemX, shmemY};

  // The distance between elements in units of sizeof(element)
  uint32_t elemStride[rank] = {1, 1};

  const CUtensorMapDataType tensorDataType = get_CUtensorMapDataType(tensor.data.dtype);
  void *dataPtr = reinterpret_cast<void *>(tensor.data.dptr);
  NVTE_CHECK(isPointerAligned(dataPtr, 16), "Tensor data must be 16B aligned");

  // Create the tensor descriptor.
  NVTE_CHECK_CUDA_DRIVER(cuDriverTensorMapEncodeTiled(
      &tensorMap,  // CUtensorMap *tensorMap,
      tensorDataType,
      rank,        // cuuint32_t tensorRank,
      dataPtr,     // void *globalAddress,
      size,        // const cuuint64_t *globalDim,
      stride,      // const cuuint64_t *globalStrides,
      boxSize,     // const cuuint32_t *boxDim,
      elemStride,  // const cuuint32_t *elementStrides,
      // Interleave patterns can be used to accelerate loading of values that
      // are less than 4 bytes long.
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,

      // Swizzling can be used to avoid shared memory bank conflicts.
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,

      // L2 Promotion can be used to widen the effect of a cache-policy to a wider
      // set of L2 cache lines.
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      // CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,

      // Any element that is outside of bounds will be set to zero by the TMA transfer.
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
  );
}

static const int32_t deviceComputeCapability = []() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  return 10 * deviceProp.major + deviceProp.minor;
}();

static bool is_supported_by_CC_100() { return deviceComputeCapability >= 100; }

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void fp8_quantize_dgated(const Tensor &grad, const Tensor &gated_input, Tensor *output,
                         cudaStream_t stream) {
  NVTE_CHECK(is_supported_by_CC_100(), "Not supported by the Arch < 10.0");

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

  NVTE_CHECK(is_fp8_dtype(output->data.dtype), "Output must have FP8 type.");
  NVTE_CHECK(output->data.shape.size() == 2, "Output must have 2 dimensions.");
  NVTE_CHECK(output->data.shape[0] == rows, "Wrong dimension of the output.");
  NVTE_CHECK(output->data.shape[1] == cols * 2, "Wrong dimension of the output.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");
  NVTE_CHECK(is_delayed_tensor_scaling(output->scaling_mode), "Only delayed scaling is supported");
  NVTE_CHECK(output->scaling_mode.x == -1 && output->scaling_mode.y == -1,
             "Only the regular FP8 cast is supported");

  const bool isFullTile = (rows % FP8_CHUNK_DIM_Y == 0) && (cols % FP8_CHUNK_DIM_X == 0);
  NVTE_CHECK(isFullTile, "Only full tiles are supported.");

  const size_t chunks_Y = DIVUP(rows, FP8_CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, FP8_CHUNK_DIM_X);
  const size_t blocks_Y = DIVUP(chunks_Y, FP8_CHUNKS_PER_BLOCK_Y);
  const size_t blocks_X = DIVUP(chunks_X, FP8_CHUNKS_PER_BLOCK_X);

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block_dim(FP8_THREADS_PER_CHUNK);
  const dim3 grid_dim(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(grad.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(output->data.dtype, OType,

          alignas(64) CUtensorMap tensor_map_grad{};
          alignas(64) CUtensorMap tensor_map_gated_input{};
          alignas(64) CUtensorMap tensor_map_output{};

          create_tensor_map<IType>(tensor_map_grad, &grad, rows, cols, FP8_SHMEM_DIM_Y, FP8_SHMEM_DIM_X);
          create_tensor_map<IType>(tensor_map_gated_input, &gated_input, rows, cols * 2, FP8_SHMEM_DIM_Y, FP8_SHMEM_DIM_X);
          create_tensor_map<OType>(tensor_map_output, output, rows, cols * 2, FP8_SHMEM_DIM_Y, FP8_SHMEM_DIM_X);


          const size_t buff_elems_total = FP8_BUFFERS_NUM * FP8_SHMEM_DIM_Y * FP8_SHMEM_DIM_X;
          const size_t buff_size_aligned_in = DIVUP(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
          const size_t buff_size_aligned_out = DIVUP(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
          const size_t grad_mem = buff_size_aligned_in;
          const size_t in_act_mem = buff_size_aligned_in;
          const size_t in_gate_mem = buff_size_aligned_in;
          const size_t out_act_mem = buff_size_aligned_out;
          const size_t out_gate_mem = buff_size_aligned_out;
          const size_t mbar_mem = FP8_ITERATIONS * sizeof(uint64_t);
          const size_t shmem_size = ALIGNMENT_SIZE + grad_mem +
                                    (in_act_mem + in_gate_mem) +
                                    (out_act_mem + out_gate_mem) + mbar_mem;

          cudaFuncSetAttribute(cast_fp8_dgated_kernel<ParamOP, ActOP, DActOP, IType, OType>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

          cast_fp8_dgated_kernel<ParamOP, ActOP, DActOP, IType, OType>
              <<<grid_dim, block_dim, shmem_size, stream>>>(
                  tensor_map_grad, tensor_map_gated_input, tensor_map_output, amax_ptr,
                  scale_inv_ptr, scale_ptr, rows, cols);
      );    // NOLINT(*)
  );        // NOLINT(*)
}

}  // namespace transformer_engine

void nvte_fp8_quantize_swiglu(const NVTETensor grad, const NVTETensor gated_input,
                              NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_swiglu);
  using namespace transformer_engine;

  constexpr auto Activation = &silu<fp32, fp32>;
  constexpr auto dActivation = &dsilu<fp32, fp32>;

  fp8_quantize_dgated<Empty, Activation, dActivation>(
      *reinterpret_cast<const Tensor *>(grad), *reinterpret_cast<const Tensor *>(gated_input),
      reinterpret_cast<Tensor *>(output), stream);
}
