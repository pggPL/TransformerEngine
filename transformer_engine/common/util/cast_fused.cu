/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast_fused.h>

#include <cfloat>
#include <iostream>
#include <limits>
#include <unordered_map>

#include "../common.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "math.h"
#include "ptx.cuh"

namespace transformer_engine {

using namespace ptx;

constexpr size_t CHUNK_DIM_Y = 64;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t CHUNKS_PER_BLOCK_Y = 1;
constexpr size_t CHUNKS_PER_BLOCK_X = 1;
constexpr size_t CHUNKS_PER_BLOCK = CHUNKS_PER_BLOCK_Y * CHUNKS_PER_BLOCK_X;
constexpr size_t THREADS_PER_CHUNK = 128;
constexpr size_t DBIAS_THREADS_PER_BLOCK = 256;
constexpr size_t BUFFERS_NUM = 2;
constexpr size_t PREFETCH_BUFFERS_NUM = 1;
static_assert(PREFETCH_BUFFERS_NUM < BUFFERS_NUM);

constexpr size_t BUFFER_DIM_Y = 16;
constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;  // 128
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;  // 16
constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;  // 128

constexpr size_t BUFF_STAGES_NUM = BUFFER_DIM_Y;            //  16
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;   //   4 = 64 / 16
static_assert(ITERATIONS >= PREFETCH_BUFFERS_NUM);

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    cast_fused_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                      const __grid_constant__ CUtensorMap tensor_map_act_input,
                      const __grid_constant__ CUtensorMap tensor_map_output,
                      float * const dbias_workspace,
                      float * const amax_ptr,
                      float * const scale_inv_ptr,
                      const float * const scale_ptr,
                      const size_t rows,
                      const size_t cols) {
// #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  const int block_offset_Y = blockIdx.y * CHUNKS_PER_BLOCK_Y * CHUNK_DIM_Y;
  const int block_offset_X = blockIdx.x * CHUNKS_PER_BLOCK_X * CHUNK_DIM_X;

  const int tid_Y = threadIdx.x / THREADS_PER_CHUNK;
  const int tid_X = threadIdx.x % THREADS_PER_CHUNK;

  const int thread_offset_Y = tid_Y;
  const int thread_offset_X = tid_X;

  const int dbias_offset_Y = blockIdx.y * CHUNKS_PER_BLOCK_Y + tid_Y;
  const int dbias_block_offset_X = blockIdx.x * CHUNKS_PER_BLOCK_X * CHUNK_DIM_X + thread_offset_X;
  const int dbias_stride = cols;

  float partial_dbias[CHUNKS_PER_BLOCK_X];

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  // The destination shared memory buffer of a bulk tensor operation should be 128-byte aligned
  __shared__ alignas(16) IType in_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];
  __shared__ alignas(16) IType act_in_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];
  __shared__ alignas(16) OType out_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];

  constexpr int shmem_buff_size = sizeof(in_sh) / BUFFERS_NUM;
  constexpr int transaction_size = shmem_buff_size * (IS_DACT ? 2 : 1);

  const bool is_master_thread = (threadIdx.x == 0);

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  if (is_master_thread) {
    // Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      mbarrier_init(&mbar[it], THREADS_PER_CHUNK);
    }
    fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int parity = 0;
  #pragma unroll
  for (int chunk = 0; chunk < CHUNKS_PER_BLOCK; ++chunk) {
    const int chunk_Y = chunk / CHUNKS_PER_BLOCK_X;
    const int chunk_X = chunk % CHUNKS_PER_BLOCK_X;

    const int chunk_offset_Y = block_offset_Y + chunk_Y * CHUNK_DIM_Y;
    const int chunk_offset_X = block_offset_X + chunk_X * CHUNK_DIM_X;

    if (is_master_thread) {
      #pragma unroll
      for (int prefetch_buff = 0; prefetch_buff < PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
        const int chunk_stage_offset_Y = chunk_offset_Y + prefetch_buff * BUFFER_DIM_Y;
        const int chunk_stage_offset_X = chunk_offset_X;
        // Initiate bulk tensor copy
        cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_sh[prefetch_buff]),
            reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_stage_offset_X,
            chunk_stage_offset_Y, &mbar[prefetch_buff]);

        if constexpr (IS_DACT) {
          cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&act_in_sh[prefetch_buff]),
              reinterpret_cast<const uint64_t *>(&tensor_map_act_input), chunk_stage_offset_X,
              chunk_stage_offset_Y, &mbar[prefetch_buff]);
        }

        // Arrive on the barrier and tell how many bytes are expected to come in.
        mbarrier_arrive_expect_tx(&mbar[prefetch_buff], transaction_size);
      }
    } else {
      // Other threads just arrive
      #pragma unroll
      for (int prefetch_buff = 0; prefetch_buff < PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
        mbarrier_arrive(&mbar[prefetch_buff]);
      }
    }

    #pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      const int buff = it % BUFFERS_NUM;
      const int next_it = it + PREFETCH_BUFFERS_NUM;
      if (next_it < ITERATIONS) {
        if (is_master_thread) {
          const int next_buff = next_it % BUFFERS_NUM;
          const int chunk_it_offset_y = chunk_offset_Y + next_it * BUFFER_DIM_Y;
          const int chunk_it_offset_x = chunk_offset_X;
          // Initiate bulk tensor copy
          cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&in_sh[next_buff]),
              reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_it_offset_x,
              chunk_it_offset_y, &mbar[next_it]);

          if constexpr (IS_DACT) {
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t *>(&act_in_sh[next_buff]),
                reinterpret_cast<const uint64_t *>(&tensor_map_act_input), chunk_it_offset_x,
                chunk_it_offset_y, &mbar[next_it]);
          }

          // Arrive on the barrier and tell how many bytes are expected to come in.
          mbarrier_arrive_expect_tx(&mbar[next_it], transaction_size);
        } else {
          // Other threads just arrive
          mbarrier_arrive(&mbar[next_it]);
        }
      }

      // Wait for the data to have arrived
      mbarrier_wait_parity(&mbar[it], parity);

      #pragma unroll
      for (int stage = 0; stage < BUFF_STAGES_NUM; ++stage) {
        const int stage_offset_Y = stage;
        const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
        const int shmem_offset_x = thread_offset_X;

        float elt = static_cast<float>(in_sh[buff][shmem_offset_y][shmem_offset_x]);
        if constexpr (IS_DACT) {
          elt *= OP(static_cast<float>(act_in_sh[buff][shmem_offset_y][shmem_offset_x]), {});
        }
        if constexpr (IS_DBIAS) {
          partial_dbias[chunk_X] += elt;
        }
        if (isfinite(elt)) {
          amax = fmaxf(amax, fabsf(elt));
        }
        out_sh[buff][shmem_offset_y][shmem_offset_x] = static_cast<OType>(elt * scale);
      }

      // Wait for shared memory writes to be visible to TMA engine.
      fence_proxy_async_shared_cta();
      __syncthreads();
      // After syncthreads, writes by all threads are visible to TMA engine.

      // Initiate TMA transfer to copy shared memory to global memory
      if (is_master_thread) {
        const int chunk_it_offset_y = chunk_offset_Y + it * BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output), chunk_it_offset_x,
            chunk_it_offset_y, reinterpret_cast<uint64_t *>(&out_sh[buff]));

        // Create a "bulk async-group" out of the previous bulk copy operation.
        cp_async_bulk_commit_group();

        // Wait for TMA transfer to have finished reading shared memory.
        cp_async_bulk_wait_group_read<PREFETCH_BUFFERS_NUM>();
      }
    }
    cp_async_bulk_wait_group_read<0>();
    __syncthreads();

    parity ^= 1;
  }

  if constexpr (IS_DBIAS) {
    #pragma unroll
    for (int i = 0; i < CHUNKS_PER_BLOCK_X; ++i) {
      const int dbias_offset_X = dbias_block_offset_X + i * CHUNK_DIM_X;
      const int dbias_offset = dbias_offset_Y * dbias_stride + dbias_offset_X;
      dbias_workspace[dbias_offset] = partial_dbias[i];
    }
  }

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
      mbarrier_invalid(&mbar[it]);
    }
  }
// #endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

static PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
  void *driver_ptr = nullptr;
  cudaDriverEntryPointQueryResult driver_status;
  NVTE_CHECK_CUDA(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &driver_ptr, cudaEnableDefault,
                                          &driver_status));
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(driver_ptr);
}

static CUtensorMapDataType get_CUtensorMapDataType(DType dtype) {
  static const std::unordered_map<DType, CUtensorMapDataType> dtypeMapping = {
      {DType::kByte, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
      {DType::kFloat32, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32},
      {DType::kFloat16, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16},
      {DType::kBFloat16, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16},
      {DType::kFloat8E4M3, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
      {DType::kFloat8E5M2, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8}
  };
  return dtypeMapping.at(dtype);
}

// Set up parameters to create TMA descriptor.
template <typename T>
static void create_tensor_map(CUtensorMap &tensorMap, const Tensor *tensor_ptr, const uint64_t globalY,
                              const uint64_t globalX, const uint32_t shmemY, const uint32_t shmemX) {
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

  // Get a function pointer to the cuTensorMapEncodeTiled driver API
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  const CUtensorMapDataType tensorDataType = get_CUtensorMapDataType(tensor.data.dtype);
  void *dataPtr = reinterpret_cast<void *>(tensor.data.dptr);

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
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
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  cudaDeviceSynchronize();
}

template <int nvec, typename OType>
__global__ void __launch_bounds__(DBIAS_THREADS_PER_BLOCK)
    reduce_dbias_kernel(OType *const dbias_output, const float *const dbias_partial, const int rows,
                        const int cols) {
  using ComputeVec = Vec<float, nvec>;
  using OutputVec = Vec<OType, nvec>;

  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id * nvec >= cols) {
    return;
  }

  const float *const thread_in_base = dbias_partial + thread_id * nvec;
  OType *const thread_out_base = dbias_output + thread_id * nvec;

  ComputeVec ldg_vec;
  ComputeVec acc_vec;
  acc_vec.clear();
  for (int i = 0; i < rows; ++i) {
    ldg_vec.load_from(thread_in_base + i * cols);
    #pragma unroll
    for (int e = 0; e < nvec; ++e) {
      acc_vec.data.elt[e] += ldg_vec.data.elt[e];
    }
  }

  OutputVec stg_vec;
  #pragma unroll
  for (int e = 0; e < nvec; ++e) {
    stg_vec.data.elt[e] = static_cast<OType>(acc_vec.data.elt[e]);
  }
  stg_vec.store_to(thread_out_base);
}

template <typename IType>
void reduce_dbias(const float *workspace_ptr, Tensor *dbias, const size_t rows, const size_t cols,
                  cudaStream_t stream) {
  constexpr size_t ELEMS_PER_THREAD = 16;
  NVTE_CHECK(cols % ELEMS_PER_THREAD == 0, "Unsupported shape.");
  const size_t reduce_dbias_num_blocks = DIVUP(cols, DBIAS_THREADS_PER_BLOCK * ELEMS_PER_THREAD);

  reduce_dbias_kernel<ELEMS_PER_THREAD, IType>
      <<<reduce_dbias_num_blocks, DBIAS_THREADS_PER_BLOCK, 0, stream>>>(
          reinterpret_cast<IType *>(dbias->data.dptr), workspace_ptr, rows, cols);
}

static const int32_t deviceComputeCapability = []() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  return 10 * deviceProp.major + deviceProp.minor;
}();

static bool is_supported_by_CC_100() { return deviceComputeCapability >= 100; }

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void cast_fused(const Tensor &input,
                const Tensor &act_input,
                Tensor *output,
                Tensor *dbias,
                Tensor *workspace,
                cudaStream_t stream) {
  CheckInputTensor(input, "fused_cast_input");
  CheckOutputTensor(*output, "fused_cast_output");

  if constexpr (IS_DBIAS) {
    CheckOutputTensor(*dbias, "dbias");
  }
  if constexpr (IS_DACT) {
    CheckInputTensor(act_input, "activation_input");
    NVTE_CHECK(input.data.dtype == act_input.data.dtype, "Types of both inputs must match.");
    NVTE_CHECK(input.data.shape == act_input.data.shape, "Shapes of both inputs must match.");
  }

  NVTE_CHECK(!is_fp8_dtype(input.data.dtype), "Input must be in higher precision.");

  NVTE_CHECK(is_fp8_dtype(output->data.dtype), "Output must have FP8 type.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");
  NVTE_CHECK(output->amax.dptr != nullptr, "Amax tensor must be allocated");

  if (!is_supported_by_CC_100()) {
    NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
  }

  const size_t rows = input.data.shape[0];
  const size_t cols = input.data.shape[1];
  const size_t chunks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, CHUNK_DIM_X);
  const size_t blocks_Y = DIVUP(chunks_Y, CHUNKS_PER_BLOCK_Y);
  const size_t blocks_X = DIVUP(chunks_X, CHUNKS_PER_BLOCK_X);

  const bool isFullTile = (rows % CHUNK_DIM_Y == 0) && (cols % CHUNK_DIM_X == 0);
  NVTE_CHECK(isFullTile, "Only full tiles are supported.");

  const size_t dbias_rows = blocks_Y;
  const size_t dbias_cols = cols;

  if constexpr (IS_DBIAS) {
    NVTE_CHECK(dbias->data.dtype == input.data.dtype, "DBias must have the same type as input.");
    NVTE_CHECK(dbias->data.shape == std::vector<size_t>{cols}, "Wrong shape of DBias.");
    NVTE_CHECK(workspace != nullptr, "Workspace must be a tensor.");

    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {dbias_rows, dbias_cols};
      workspace->data.dtype = DType::kFloat32;
      return;
    }
  }
  float * const workspace_ptr = IS_DBIAS ? reinterpret_cast<float *>(workspace->data.dptr) : nullptr;
  float * const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float * const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float * const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block(THREADS_PER_CHUNK);
  const dim3 grid(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(output->data.dtype, OType,

          CUtensorMap tensor_map_input{};
          CUtensorMap tensor_map_act_input{};
          CUtensorMap tensor_map_output{};

          create_tensor_map<IType>(tensor_map_input, &input, rows, cols, SHMEM_DIM_Y, SHMEM_DIM_X);

          if constexpr (IS_DACT) {
            create_tensor_map<IType>(tensor_map_act_input, &act_input, rows, cols, SHMEM_DIM_Y, SHMEM_DIM_X);
          }
          create_tensor_map<OType>(tensor_map_output, output, rows, cols, SHMEM_DIM_Y, SHMEM_DIM_X);

          cast_fused_kernel<IS_DBIAS, IS_DACT, ParamOP, OP, IType, OType>
              <<<grid, block, 0, stream>>>(
                  tensor_map_input, tensor_map_act_input, tensor_map_output,
                  workspace_ptr, amax_ptr, scale_inv_ptr, scale_ptr, rows, cols);

          if constexpr (IS_DBIAS) {
            reduce_dbias<IType>(workspace_ptr, dbias, dbias_rows, dbias_cols, stream);
          }
      );               // NOLINT(*)
  );                   // NOLINT(*)
}
}  // namespace transformer_engine

void nvte_cast(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;

  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;

  constexpr const NVTETensor activation_input = nullptr;

  cast_fused<IS_DBIAS, IS_DACT, Empty, nullptr>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_cast_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                     NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_dbias);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = false;

  constexpr const NVTETensor activation_input = nullptr;

  cast_fused<IS_DBIAS, IS_DACT, Empty, nullptr>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_cast_dbias_dgelu(const NVTETensor input, const NVTETensor activation_input,
                           NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                           cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_dbias_dgelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  constexpr auto dActivation = &dgelu<fp32, fp32>;

  cast_fused<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_cast_dbias_dsilu(const NVTETensor input, const NVTETensor activation_input,
                           NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                           cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_dbias_dsilu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  constexpr auto dActivation = &dsilu<fp32, fp32>;

  cast_fused<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_cast_dbias_drelu(const NVTETensor input, const NVTETensor activation_input,
                           NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                           cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_dbias_drelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  constexpr auto dActivation = &drelu<fp32, fp32>;

  cast_fused<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_cast_dbias_dqgelu(const NVTETensor input, const NVTETensor activation_input,
                            NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                            cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_dbias_dqgelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  constexpr auto dActivation = &dqgelu<fp32, fp32>;

  cast_fused<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_cast_dbias_dsrelu(const NVTETensor input, const NVTETensor activation_input,
                            NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                            cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_dbias_dsrelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  constexpr auto dActivation = &dsrelu<fp32, fp32>;

  cast_fused<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}
