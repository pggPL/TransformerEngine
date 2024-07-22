/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <iostream>
#include <limits>
#include <cfloat>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include "../common.h"
#include "../utils.cuh"
#include <transformer_engine/cast.h>
#include "../util/vectorized_pointwise.h"

namespace transformer_engine {

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

constexpr uint32_t FP32_EXPONENT_BIAS = 127;
constexpr uint32_t FP32_EXPONENT_BITS = 8;
constexpr uint32_t FP32_MANTISSA_BITS = 23;                                         // FP32 = [S1] [E8] [M23]
constexpr uint32_t SIGN_MASK = 1U << (FP32_MANTISSA_BITS + FP32_EXPONENT_BITS);     // most significant bit mask
constexpr uint32_t NUMBER_MASK = ~SIGN_MASK;
constexpr uint32_t MANTISSA_MASK = (1U << FP32_MANTISSA_BITS) - 1;
constexpr uint32_t EXPONENT_MASK = NUMBER_MASK & (~MANTISSA_MASK);

template <typename T>
struct Numeric_Traits {
    static constexpr double minSubnorm = 1.0;
    static constexpr double maxSubnorm = 1.0;
    static constexpr double minNorm    = 1.0;
    static constexpr double maxNorm    = 1.0;
    static constexpr double artifInf   = 1.0;
    static constexpr int maxBiasedExponent = 1;
};

template <>
struct Numeric_Traits<fp8e4m3> {
    static constexpr double minSubnorm = 1.0   / static_cast<double>(1 << 9);   // std::pow(2.0, -9.0);
    static constexpr double maxSubnorm = 0.875 / static_cast<double>(1 << 6);   // std::pow(2.0, -6.0);
    static constexpr double minNorm    = 1.0   / static_cast<double>(1 << 6);   // std::pow(2.0, -6.0);
    static constexpr double maxNorm    = 448.0;
    static constexpr double artifInf   = 10.0 * maxNorm;                        // artificial Infinity
    static constexpr int maxBiasedExponentAsFP32 = 8 + FP32_EXPONENT_BIAS;
    static constexpr int maxUnbiasedExponentAsFP32 = 8;

};

template <>
struct Numeric_Traits<fp8e5m2> {
    static constexpr double minSubnorm = 1.0  / static_cast<double>(1 << 16);   // std::pow(2.0, -16.0);
    static constexpr double maxSubnorm = 0.75 / static_cast<double>(1 << 14);   // std::pow(2.0, -14.0);
    static constexpr double minNorm    = 1.0  / static_cast<double>(1 << 14);   // std::pow(2.0, -14.0);
    static constexpr double maxNorm    = 57344.0;
    static constexpr double artifInf   = 10.0 * maxNorm;                        // artificial Infinity
    static constexpr int maxBiasedExponentAsFP32 = 15 + FP32_EXPONENT_BIAS;
    static constexpr int maxUnbiasedExponentAsFP32 = 15;
};

template <>
struct Numeric_Traits<fp32> {
    static constexpr double minSubnorm = std::numeric_limits<fp32>::denorm_min();   // std::pow(2.0, -149.0);
    static constexpr double maxSubnorm = std::numeric_limits<fp32>::min()
                                         - std::numeric_limits<fp32>::denorm_min(); // minNormalized - minDenormalized
    static constexpr double minNorm    = std::numeric_limits<fp32>::min();          // std::pow(2.0, -126.0);
    static constexpr double maxNorm    = std::numeric_limits<fp32>::max();          // (1 - pow(2, -24)) * pow(2, 128)
    static constexpr double artifInf   = std::numeric_limits<fp32>::infinity();
    static constexpr int maxBiasedExponentAsFP32 = 255;
    static constexpr int maxUnbiasedExponentAsFP32 = 128;
};

template <typename T>
struct Quantized_Limits {
    static constexpr double ranges[]  = {
        0.0,
        Numeric_Traits<T>::minNorm,
        Numeric_Traits<T>::maxNorm,
        Numeric_Traits<T>::artifInf
    };
    static constexpr inline fp32 min_norm() { return static_cast<fp32>(Numeric_Traits<T>::minNorm); }
    static constexpr inline fp32 max_norm() { return static_cast<fp32>(Numeric_Traits<T>::maxNorm); }
    static constexpr inline int max_norm_biased_exponent() { return Numeric_Traits<T>::maxBiasedExponentAsFP32; }
    static constexpr inline int max_norm_unbiased_exponent() { return Numeric_Traits<T>::maxUnbiasedExponentAsFP32; }
};

constexpr size_t THREAD_SEGMENT_DIM_Y = 1;         // elements per thread along Y-axis
constexpr size_t THREAD_SEGMENT_DIM_X = 8;         // elements per thread along X-axis
constexpr size_t CHUNK_DIM_Y = 32;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t TILE_DIM_Y = 32;
constexpr size_t TILE_DIM_X = 32;
constexpr size_t PREFETCH_STAGES = 3;

static_assert(CHUNK_DIM_Y % TILE_DIM_Y == 0);
static_assert(CHUNK_DIM_X % TILE_DIM_X == 0);

constexpr size_t TILES_PER_CHUNK_Y = CHUNK_DIM_Y / TILE_DIM_Y;                  //   1 = 32 / 32
constexpr size_t TILES_PER_CHUNK_X = CHUNK_DIM_X / TILE_DIM_X;                  //   4 = 128 / 32
constexpr size_t TILES_PER_CHUNK = TILES_PER_CHUNK_Y * TILES_PER_CHUNK_X;       //   4 = 1 * 4
constexpr size_t THREADS_PER_TILE = THREADS_PER_WARP;                           //  32
constexpr size_t THREADS_PER_TILE_X = TILE_DIM_X / THREAD_SEGMENT_DIM_X;        //   4 = 32 / 8
constexpr size_t THREADS_PER_TILE_Y = THREADS_PER_TILE / THREADS_PER_TILE_X;    //   8 = 32 / 4
constexpr size_t THREADS_PER_CHUNK = THREADS_PER_TILE * TILES_PER_CHUNK;        // 128 = 32 * 4


constexpr size_t STAGE_DIM_Y = THREADS_PER_TILE_Y * THREAD_SEGMENT_DIM_Y;       //   8 = 8 * 1
constexpr size_t STAGES_NUM = TILE_DIM_Y / STAGE_DIM_Y;                         //   4 = 32 / 8   TMA Stages
constexpr size_t SHMEM_DIM_Y = CHUNK_DIM_Y / STAGES_NUM;                        //   8 = 32 / 4
constexpr size_t SHMEM_DIM_X = CHUNK_DIM_X;                                     // 128

static_assert(STAGES_NUM >= PREFETCH_STAGES);

constexpr size_t MXFP8_BLOCK_DIM_Y = 1;
constexpr size_t MXFP8_BLOCK_DIM_X = 32;
constexpr size_t SCALES_PER_CHUNK_Y = CHUNK_DIM_Y / MXFP8_BLOCK_DIM_Y;
constexpr size_t SCALES_PER_CHUNK_X = CHUNK_DIM_X / MXFP8_BLOCK_DIM_X;
// constexpr size_t SCALES_PER_TILE_Y = TILE_DIM_Y / MXFP8_BLOCK_DIM_Y;
constexpr size_t SCALES_PER_TILE_X = TILE_DIM_X / MXFP8_BLOCK_DIM_X;
constexpr size_t THREADS_PER_SCALE_X = THREADS_PER_TILE_X / SCALES_PER_TILE_X;  //   4 = 4 / 1
constexpr size_t SUBWARP_WIDTH = THREADS_PER_SCALE_X;                           //   4

using byte = uint8_t;

// // Available from PTX ISA 8.6 and requires sm_100a
// __device__ __forceinline__ uint16_t cvt_fp32x2_to_e8m0x2(const float v1, const float v2) {
//     uint16_t e8m0_2pack;
//     asm volatile (
//         "{\n\t"
//         "cvt.rz.satfinite.ue8m0x2.f32 %0, %1, %2; \n\t"
//         "}"
//         : "=h"(e8m0_2pack)
//         : "f"(v1),
//           "f"(v2)
//     :);
//     int biased_exp_1 = e8m0_packed2 >> 8;
//     int biased_exp_2 = (e8m0_packed2 & 0xFF00) >> 8;
//     return e8m0_2pack; 
// }

__device__ __forceinline__ int32_t
extract_biased_exponent(const float val) {
    return (__float_as_int(val) & EXPONENT_MASK) >> FP32_MANTISSA_BITS;
}

__device__ __forceinline__ int
lower_biased_exp_limit_fp32(const int biased_exponent) {
    return (biased_exponent < 0) ? 0 : biased_exponent;
}

template <typename OType>
__device__ __forceinline__ byte
compute_shared_biased_exponent(const float amax) {
    __builtin_assume(amax >= 0);
    if (amax == 0.0f) {
        return 0 + FP32_EXPONENT_BIAS;
    }
    int exponent = extract_biased_exponent(amax)
                   - Quantized_Limits<OType>::max_norm_unbiased_exponent();

    // Clamp the shared unbiased exponent between the representable numbers of uint8_t
    // i.e., between [0, 255]
    return static_cast<byte>(lower_biased_exp_limit_fp32(exponent));
}

// Max reduction in subwarps
// E.g., if nvec=4, each warp processes 128 elements (32 x 4), that covers four MXFP8 scaling factors.
// To compute an actual scaling factor for 32 consequentive elements, only 8 threads need to participate,
// thus splitting the warp into 4x smaller subwarps 8-thread width. 'Butterfly' reduction is implemented
// inside those subwarps.
template <int subwarp_width>
__forceinline__ __device__ float
subwarp_max_reduction(const float val) {
    float val_tmp = val;
    #pragma unroll
    for (int offset = subwarp_width / 2; offset > 0; offset /= 2) {
        const float val_other = __shfl_down_sync(0xFFFFFFFF, val_tmp, offset, subwarp_width);
        __builtin_assume(val_tmp >= 0);
        __builtin_assume(val_other >= 0);
        val_tmp = fmaxf(val_tmp, val_other);
    }
    // Broadcast the amax to other threads of the subwarp from the zero subwarp lane_id
    constexpr int subwarp_lane_zero = 0;
    val_tmp = __shfl_sync(0xFFFFFFFF, val_tmp, subwarp_lane_zero, subwarp_width);
    return val_tmp;
}

template <typename CType, typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
cast_mxfp8_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                  const __grid_constant__ CUtensorMap tensor_map_output,
                  byte* const scales,
                  const size_t rows,
                  const size_t cols) {

    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int tile_id_y = warp_id / TILES_PER_CHUNK_X;
    const int tile_id_x = warp_id % TILES_PER_CHUNK_X;
    const int lane_id = threadIdx.x % THREADS_PER_WARP;
    const int thread_id_y = lane_id / THREADS_PER_TILE_X;
    const int thread_id_x = lane_id % THREADS_PER_TILE_X;
    const int scales_id_x = thread_id_x / THREADS_PER_SCALE_X;

    const int chunk_offset_y = blockIdx.y * CHUNK_DIM_Y;
    const int chunk_offset_x = blockIdx.x * CHUNK_DIM_X;
    const int tile_offset_y = tile_id_y * TILE_DIM_Y;
    const int tile_offset_x = tile_id_x * TILE_DIM_X;
    const int thread_offset_y = thread_id_y * THREAD_SEGMENT_DIM_Y;
    const int thread_offset_x = thread_id_x * THREAD_SEGMENT_DIM_X;

    // The destination shared memory buffer of a bulk tensor operation should be 128 byte aligned
    __shared__ alignas(128) IType in_sh[STAGES_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];
    __shared__ alignas(128) OType out_sh[SHMEM_DIM_Y * STAGES_NUM][SHMEM_DIM_X];
    constexpr int shmem_buff_size = sizeof(in_sh) / STAGES_NUM;

    const bool is_master_thread = (threadIdx.x == 0);

    // Initialize shared memory barrier with the number of threads participating in the barrier.
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[STAGES_NUM];

    if (is_master_thread) {
        // Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
        #pragma unroll
        for (int stage = 0; stage < STAGES_NUM; ++stage) {
            init(&bar[stage], THREADS_PER_CHUNK);
        }
        cde::fence_proxy_async_shared_cta();
    }
    // Syncthreads so initialized barrier is visible to all threads.
    __syncthreads();

    barrier::arrival_token token[STAGES_NUM];

    if (is_master_thread) {
        #pragma unroll
        for (int prefetch_stage = 0; prefetch_stage < PREFETCH_STAGES; ++prefetch_stage) {
            const int chunk_stage_offset_y = chunk_offset_y + prefetch_stage * STAGE_DIM_Y;
            const int chunk_stage_offset_x = chunk_offset_x;
            // Initiate bulk tensor copy
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &in_sh[prefetch_stage], &tensor_map_input, chunk_stage_offset_x, chunk_stage_offset_y, bar[prefetch_stage]);

            // Arrive on the barrier and tell how many bytes are expected to come in.
            token[prefetch_stage] = cuda::device::barrier_arrive_tx(bar[prefetch_stage], 1, shmem_buff_size);
        }
    } else {
        // Other threads just arrive
        #pragma unroll
        for (int prefetch_stage = 0; prefetch_stage < PREFETCH_STAGES; ++prefetch_stage) {
            token[prefetch_stage] = bar[prefetch_stage].arrive();
        }
    }

    using IVec = Vec<IType, THREAD_SEGMENT_DIM_X>;
    using OVecCast = Vec<OType, THREAD_SEGMENT_DIM_X>;

    IVec in[THREAD_SEGMENT_DIM_Y];
    OVecCast out_c[THREAD_SEGMENT_DIM_Y];
    __shared__ byte scales_sh[SCALES_PER_CHUNK_Y][SCALES_PER_CHUNK_X];

    #pragma unroll
    for (int stage = 0; stage < STAGES_NUM; ++stage) {
        const int next_stage = stage + PREFETCH_STAGES;
        if (next_stage < STAGES_NUM) {
            if (is_master_thread) {
                const int chunk_stage_offset_y = chunk_offset_y + next_stage * STAGE_DIM_Y;
                const int chunk_stage_offset_x = chunk_offset_x;
                // Initiate bulk tensor copy
                cde::cp_async_bulk_tensor_2d_global_to_shared(
                    &in_sh[next_stage], &tensor_map_input, chunk_stage_offset_x, chunk_stage_offset_y, bar[next_stage]);

                // Arrive on the barrier and tell how many bytes are expected to come in.
                token[next_stage] = cuda::device::barrier_arrive_tx(bar[next_stage], 1, shmem_buff_size);
            } else {
                // Other threads just arrive
                token[next_stage] = bar[next_stage].arrive();
            }
        }

        // Wait for the data to have arrived
        bar[stage].wait(std::move(token[stage]));
        #pragma unroll
        for (int i = 0; i < THREAD_SEGMENT_DIM_Y; ++i) {
            const int shmem_segment_offset_y = tile_offset_y + thread_offset_y;
            const int shmem_segment_offset_x = tile_offset_x + thread_offset_x;

            const int shmem_offset_y = shmem_segment_offset_y + i;
            const int shmem_offset_x = shmem_segment_offset_x;

            in[i].load_from(&in_sh[stage][shmem_offset_y][shmem_offset_x]);

            CType in_compute[THREAD_SEGMENT_DIM_X];
            CType thread_amax = 0;
            #pragma unroll
            for (int j = 0; j < THREAD_SEGMENT_DIM_X; ++j) {
                const float elt = static_cast<CType>(in[i].data.elt[j]);
                in_compute[j] = elt;
                if (isinf(elt) || isnan(elt)) {
                    continue;
                }
                thread_amax = fmaxf(thread_amax, fabsf(elt));
            }
            const CType subwarp_amax = subwarp_max_reduction<SUBWARP_WIDTH>(thread_amax);
            const byte biased_exponent = compute_shared_biased_exponent<OType>(subwarp_amax);
            const CType block_scale_inverse = exp2f(FP32_EXPONENT_BIAS - static_cast<float>(biased_exponent));

            const int scales_shmem_offset_y = shmem_offset_y + stage * STAGE_DIM_Y;
            const int scales_shmem_offset_x = tile_id_x * SCALES_PER_TILE_X + scales_id_x;
            scales_sh[scales_shmem_offset_y][scales_shmem_offset_x] = biased_exponent;

            #pragma unroll
            for (int j = 0; j < THREAD_SEGMENT_DIM_X; ++j) {
                out_c[i].data.elt[j] = static_cast<OType>(in_compute[j] * block_scale_inverse);
            }
            out_c[i].store_to(&out_sh[scales_shmem_offset_y][shmem_offset_x]);
        }
    }

    // Wait for shared memory writes to be visible to TMA engine.
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Only logical zero lane_id of a subwarp writes to global memory
    if (threadIdx.x < SCALES_PER_CHUNK_Y) {
        using ScalesVec = Vec<byte, SCALES_PER_CHUNK_X>;
        ScalesVec scales_vec;
        const int y = threadIdx.x;
        const int x = 0;
        scales_vec.load_from(&scales_sh[y][x]);
        const int scale_offset_y = blockIdx.y * SCALES_PER_CHUNK_Y + y;     // 1D scaling
        const int scale_offset_x = blockIdx.x * SCALES_PER_CHUNK_X + x;
        const int scale_stride = (cols + MXFP8_BLOCK_DIM_X - 1) / MXFP8_BLOCK_DIM_X;
        const int scale_idx = scale_offset_y * scale_stride + scale_offset_x;
        scales_vec.store_to(&scales[scale_idx]);
    }

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map_output, chunk_offset_x, chunk_offset_y, &out_sh);
        // Create a "bulk async-group" out of the previous bulk copy operation.
        cde::cp_async_bulk_commit_group();

        // Wait for TMA transfer to have finished reading shared memory.
        cde::cp_async_bulk_wait_group_read<0>();
    }
    // Destroy barrier. This invalidates the memory region of the barrier. If
    // further computations were to take place in the kernel, this allows the
    // memory location of the shared memory barrier to be reused.
    if (is_master_thread) {
        #pragma unroll
        for (int stage = 0; stage < STAGES_NUM; ++stage) {
            (&bar[stage])->~barrier();
        }
    }
}

void populate_cast_workspace_config(const size_t mxfp8_blocks_Y,
                                    const size_t mxfp8_blocks_X,
                                    Tensor& scales) {
    if (scales.data.dptr == nullptr) {
        scales.data.shape = {mxfp8_blocks_Y, mxfp8_blocks_X};
        scales.data.dtype = DType::kByte;
    }
}

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    void* driver_ptr = nullptr;
    cudaDriverEntryPointQueryResult driver_status;
    NVTE_CHECK_CUDA(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &driver_ptr, cudaEnableDefault, &driver_status));
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(driver_ptr);
}

CUtensorMapDataType get_CUtensorMapDataType(DType dtype) {
    static const std::unordered_map<DType, CUtensorMapDataType> dtypeMapping = {
        {DType::kByte,       CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
        {DType::kFloat32,    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32},
        {DType::kFloat16,    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16},
        {DType::kBFloat16,   CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16},
        {DType::kFloat8E4M3, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
        {DType::kFloat8E5M2, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8}
    };
    return dtypeMapping.at(dtype);
}

// Set up parameters to create TMA descriptor.
template <typename T>
void create_tensor_map(CUtensorMap& tensorMap,
                       const Tensor& tensor,
                       const uint64_t globalY,
                       const uint64_t globalX,
                       const uint32_t shmemY,
                       const uint32_t shmemX) {
    // rank is the number of dimensions of the array
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {globalX, globalY};

    // The stride is the number of bytes to traverse from the first element of one row to the next
    uint64_t stride[rank-1] = {globalX * sizeof(T)};

    // The boxSize is the size of the shared memory buffer that is used as the
    // source/destination of a TMA transfer
    uint32_t boxSize[rank] = {shmemX, shmemY};

    // The distance between elements in units of sizeof(element)
    uint32_t elemStride[rank] = {1, 1};

    // Get a function pointer to the cuTensorMapEncodeTiled driver API
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

    const CUtensorMapDataType tensorDataType = get_CUtensorMapDataType(tensor.data.dtype);
    void* dataPtr = reinterpret_cast<void*>(tensor.data.dptr);

    // Create the tensor descriptor.
    CUresult res = cuTensorMapEncodeTiled(
        &tensorMap,                                         // CUtensorMap *tensorMap,
        tensorDataType,
        rank,                                               // cuuint32_t tensorRank,
        dataPtr,                                            // void *globalAddress,
        size,                                               // const cuuint64_t *globalDim,
        stride,                                             // const cuuint64_t *globalStrides,
        boxSize,                                            // const cuuint32_t *boxDim,
        elemStride,                                         // const cuuint32_t *elementStrides,
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
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    cudaDeviceSynchronize();
    NVTE_CHECK_CUDA(cudaGetLastError());
}

void cast_mxfp8(const Tensor& input,
                Tensor* output_,
                Tensor* scales_,
                cudaStream_t stream) {
    Tensor& output = *output_;
    Tensor& scales = *scales_;

    CheckInputTensor(input, "cast_mxfp8_input");

    constexpr bool allow_empty = false;
    CheckOutputTensor(output, "output", allow_empty);

    NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
    NVTE_CHECK(output.data.shape.size() == 2, "C output must have 2 dimensions.");
    NVTE_CHECK(input.data.shape == output.data.shape, "Input and C output must have the same shape.");

    const size_t rows = input.data.shape[0];
    const size_t cols = input.data.shape[1];
    const size_t chunks_Y = DIVUP(rows, CHUNK_DIM_Y);
    const size_t chunks_X = DIVUP(cols, CHUNK_DIM_X);
    const size_t mxfp8_blocks_Y = DIVUP(rows, MXFP8_BLOCK_DIM_Y);
    const size_t mxfp8_blocks_X = DIVUP(cols, MXFP8_BLOCK_DIM_X);

    const bool isFullTile = (rows % CHUNK_DIM_Y == 0) && (cols % CHUNK_DIM_X == 0);
    NVTE_CHECK(isFullTile, "Only full tiles are supported.");

    if (scales.data.dptr == nullptr) {
        populate_cast_workspace_config(mxfp8_blocks_Y, mxfp8_blocks_X, scales);
        return;
    }
    const dim3 block(THREADS_PER_CHUNK);
    const dim3 grid(chunks_X, chunks_Y);

    TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, IType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(output.data.dtype, OType,
            CUtensorMap tensor_map_input{};
            CUtensorMap tensor_map_output{};

            byte* scales_ptr = reinterpret_cast<byte*>(scales.data.dptr);

            create_tensor_map<IType>(tensor_map_input, input, rows, cols, SHMEM_DIM_Y, SHMEM_DIM_X);
            create_tensor_map<OType>(tensor_map_output, output, rows, cols, SHMEM_DIM_Y * STAGES_NUM, SHMEM_DIM_X);

            cast_mxfp8_kernel<fp32, IType, OType>
                <<<grid, block, 0, stream>>>
                (tensor_map_input, tensor_map_output, scales_ptr, rows, cols);
        ); // NOLINT(*)
    );     // NOLINT(*)

    NVTE_CHECK_CUDA(cudaGetLastError());
}
} // namespace transformer_engine

void nvte_cast_mxfp8(const NVTETensor input,
                     NVTETensor output,
                     NVTETensor scales,
                     cudaStream_t stream) {
    NVTE_API_CALL(nvte_cast_mxfp8);
    using namespace transformer_engine;
    cast_mxfp8(*reinterpret_cast<const Tensor*>(input),
               reinterpret_cast<Tensor*>(output),
               reinterpret_cast<Tensor*>(scales),
               stream);
}
