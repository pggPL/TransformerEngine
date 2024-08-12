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
#include <transformer_engine/cast.h>

#include "../common.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "ptx.cuh"

namespace transformer_engine {

using namespace ptx;

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

constexpr size_t MXFP8_BLOCK_DIM_Y = 1;
constexpr size_t MXFP8_BLOCK_DIM_X = 32;
constexpr size_t ELEMS_PER_THREAD = 16;         // along X-axis
constexpr size_t CHUNK_DIM_Y = 64;
constexpr size_t CHUNK_DIM_X = 64;
constexpr size_t CHUNKS_PER_BLOCK_Y = 1;
constexpr size_t CHUNKS_PER_BLOCK_X = 1;
constexpr size_t CHUNKS_PER_BLOCK = CHUNKS_PER_BLOCK_Y * CHUNKS_PER_BLOCK_X;
constexpr size_t THREADS_PER_CHUNK = 128;
constexpr size_t PREFETCH_STAGES = 1;
constexpr size_t STAGES_NUM = 2;
static_assert(PREFETCH_STAGES < STAGES_NUM);
static_assert(ELEMS_PER_THREAD >= 2);

constexpr size_t THREADS_PER_CHUNK_X = CHUNK_DIM_X / ELEMS_PER_THREAD;              //   8 = 128 / 16
constexpr size_t THREADS_PER_CHUNK_Y = THREADS_PER_CHUNK / THREADS_PER_CHUNK_X;     //  16 = 128 / 8
constexpr size_t ITERATIONS = CHUNK_DIM_Y / THREADS_PER_CHUNK_Y;                    //   8 = 128 / 16
constexpr size_t STAGE_DIM_Y = THREADS_PER_CHUNK_Y;                                 //  16
static_assert(ITERATIONS >= PREFETCH_STAGES);

constexpr size_t SCALES_PER_CHUNK_Y = CHUNK_DIM_Y;                                  // 128
constexpr size_t SCALES_PER_CHUNK_X = CHUNK_DIM_X / MXFP8_BLOCK_DIM_X;              //   4 = 128 / 32
constexpr size_t SCALES_PER_BLOCK_Y = SCALES_PER_CHUNK_Y * CHUNKS_PER_BLOCK_Y;      // 128 = 128 * 1
constexpr size_t SCALES_PER_BLOCK_X = SCALES_PER_CHUNK_X * CHUNKS_PER_BLOCK_X;      //   4 = 4 * 1

constexpr size_t THREADS_PER_SCALE_X = MXFP8_BLOCK_DIM_X / ELEMS_PER_THREAD;        //   2 = 32 / 16
constexpr size_t SUBWARP_WIDTH = THREADS_PER_SCALE_X;                               //   2

constexpr size_t SHMEM_DIM_Y = CHUNK_DIM_Y / ITERATIONS;                            //  16 = 128 / 8
constexpr size_t SHMEM_DIM_X = CHUNK_DIM_X;                                         // 128

using e8m0_t = uint8_t;

__device__ __forceinline__ int32_t
extract_biased_exponent(const float val) {
    return (__float_as_int(val) & EXPONENT_MASK) >> FP32_MANTISSA_BITS;
}

__device__ __forceinline__ int
lower_biased_exp_limit_fp32(const int biased_exponent) {
    return (biased_exponent < 0) ? 0 : biased_exponent;
}

template <typename OType>
__device__ __forceinline__ e8m0_t
compute_shared_biased_exponent(float amax) {
    __builtin_assume(amax >= 0);
    if (amax == 0) {
        constexpr int exponent_ = 0 + FP32_EXPONENT_BIAS;
        return static_cast<e8m0_t>(exponent_);
    }
    int exponent = extract_biased_exponent(amax)
                   - Quantized_Limits<OType>::max_norm_unbiased_exponent();

    // Clamp the shared unbiased exponent between the representable numbers of uint8_t
    // i.e., between [0, 255]
    return static_cast<e8m0_t>(lower_biased_exp_limit_fp32(exponent));
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
                  e8m0_t* const scales,
                  const size_t rows,
                  const size_t cols) {

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

    const int block_offset_y = blockIdx.y * CHUNKS_PER_BLOCK_Y * CHUNK_DIM_Y;
    const int block_offset_x = blockIdx.x * CHUNKS_PER_BLOCK_X * CHUNK_DIM_X;

    const int thread_id_y = threadIdx.x / THREADS_PER_CHUNK_X;
    const int thread_id_x = threadIdx.x % THREADS_PER_CHUNK_X;
    const int scales_id_x = thread_id_x / THREADS_PER_SCALE_X;

    const int thread_offset_y = thread_id_y;
    const int thread_offset_x = thread_id_x * ELEMS_PER_THREAD;

    // The destination shared memory buffer of a bulk tensor operation should be 128 e8m0_t aligned
    __shared__ alignas(16) IType in_sh[STAGES_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];
    __shared__ alignas(16) OType out_sh[STAGES_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];
    constexpr int shmem_buff_size = sizeof(in_sh) / STAGES_NUM;

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

        const int chunk_offset_y = block_offset_y + chunk_Y * CHUNK_DIM_Y;
        const int chunk_offset_x = block_offset_x + chunk_X * CHUNK_DIM_X;

        if (is_master_thread) {
            #pragma unroll
            for (int prefetch_stage = 0; prefetch_stage < PREFETCH_STAGES; ++prefetch_stage) {
                const int chunk_stage_offset_y = chunk_offset_y + prefetch_stage * STAGE_DIM_Y;
                const int chunk_stage_offset_x = chunk_offset_x;
                // Initiate bulk tensor copy
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&in_sh[prefetch_stage]),
                    reinterpret_cast<const uint64_t*>(&tensor_map_input),
                    chunk_stage_offset_x,
                    chunk_stage_offset_y,
                    &mbar[prefetch_stage]);

                // Arrive on the barrier and tell how many bytes are expected to come in.
                mbarrier_arrive_expect_tx(&mbar[prefetch_stage], shmem_buff_size);
            }
        } else {
            // Other threads just arrive
            #pragma unroll
            for (int prefetch_stage = 0; prefetch_stage < PREFETCH_STAGES; ++prefetch_stage) {
                mbarrier_arrive(&mbar[prefetch_stage]);
            }
        }

        using IVec = Vec<IType, ELEMS_PER_THREAD>;
        using OVecCast = Vec<OType, ELEMS_PER_THREAD>;

        IVec in;
        OVecCast out_c;
        __shared__ e8m0_t scales_sh[SCALES_PER_CHUNK_Y][SCALES_PER_CHUNK_X];

        #pragma unroll
        for (int it = 0; it < ITERATIONS; ++it) {
            const int stage = it % STAGES_NUM;
            const int next_it = it + PREFETCH_STAGES;
            if (next_it < ITERATIONS) {
                if (is_master_thread) {
                    const int next_stage = next_it % STAGES_NUM;
                    const int chunk_it_offset_y = chunk_offset_y + next_it * STAGE_DIM_Y;
                    const int chunk_it_offset_x = chunk_offset_x;
                    // Initiate bulk tensor copy
                    cp_async_bulk_tensor_2d_global_to_shared(
                        reinterpret_cast<uint64_t*>(&in_sh[next_stage]),
                        reinterpret_cast<const uint64_t*>(&tensor_map_input),
                        chunk_it_offset_x,
                        chunk_it_offset_y,
                        &mbar[next_it]);

                    // Arrive on the barrier and tell how many bytes are expected to come in.
                    mbarrier_arrive_expect_tx(&mbar[next_it], shmem_buff_size);
                } else {
                    // Other threads just arrive
                    mbarrier_arrive(&mbar[next_it]);
                }
            }

            // Wait for the data to have arrived
            mbarrier_wait_parity(&mbar[it], parity);

            const int shmem_offset_y = thread_offset_y;
            const int shmem_offset_x = thread_offset_x;

            in.load_from(&in_sh[stage][shmem_offset_y][shmem_offset_x]);

            CType in_compute[ELEMS_PER_THREAD];
            CType thread_amax = 0;

            #pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
                const float elt = static_cast<CType>(in.data.elt[j]);
                in_compute[j] = elt;
                if (isfinite(elt)) {
                    thread_amax = fmaxf(thread_amax, fabsf(elt));
                }
            }

            const CType subwarp_amax = subwarp_max_reduction<SUBWARP_WIDTH>(thread_amax);
            const e8m0_t biased_exponent = compute_shared_biased_exponent<OType>(subwarp_amax);
            const CType block_scale_inverse = exp2f(FP32_EXPONENT_BIAS - static_cast<float>(biased_exponent));

            const int scales_shmem_offset_y = shmem_offset_y + it * STAGE_DIM_Y;
            const int scales_shmem_offset_x = scales_id_x;
            if (thread_id_x % THREADS_PER_SCALE_X == 0) {
                scales_sh[scales_shmem_offset_y][scales_shmem_offset_x] = biased_exponent;
            }

            #pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
                out_c.data.elt[j] = static_cast<OType>(in_compute[j] * block_scale_inverse);
            }
            out_c.store_to(&out_sh[stage][shmem_offset_y][shmem_offset_x]);

            // Wait for shared memory writes to be visible to TMA engine.
            fence_proxy_async_shared_cta();
            __syncthreads();
            // After syncthreads, writes by all threads are visible to TMA engine.

            // Initiate TMA transfer to copy shared memory to global memory
            if (is_master_thread) {
                const int chunk_it_offset_y = chunk_offset_y + it * STAGE_DIM_Y;
                const int chunk_it_offset_x = chunk_offset_x;
                cp_async_bulk_tensor_2d_shared_to_global(
                    reinterpret_cast<const uint64_t*>(&tensor_map_output),
                    chunk_it_offset_x,
                    chunk_it_offset_y,
                    reinterpret_cast<uint64_t*>(&out_sh[stage]));
                // Create a "bulk async-group" out of the previous bulk copy operation.
                cp_async_bulk_commit_group();

                // Wait for TMA transfer to have finished reading shared memory.
                cp_async_bulk_wait_group_read<PREFETCH_STAGES>();
            }
        }
        cp_async_bulk_wait_group_read<0>();
        __syncthreads();

        // Only logical zero lane_id of a subwarp writes to global memory
        if (threadIdx.x < SCALES_PER_CHUNK_Y) {
            using ScalesVec = Vec<e8m0_t, SCALES_PER_CHUNK_X>;
            ScalesVec scales_vec;
            const int y = threadIdx.x;
            const int x = 0;
            scales_vec.load_from(&scales_sh[y][x]);
            const int scale_offset_y = blockIdx.y * SCALES_PER_BLOCK_Y + chunk_Y * SCALES_PER_CHUNK_Y + y;
            const int scale_offset_x = blockIdx.x * SCALES_PER_BLOCK_X + chunk_X * SCALES_PER_CHUNK_X + x;
            const int scale_stride = (cols + MXFP8_BLOCK_DIM_X - 1) / MXFP8_BLOCK_DIM_X;
            const int scale_idx = scale_offset_y * scale_stride + scale_offset_x;
            scales_vec.store_to(&scales[scale_idx]);
        }
        parity ^= 1;
    }


    // Destroy barrier. This invalidates the memory region of the barrier. If
    // further computations were to take place in the kernel, this allows the
    // memory location of the shared memory barrier to be reused.
    if (is_master_thread) {
        #pragma unroll
        for (int it = 0; it < ITERATIONS; ++it) {
            mbarrier_invalid(&mbar[it]);
        }
    }
#endif      // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
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
}

void cast_mxfp8(const Tensor& input, Tensor* output_, cudaStream_t stream) {
    Tensor& output = *output_;

    CheckInputTensor(input, "cast_mxfp8_input");
    NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
    NVTE_CHECK(output.data.shape.size() == 2, "C output must have 2 dimensions.");

    const size_t rows = input.data.shape[0];
    const size_t cols = input.data.shape[1];
    const size_t chunks_Y = DIVUP(rows, CHUNK_DIM_Y);
    const size_t chunks_X = DIVUP(cols, CHUNK_DIM_X);
    const size_t blocks_Y = DIVUP(chunks_Y, CHUNKS_PER_BLOCK_Y);
    const size_t blocks_X = DIVUP(chunks_X, CHUNKS_PER_BLOCK_X);

    const bool isFullTile = (rows % CHUNK_DIM_Y == 0) && (cols % CHUNK_DIM_X == 0);
    NVTE_CHECK(isFullTile, "Only full tiles are supported.");

    NVTE_CHECK(output.scale_inv.dptr != nullptr, "Scaling tensor must be allocated");
    e8m0_t* scales_ptr = reinterpret_cast<e8m0_t*>(output.scale_inv.dptr);

    const dim3 block(THREADS_PER_CHUNK);
    const dim3 grid(blocks_X, blocks_Y);

    TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, IType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(output.data.dtype, OType,
            CUtensorMap tensor_map_input{};
            CUtensorMap tensor_map_output{};

            create_tensor_map<IType>(tensor_map_input, input, rows, cols, SHMEM_DIM_Y, SHMEM_DIM_X);
            create_tensor_map<OType>(tensor_map_output, output, rows, cols, SHMEM_DIM_Y, SHMEM_DIM_X);

            cast_mxfp8_kernel<fp32, IType, OType>
                <<<grid, block, 0, stream>>>
                (tensor_map_input, tensor_map_output, scales_ptr, rows, cols);
        ); // NOLINT(*)
    );     // NOLINT(*)

    NVTE_CHECK_CUDA(cudaGetLastError());
}

namespace detail {

struct Empty {};

__device__ inline fp32 identity(fp32 value, const Empty &) { return value; }

struct DequantizeParam {
  const fp32 *scale_inv;
};

__device__ inline fp32 dequantize_func(fp32 value, const DequantizeParam &param) {
  return value * (*(param.scale_inv));
}

}  // namespace detail

static const int32_t deviceComputeCapability = [](){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    return 10 * deviceProp.major + deviceProp.minor;
}();

bool is_supported_on_CC_100(const Tensor *output) {
    if (deviceComputeCapability < 100) {
        return false;
    }
    if (output->scale_inv.dptr == nullptr) {
        return false;
    }
    const NVTEScalingMode& scaling_mode = output->scaling_mode;
    const bool is_shape_supported = (scaling_mode.x == MXFP8_BLOCK_DIM_Y)
                                    && (scaling_mode.y == MXFP8_BLOCK_DIM_X)
                                    && (scaling_mode.delayed_scaling == 0);
    return is_shape_supported;
}

void fp8_quantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output");

  NVTE_CHECK(!is_fp8_dtype(input.data.dtype), "Input must be in higher precision.");

  NVTE_CHECK(is_fp8_dtype(output->data.dtype), "Output must have FP8 type.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  if (is_supported_on_CC_100(output)) {
      cast_mxfp8(input, output, stream);
  } else if (is_delayed_tensor_scaling(output->scaling_mode)) {
    const size_t N = product(input.data.shape);
    TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
        input.data.dtype, IType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            output->data.dtype, OType, constexpr int nvec = 32 / sizeof(IType);
            VectorizedUnaryKernelLauncher<nvec, detail::Empty, detail::identity>(
                reinterpret_cast<const IType *>(input.data.dptr),
                reinterpret_cast<OType *>(output->data.dptr),
                reinterpret_cast<const fp32 *>(output->scale.dptr),
                reinterpret_cast<fp32 *>(output->amax.dptr), N, {},
                stream););  // NOLINT(*)
    );                      // NOLINT(*)
  } else {
    NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
  }
}

void fp8_dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output");
  NVTE_CHECK(is_fp8_dtype(input.data.dtype), "Input must have FP8 type.");

  NVTE_CHECK(!is_fp8_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  if (is_tensor_scaling(input.scaling_mode)) {
    const size_t N = product(input.data.shape);
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        input.data.dtype, IType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
            output->data.dtype, OType, constexpr int nvec = 32 / sizeof(OType);
            detail::DequantizeParam p;
            p.scale_inv = reinterpret_cast<const fp32 *>(input.scale_inv.dptr);
            VectorizedUnaryKernelLauncher<nvec, detail::DequantizeParam, detail::dequantize_func>(
                reinterpret_cast<const IType *>(input.data.dptr),
                reinterpret_cast<OType *>(output->data.dptr), nullptr, nullptr, N, p,
                stream););  // NOLINT(*)
    );                      // NOLINT(*)
  } else {
    NVTE_ERROR("Not implemented scaling mode: " + to_string(input.scaling_mode) + ".");
  }
}

}  // namespace transformer_engine

void nvte_fp8_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize);
  using namespace transformer_engine;
  fp8_quantize(*reinterpret_cast<const Tensor *>(input), reinterpret_cast<Tensor *>(output),
               stream);
}

void nvte_fp8_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_dequantize);
  using namespace transformer_engine;
  fp8_dequantize(*reinterpret_cast<const Tensor *>(input), reinterpret_cast<Tensor *>(output),
                 stream);
}
