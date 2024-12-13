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
#include <limits>

#include "../common.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "cast_kernels.cuh"
#include "math.h"
#include "ptx.cuh"
#include "transformer_engine/transpose.h"
#include "transformer_engine/activation.h"
#include "../transpose/cast_transpose.h"

namespace transformer_engine {

template <typename IType>
void reduce_dbias(const float *workspace_ptr, Tensor *dbias, const size_t rows, const size_t cols,
                  cudaStream_t stream) {
  constexpr int reduce_dbias_store_bytes = 8;  // stg.64
  constexpr int reduce_dbias_nvec = reduce_dbias_store_bytes / sizeof(IType);

  NVTE_CHECK(cols % reduce_dbias_nvec == 0, "Unsupported shape.");
  const size_t reduce_dbias_num_blocks = DIVUP(cols, DBIAS_THREADS_PER_BLOCK * reduce_dbias_nvec);

  reduce_dbias_kernel<reduce_dbias_nvec, IType>
      <<<reduce_dbias_num_blocks, DBIAS_THREADS_PER_BLOCK, 0, stream>>>(
          reinterpret_cast<IType *>(dbias->data.dptr), workspace_ptr, rows, cols);
}

void cast_fp8_1D(const Tensor &input, Tensor *output, cudaStream_t stream) {
  const size_t N = product(input.data.shape);

  const bool isFullTile = (N % ELEMS_PER_BLOCK == 0);
  NVTE_CHECK(isFullTile, "Only full tiles are supported.");

  const size_t chunks = DIVUP(N, CHUNK_SIZE);
  const size_t blocks = DIVUP(chunks, CHUNKS_PER_BLOCK);

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  const float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block(THREADS_PER_BLOCK);
  const dim3 grid(blocks);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(output->dtype(), OType,
          const IType *input_ptr = reinterpret_cast<const IType *>(input.data.dptr);
          OType *output_ptr = reinterpret_cast<OType *>(output->data.dptr);

          cast_fp8_1D_kernel<IType, OType><<<grid, block, 0, stream>>>(
              input_ptr, output_ptr, amax_ptr, scale_inv_ptr, scale_ptr, N););  // NOLINT(*)
  );                                                                            // NOLINT(*)
}

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void cast_fp8_2D(const Tensor &input, const Tensor *act_input, Tensor *output, Tensor *dbias,
                 Tensor *workspace, cudaStream_t stream) {
  checkCuDriverContext(stream);
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");

  const size_t rows = input.data.shape[0];
  const size_t cols = input.data.shape[1];
  const size_t chunks_Y = DIVUP(rows, FP8_CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, FP8_CHUNK_DIM_X);
  const size_t blocks_Y = chunks_Y;
  const size_t blocks_X = chunks_X;

  const size_t dbias_rows = blocks_Y;
  const size_t dbias_cols = cols;

  const bool isFullTile = (rows % FP8_CHUNK_DIM_Y == 0) && (cols % FP8_CHUNK_DIM_X == 0);
  NVTE_CHECK(isFullTile, "Only full tiles are supported.");

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
  float *const workspace_ptr = IS_DBIAS ? reinterpret_cast<float *>(workspace->data.dptr) : nullptr;
  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block(FP8_THREADS_PER_CHUNK);
  const dim3 grid(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->data.dtype, OType,

          alignas(64) CUtensorMap tensor_map_input{};
          alignas(64) CUtensorMap tensor_map_act_input{};
          alignas(64) CUtensorMap tensor_map_output{};

          create_2D_tensor_map(tensor_map_input, input.data, rows, cols, FP8_SHMEM_DIM_Y,
                               FP8_SHMEM_DIM_X, sizeof(IType));

          if constexpr (IS_DACT) {
            create_2D_tensor_map(tensor_map_act_input, act_input->data, rows, cols,
                                 FP8_SHMEM_DIM_Y, FP8_SHMEM_DIM_X, sizeof(IType));
          }
          create_2D_tensor_map(tensor_map_output, output->data, rows, cols, FP8_SHMEM_DIM_Y,
                               FP8_SHMEM_DIM_X, sizeof(OType));

          cast_fp8_2D_kernel<IS_DBIAS, IS_DACT, ParamOP, OP, IType, OType>
          <<<grid, block, 0, stream>>>(tensor_map_input, tensor_map_act_input, tensor_map_output,
                                       workspace_ptr, amax_ptr, scale_inv_ptr, scale_ptr, rows,
                                       cols);

          if constexpr (IS_DBIAS) {
            reduce_dbias<IType>(workspace_ptr, dbias, dbias_rows, dbias_cols, stream);
          });  // NOLINT(*)
  );           // NOLINT(*)
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void mxfp8_quantize(const Tensor &input, const Tensor *act_input, Tensor *output,
                    Tensor *dbias, Tensor *workspace, cudaStream_t stream) {
  bool use_rowwise_scaling = output->has_data();
  bool use_colwise_scaling = output->has_columnwise_data();
  checkCuDriverContext(stream);
  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");
  const auto& input_shape = input.data.shape;
  NVTE_CHECK(input_shape.size() >= 2, "Input must have at least 2 dimensions.");

  // TODO: Make more general
  const size_t scale_dim_X_rowwise = use_rowwise_scaling ? 32 : 1;
  const size_t scale_dim_Y_colwise = use_colwise_scaling ? 32 : 1;

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  const size_t chunks_Y = DIVUP(rows, MXFP8_CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, MXFP8_CHUNK_DIM_X);
  const size_t blocks_Y = DIVUP(chunks_Y, MXFP8_CHUNKS_PER_BLOCK_Y);
  const size_t blocks_X = DIVUP(chunks_X, MXFP8_CHUNKS_PER_BLOCK_X);
  const size_t scale_stride_rowwise = DIVUP(cols, scale_dim_X_rowwise);
  const size_t scale_stride_colwise = cols;

  const bool isFullTile = (rows % MXFP8_CHUNK_DIM_Y == 0) && (cols % MXFP8_CHUNK_DIM_X == 0);
  NVTE_CHECK(isFullTile, "Only full tiles are supported.");

  e8m0_t *const scales_rowwise_ptr =
      use_rowwise_scaling ? reinterpret_cast<e8m0_t *>(output->scale_inv.dptr) : nullptr;
  e8m0_t *const scales_colwise_ptr =
      use_colwise_scaling ? reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr) : nullptr;
  const size_t dbias_rows = blocks_Y;
  const size_t dbias_cols = cols;

  if constexpr (IS_DBIAS) {
    NVTE_CHECK(dbias->data.dtype == input.dtype(), "DBias must have the same type as input.");
    NVTE_CHECK(dbias->data.shape == std::vector<size_t>{cols}, "Wrong shape of DBias.");
    NVTE_CHECK(workspace != nullptr, "Workspace must be a tensor.");

    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {dbias_rows, dbias_cols};
      workspace->data.dtype = DType::kFloat32;
      return;
    }
  }

  float *const workspace_ptr = IS_DBIAS ? reinterpret_cast<float *>(workspace->data.dptr) : nullptr;
  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);

  const dim3 block(MXFP8_THREADS_PER_CHUNK);
  const dim3 grid(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(scale_dim_Y_colwise, SCALE_DIM_Y,
      TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(scale_dim_X_rowwise, SCALE_DIM_X,
          TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.dtype(), IType,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(output->dtype(), OType,

                  alignas(64) CUtensorMap tensor_map_input{};
                  alignas(64) CUtensorMap tensor_map_act_input{};
                  alignas(64) CUtensorMap tensor_map_output_rowwise{};
                  alignas(64) CUtensorMap tensor_map_output_colwise{};

                  create_2D_tensor_map(tensor_map_input, input.data,
                                       rows, cols,
                                       MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X,
                                       sizeof(IType));

                  if constexpr (IS_DACT) {
                    create_2D_tensor_map(tensor_map_act_input, act_input->data,
                                         rows, cols,
                                         MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X,
                                         sizeof(IType));
                  }
                  if (use_rowwise_scaling) {
                    create_2D_tensor_map(tensor_map_output_rowwise, output->data,
                                         rows, cols,
                                         MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X,
                                         sizeof(OType));
                  }
                  if (use_colwise_scaling) {
                    create_2D_tensor_map(tensor_map_output_colwise, output->columnwise_data,
                                         rows, cols,
                                         MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X,
                                         sizeof(OType));
                  }

                  cast_mxfp8_2D_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType, SCALE_DIM_Y,
                                       SCALE_DIM_X><<<grid, block, 0, stream>>>(
                      tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                      tensor_map_output_colwise, scales_rowwise_ptr, scales_colwise_ptr,
                      workspace_ptr, amax_ptr, rows, cols,
                      scale_stride_rowwise, scale_stride_colwise);

                  if constexpr (IS_DBIAS) {
                    reduce_dbias<IType>(workspace_ptr, dbias, dbias_rows, dbias_cols, stream);
                  });  // NOLINT(*)
          );           // NOLINT(*)
      );               // NOLINT(*)
  );                   // NOLINT(*)
}

namespace detail {

using Empty = transformer_engine::Empty;

__device__ inline float identity(float value, const Empty &) { return value; }

struct DequantizeParam {
  const float *scale_inv;
};

__device__ inline float dequantize_func(float value, const DequantizeParam &param) {
  return value * (*(param.scale_inv));
}

}  // namespace detail

// Supported by the Arch < 10.0
void CastVectorizedUnaryKernelLauncher(const Tensor &input, Tensor *output,
                                       Tensor *workspace, cudaStream_t stream) {
  const size_t N = product(input.data.shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->data.dtype, OType,

          constexpr int nvec = 32 / sizeof(IType);
          VectorizedUnaryKernelLauncher<nvec, detail::Empty, detail::identity>(
              reinterpret_cast<const IType *>(input.data.dptr),
              reinterpret_cast<OType *>(output->data.dptr),
              reinterpret_cast<const fp32 *>(output->scale.dptr),
              reinterpret_cast<fp32 *>(output->amax.dptr),
              reinterpret_cast<fp32 *>(output->scale_inv.dptr), N, {},
              stream););  // NOLINT(*)
  );                      // NOLINT(*)
}

// Supported by the Arch >= 10.0
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void fp8_quantize_arch_ge_100(const Tensor &input, const Tensor *act_input, Tensor *output,
                              Tensor *dbias, Tensor *workspace, cudaStream_t stream)
{
  if (is_mxfp_scaling(output->scaling_mode)) {
    // MXFP8 Scaling
    mxfp8_quantize<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>(
        input, act_input, output, dbias, workspace, stream);
  } else if (is_delayed_tensor_scaling(output->scaling_mode)) {
    // Regular FP8 Scaling
    if (!IS_DBIAS && !IS_DACT) {
      const size_t N = product(input.data.shape);
      const bool isFullTile = (N % ELEMS_PER_BLOCK == 0);
      if (isFullTile) {
        // Aligned
        cast_fp8_1D(input, output, stream);
      } else {
        // Unaligned
        CastVectorizedUnaryKernelLauncher(input, output, workspace, stream);
      }
    } else {
      cast_fp8_2D<IS_DBIAS, IS_DACT, ParamOP, OP>(input, act_input, output, dbias, workspace,
                                                  stream);
    }
  } else {
    NVTE_ERROR("Not implemented on Arch >= 10.0: " + to_string(output->scaling_mode) + ".");
  }
}

// Supported by the Arch < 10.0
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
void fp8_quantize_arch_l_100(const Tensor &input, const Tensor *act_input, Tensor *output,
                             Tensor *dbias, Tensor *workspace, cudaStream_t stream) {
  if (is_delayed_tensor_scaling(output->scaling_mode) && !IS_DBIAS && !IS_DACT && !IS_ACT) {
    CastVectorizedUnaryKernelLauncher(input, output, workspace, stream);
  } else {
    NVTE_ERROR("Not implemented on Arch < 10.0: " + to_string(output->scaling_mode) + ".");
  }
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void fp8_quantize(const Tensor &input, const Tensor *act_input, Tensor *output, Tensor *dbias,
                  Tensor *workspace, cudaStream_t stream) {
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output");

  if constexpr (IS_DBIAS) {
    NVTE_CHECK(dbias != nullptr);
    CheckOutputTensor(*dbias, "dbias");
  }
  if constexpr (IS_DACT) {
    NVTE_CHECK(act_input != nullptr);
    CheckInputTensor(*act_input, "activation_input");
    NVTE_CHECK(input.dtype() == act_input->dtype(), "Types of both inputs must match.");
    NVTE_CHECK(input.data.shape == act_input->data.shape, "Shapes of both inputs must match.");
  }

  NVTE_CHECK(!is_fp8_dtype(input.dtype()), "Input must be in higher precision.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");

  // Supported by the Arch >= 10.0
  if (is_supported_by_CC_100()) {
    fp8_quantize_arch_ge_100<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>
        (input, act_input, output, dbias, workspace, stream);
  } else {
    // Supported by the Arch < 10.0
    fp8_quantize_arch_l_100<IS_DBIAS, IS_DACT, IS_ACT>
      (input, act_input, output, dbias, workspace, stream);
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
                reinterpret_cast<OType *>(output->data.dptr), nullptr, nullptr, nullptr, N, p,
                stream););  // NOLINT(*)
    );                      // NOLINT(*)
  } else {
    NVTE_ERROR("Not implemented scaling mode: " + to_string(input.scaling_mode) + ".");
  }
}

namespace detail {

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void quantize_helper(const NVTETensor input,
                     const NVTETensor activation_input,
                     const NVTETensor noop,
                     NVTETensor output,
                     NVTETensor dbias,
                     NVTETensor workspace,
                     cudaStream_t stream) {
  const auto& input_tensor = *(reinterpret_cast<const Tensor*>(input));
  auto output_tensor = reinterpret_cast<Tensor*>(output);
  const auto activation_tensor = reinterpret_cast<const Tensor*>(activation_input);
  auto dbias_tensor = reinterpret_cast<Tensor*>(dbias);
  auto workspace_tensor = reinterpret_cast<Tensor*>(workspace);

  if (is_tensor_scaling(output_tensor->scaling_mode)) {
    if (output_tensor->has_columnwise_data()) {
      NVTE_CHECK(output_tensor->has_data(),
                 "Quantizing in only the columnwise direction not supported yet!");
      // TODO: Change to calling some C++ function and finish
      // cast+transpose
      // TODO: handle noop
      if constexpr (!IS_DBIAS && !IS_DACT && !IS_ACT) {
        const auto noop_tensor = noop != nullptr ? *(reinterpret_cast<const Tensor*>(noop))
                                                 : Tensor();
        cast_transpose(input_tensor, noop_tensor, output_tensor, stream);
      }
      if constexpr (IS_DBIAS && !IS_ACT) {
        cast_transpose_fused<IS_DBIAS, IS_DACT, float, ParamOP, OP>(input_tensor, activation_tensor,
                                                                    output_tensor, dbias_tensor,
                                                                    workspace_tensor, stream);
      }
      if constexpr (!IS_DBIAS && (IS_DACT || IS_ACT)) {
        NVTE_ERROR("Not implemented yet!");
      }
    } else if (output_tensor->has_data()) {
      fp8_quantize<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>(
          input_tensor,
          activation_tensor,
          output_tensor,
          dbias_tensor,
          workspace_tensor, stream);
    }
  } else {
    // MX scaling
    mxfp8_quantize<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>(input_tensor, activation_tensor,
                                                           output_tensor, dbias_tensor,
                                                           workspace_tensor, stream);
  }
}

}  // namespace detail

}  // namespace transformer_engine

void nvte_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = false;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor activation_input = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, nullptr>
      (input, activation_input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_gelu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_gelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = true;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor activation_input = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, gelu<fp32,fp32>>
      (input, activation_input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_qgelu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_qgelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = true;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor activation_input = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, qgelu<fp32,fp32>>
      (input, activation_input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_silu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_silu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = true;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor activation_input = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, silu<fp32,fp32>>
      (input, activation_input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_relu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_relu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = true;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor activation_input = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, relu<fp32,fp32>>
      (input, activation_input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_srelu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_srelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = true;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor activation_input = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, srelu<fp32,fp32>>
      (input, activation_input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_noop(const NVTETensor input, NVTETensor output, NVTETensor noop, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_noop);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = false;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor activation_input = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, nullptr>
      (input, activation_input, noop, output, dbias, workspace, stream);
}

void nvte_quantize_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                             NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = false;
  constexpr const NVTETensor activation_input = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, nullptr>
      (input, activation_input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_dbias_dgelu(const NVTETensor input, const NVTETensor activation_input,
                                   NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                   cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_dgelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, dgelu<fp32, fp32>>
      (input, activation_input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_dbias_dsilu(const NVTETensor input, const NVTETensor activation_input,
                                   NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                   cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_dsilu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, dsilu<fp32, fp32>>
      (input, activation_input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_dbias_drelu(const NVTETensor input, const NVTETensor activation_input,
                                   NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                   cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_drelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, drelu<fp32, fp32>>
      (input, activation_input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_dbias_dqgelu(const NVTETensor input, const NVTETensor activation_input,
                                    NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                    cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_dqgelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, dqgelu<fp32, fp32>>
      (input, activation_input, nullptr, output, dbias, workspace, stream);
}

void nvte_quantize_dbias_dsrelu(const NVTETensor input, const NVTETensor activation_input,
                                    NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                    cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_dbias_dsrelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, dsrelu<fp32, fp32>>
      (input, activation_input, nullptr, output, dbias, workspace, stream);
}

void nvte_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dequantize);
  using namespace transformer_engine;
  fp8_dequantize(*reinterpret_cast<const Tensor *>(input), reinterpret_cast<Tensor *>(output),
                 stream);
}
