/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast.h
 *  \brief Functions to cast to/from FP8/MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_CAST_H_
#define TRANSFORMER_ENGINE_CAST_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Casts the tensor to FP8
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_cast(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Cast tensor to FP8. Additionally, reduce the input along columns.
 *
 * This function casts the input and produces 2 results:
 *  - `output` is the result of the cast including the scaling factors
 *  - `dbias` is the result of the reduction of the input along columns.
 *
 *  Calling this function with workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]      input            Input tensor to be cast.
 *  \param[in,out]  output           Output MXFP8 tensor.
 *  \param[out]     dbias            Result of the reduction of the input along columns.
 *  \param[out]     workspace        Workspace tensor.
 *  \param[in]      stream           CUDA stream used for the operation.
 */
void nvte_cast_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                             NVTETensor workplace, cudaStream_t stream);

/*! \brief Compute backward of ActLU operation on the input, then cast to FP8.
 *         Additionally, reduce the result of the ActLU backward along columns.
 *         Supported by the devices with the compute capability 10.0 or newer.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 *
 *  Supported activations: GeLU, SiLU, ReLU, QuickGeLU, SquaredReLU
 */
void nvte_cast_dbias_dgelu(const NVTETensor input, const NVTETensor act_input,
                           NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                           cudaStream_t stream);
void nvte_cast_dbias_dsilu(const NVTETensor input, const NVTETensor act_input,
                           NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                           cudaStream_t stream);
void nvte_cast_dbias_drelu(const NVTETensor input, const NVTETensor act_input,
                           NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                           cudaStream_t stream);
void nvte_cast_dbias_dqgelu(const NVTETensor input, const NVTETensor act_input,
                            NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                            cudaStream_t stream);
void nvte_cast_dbias_dsrelu(const NVTETensor input, const NVTETensor act_input,
                            NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                            cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CAST_H_
