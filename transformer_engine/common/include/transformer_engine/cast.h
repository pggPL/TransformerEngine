/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast.h
 *  \brief Functions to cast to/from FP8.
 */

#ifndef TRANSFORMER_ENGINE_CAST_H_
#define TRANSFORMER_ENGINE_CAST_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Cast tensor to FP8.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in,out] output           Output FP8 tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_fp8_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Cast tensor to FP8 along both dimensions.
 *  Produces two sets of output data:
 *  1) Scaled rows + row-wise scaling factors, AND
 *  2) Scaled columns + column-wise scaling factors
 *  \param[in]     input                Input tensor to be cast.
 *  \param[in,out] output_rowwise       Output FP8 tensor scaled along rows
 *  \param[in,out] output_columnwise    Output FP8 tensor scaled along columns
 *  \param[in]     stream               CUDA stream used for the operation.
 */
void nvte_fp8_quantize_x2(const NVTETensor input, NVTETensor output_rowwise,
                          NVTETensor output_columnwise, cudaStream_t stream);

/*! \brief Cast tensor from FP8.
 *
 *  \param[in]     input     Input tensor to be cast.
 *  \param[out]    output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_fp8_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CAST_H_
