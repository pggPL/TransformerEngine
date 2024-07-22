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
 *  \param[in]     input     Input tensor to be cast.
 *  \param[in,out] output    Output FP8 tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_fp8_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Cast tensor from FP8.
 *
 *  \param[in]     input     Input tensor to be cast.
 *  \param[out]    output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_fp8_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Cast tensor to MXFP8 format
 * Takes the input tensor (datatype FP32, BF16 or FP16) and produces two output tensors: 
 * the tensor of scaled elements (datatype FP8_E4M3 or FP8_E5M2), and the tensor of 
 * corresponding scaling factors (FP8_E8M0).
 * The algorithm implements the OCP Microscaling Formats (MX) Specification v1.0:
 * (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
 * Currently, the casting supports 1D scaling factors of 32 contiguous elements, i.e.,
 * for every row of the input tensor, algorithm processes elements in groups of 32,
 * dynamically rescaling them per the absolute maximum value found in that group and 
 * the maximum absolute value which can be represented by the output datatype to fully 
 * use its dynamic range. The resulting values are then cast to the specified output 
 * datatype.
 * NOTES:
 * - Intermediate computations and scaling are implemented in FP32 with saturation. 
 * - Scaled values smaller than the minimum normalized value of the output datatype 
 * are flushed to zero.
 * - The scaling factors are computed by extracting the corresponding bits of the 
 * exponent, which may be replaced by the PTX ISA 8.6 instructions specific to sm_100a.
 * - The row dimension of the input tensor should be a multiple of 32  
 * - The column dimension of the input tensor should be a multiple of 128  
 *
 *  \param[in]     input     Input tensor to be cast.
 *  \param[out]    output    Output cast tensor.
 *  \param[out]    scales    Output scales tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_cast_mxfp8(const NVTETensor input,
                     NVTETensor output,
                     NVTETensor block_scales,
                     cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CAST_H_
