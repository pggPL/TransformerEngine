/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file gated.cuh
 *  \brief Gated dispatcher.
 */

#ifndef TRANSFORMER_ENGINE_DISPATCH_GATED_CUH_
#define TRANSFORMER_ENGINE_DISPATCH_GATED_CUH_

#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../transpose/transpose.h"
#include "../../utils.cuh"
#include "../fp8/gated_fp8.cuh"
#include "../mxfp8/gated_mxfp8.cuh"

namespace transformer_engine {
namespace dispatch {

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
void quantize_gated_fwd_helper(const NVTETensor nvte_input, NVTETensor nvte_output, ParamOP &p,
                               cudaStream_t stream) {
  const Tensor input = *convertNVTETensorCheck(nvte_input);
  Tensor *output = convertNVTETensorCheck(nvte_output);

  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", /*allow_empty=*/false);

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim() / 2;

  NVTE_CHECK(input.flat_last_dim() % 2 == 0,
             "Wrong input shape. Expected (after flattening) last dimension to be even, ", "got [",
             input.flat_first_dim(), ", ", input.flat_last_dim(), "].");
  NVTE_CHECK(output->flat_last_dim() == cols,
             "Wrong output shape. Expected (after flattening) [*, ", cols, "], got [",
             output->flat_first_dim(), ", ", output->flat_last_dim(), "].");

  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");

  switch (output->scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      const bool use_tma_kernels =
          (cols % 32 == 0) && is_supported_by_CC_100() &&
          fp8::cast_gated_tma_fits_device</*IS_BWD=*/false, ParamOP, ActOP, nullptr>(
              input.dtype(), output->dtype());
      if (use_tma_kernels) {
        Tensor dummy_grad_tensor;
        fp8::cast_gated_tma</*IS_BWD=*/false, ParamOP, ActOP, nullptr>(input, dummy_grad_tensor,
                                                                       output, p, stream);
      } else {
        fp8::cast_gated_fwd<ParamOP, ActOP>(input, output, p, stream);
      }
      if (is_fp8_dtype(output->dtype()) && output->has_columnwise_data()) {
        // FP8 kernel only populates row-wise data, so perform
        // transpose separately if needed
        Tensor transpose_in, transpose_out, dummy;
        transpose_in.scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
        transpose_in.data.dptr = output->data.dptr;
        transpose_in.data.shape = {output->flat_first_dim(), output->flat_last_dim()};
        transpose_in.data.dtype = output->data.dtype;
        transpose_out.scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
        transpose_out.data.dptr = output->columnwise_data.dptr;
        transpose_out.data.shape = {output->flat_last_dim(), output->flat_first_dim()};
        transpose_out.data.dtype = output->data.dtype;
        detail::transpose(transpose_in, /*noop=*/dummy, &transpose_out, stream);
      }
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      NVTE_CHECK(cols % 32 == 0,
                 "Invalid input shape. Expected the last dimension to be "
                 "divisible by 32, but got ",
                 cols, ".");
      if (output->has_data()) {
        NVTE_CHECK(is_fp8_dtype(output->data.dtype),
                   "The type of the output tensor should be FP8.");
      }
      if (output->has_columnwise_data()) {
        NVTE_CHECK(is_fp8_dtype(output->columnwise_data.dtype),
                   "The type of the columnwise output tensor should be FP8.");
      }
      NVTE_CHECK(is_supported_by_CC_100(),
                 "Gated FWD NVTE_MXFP8_1D_SCALING is only supported on SM 10.0+");
      Tensor dummy_grad_tensor;
      mxfp8::quantize_gated</*IS_BWD=*/false, ParamOP, ActOP, nullptr>(input, dummy_grad_tensor,
                                                                       output, p, stream);
      break;
    }
    default:
      NVTE_ERROR("Not supported scaling mode: " + to_string(output->scaling_mode) + ".");
  }
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void quantize_gated_bwd_helper(const NVTETensor nvte_grad, const NVTETensor nvte_gated_input,
                               NVTETensor nvte_output, ParamOP &p, cudaStream_t stream) {
  const Tensor &grad = *(convertNVTETensorCheck(nvte_grad));
  const Tensor gated_input = *convertNVTETensorCheck(nvte_gated_input);
  Tensor *output = convertNVTETensorCheck(nvte_output);

  CheckInputTensor(grad, "grad");
  CheckInputTensor(gated_input, "gated_input");
  CheckOutputTensor(*output, "output", /*allow_empty=*/false);

  NVTE_CHECK(gated_input.flat_last_dim() % 2 == 0, "Number of columns must be even, but got ",
             gated_input.flat_last_dim(), ".");

  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;

  NVTE_CHECK(!is_fp8_dtype(grad.dtype()), "Grad input must be in higher precision.");
  NVTE_CHECK(grad.dtype() == gated_input.dtype(), "Types of both inputs must match.");

  NVTE_CHECK(grad.flat_first_dim() == rows,
             "Wrong Grad shape. Expected first dimension (after flattening) [", rows, ", *], got [",
             grad.flat_first_dim(), ", ", grad.flat_last_dim(), "].");
  NVTE_CHECK(grad.flat_last_dim() == cols,
             "Wrong Grad shape. Expected last dimension (after flattening) [", cols, ", *], got [",
             grad.flat_first_dim(), ", ", grad.flat_last_dim(), "].");

  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");

  NVTE_CHECK(output->flat_first_dim() == rows, "Wrong output shape. Expected (after flattening) [",
             rows, ", *], got [", output->flat_first_dim(), ", ", output->flat_last_dim(), "].");
  NVTE_CHECK(output->flat_last_dim() == cols * 2,
             "Wrong output shape. Expected (after flattening) [*, ", cols * 2, "], got [",
             output->flat_first_dim(), ", ", output->flat_last_dim(), "].");
  NVTE_CHECK(gated_input.shape() == output->shape(),
             "Gated input and output shapes must match. Input shape: ", gated_input.shape(),
             ", output shape: ", output->shape(), ".");

  switch (output->scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      const bool use_tma_kernels =
          (cols % 32 == 0) && is_supported_by_CC_100() &&
          fp8::cast_gated_tma_fits_device</*IS_BWD=*/true, ParamOP, ActOP, DActOP>(
              gated_input.dtype(), output->dtype());
      if (use_tma_kernels) {
        fp8::cast_gated_tma</*IS_BWD=*/true, ParamOP, ActOP, DActOP>(gated_input, grad, output, p,
                                                                     stream);
      } else {
        fp8::cast_gated_bwd<ParamOP, ActOP, DActOP>(gated_input, grad, output, p, stream);
      }
      if (is_fp8_dtype(output->dtype()) && output->has_columnwise_data()) {
        // FP8 kernel only populates row-wise data, so perform
        // transpose separately if needed
        Tensor transpose_in, transpose_out, dummy;
        transpose_in.scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
        transpose_in.data.dptr = output->data.dptr;
        transpose_in.data.shape = {output->flat_first_dim(), output->flat_last_dim()};
        transpose_in.data.dtype = output->data.dtype;
        transpose_out.scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
        transpose_out.data.dptr = output->columnwise_data.dptr;
        transpose_out.data.shape = {output->flat_last_dim(), output->flat_first_dim()};
        transpose_out.data.dtype = output->data.dtype;
        detail::transpose(transpose_in, /*noop=*/dummy, &transpose_out, stream);
      }
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      NVTE_CHECK(cols % 32 == 0,
                 "Invalid input shape. Expected the last dimension to be "
                 "divisible by 32, but got ",
                 cols, ".");
      if (output->has_data()) {
        NVTE_CHECK(is_fp8_dtype(output->data.dtype),
                   "The type of the output tensor should be FP8.");
      }
      if (output->has_columnwise_data()) {
        NVTE_CHECK(is_fp8_dtype(output->columnwise_data.dtype),
                   "The type of the columnwise output tensor should be FP8.");
      }
      NVTE_CHECK(is_supported_by_CC_100(),
                 "Gated BWD NVTE_MXFP8_1D_SCALING is only supported on SM 10.0+");

      mxfp8::quantize_gated</*IS_BWD=*/true, ParamOP, ActOP, DActOP>(gated_input, grad, output, p,
                                                                     stream);
      break;
    }
    default:
      NVTE_ERROR("Not supported scaling mode: " + to_string(output->scaling_mode) + ".");
  }
}

template <bool Backward, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &) = nullptr>
void group_gated_act_helper(const NVTEGroupedTensor input_handle,
                            const NVTEGroupedTensor grad_handle, NVTEGroupedTensor output_handle,
                            const ParamOP &params, cudaStream_t stream) {
  const auto *input = convertNVTEGroupedTensorCheck(input_handle);
  auto *output = convertNVTEGroupedTensorCheck(output_handle);
  NVTE_CHECK(input->logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "Grouped GLU requires rank-2 logical shapes.");
  NVTE_CHECK(input->all_same_last_dim() && output->all_same_last_dim(),
             "Grouped GLU requires a uniform feature dimension.");
  const size_t capacity = input->logical_shape.data[0];
  const size_t input_width = input->logical_shape.data[1];
  NVTE_CHECK(input_width > 0 && input_width % 2 == 0,
             "Grouped GLU input width must be positive and even.");
  const size_t hidden = input_width / 2;
  NVTE_CHECK(output->num_tensors == input->num_tensors &&
                 output->logical_shape.data[0] == capacity &&
                 output->logical_shape.data[1] == (Backward ? input_width : hidden),
             "Grouped GLU output shape or group count is inconsistent with input.");
  const DType dtype = input->data.dtype;
  NVTE_CHECK(dtype == DType::kFloat32 || dtype == DType::kFloat16 || dtype == DType::kBFloat16,
             "Grouped GLU supports float32, float16 and bfloat16.");
  NVTE_CHECK(output->data.dtype == dtype, "Grouped GLU input/output dtypes must match.");
  NVTE_CHECK(input->data.numel() >= capacity * input_width &&
                 output->data.numel() >= capacity * (Backward ? input_width : hidden),
             "Grouped GLU input/output storage is smaller than its logical shape.");
  if (input->first_dims.has_data()) {
    NVTE_CHECK(
        input->first_dims.dtype == DType::kInt64 && input->first_dims.numel() == input->num_tensors,
        "Grouped GLU first_dims must contain one int64 count per group.");
  }
  const auto *grad = Backward ? convertNVTEGroupedTensorCheck(grad_handle) : nullptr;
  if constexpr (Backward) {
    NVTE_CHECK(grad->logical_shape.ndim == 2 && grad->all_same_last_dim() &&
                   grad->logical_shape.data[0] == capacity &&
                   grad->logical_shape.data[1] == hidden &&
                   grad->num_tensors == input->num_tensors && grad->data.dtype == dtype,
               "Grouped GLU gradient shape, dtype or group count is inconsistent with input.");
    NVTE_CHECK(grad->data.numel() >= capacity * hidden,
               "Grouped GLU gradient storage is smaller than its logical shape.");
  }
  if (capacity == 0) return;
  NVTE_CHECK(input->data.has_data() && output->data.has_data(),
             "Grouped GLU input/output data must be allocated.");
  if constexpr (Backward) NVTE_CHECK(grad->data.has_data(), "Grouped GLU grad is not allocated.");
  const auto *sizes = static_cast<const int64_t *>(input->first_dims.dptr);
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      dtype, T, constexpr int nvec = 32 / sizeof(T); if constexpr (Backward) {
        DGatedActivationKernelLauncher<nvec, fp32, ParamOP, ActOP, DActOP, true>(
            static_cast<const T *>(grad->data.dptr), static_cast<const T *>(input->data.dptr),
            static_cast<T *>(output->data.dptr), nullptr, nullptr, nullptr, capacity, hidden,
            params, stream, sizes, input->num_tensors);
      } else {
        GatedActivationKernelLauncher<nvec, fp32, ParamOP, ActOP, true>(
            static_cast<const T *>(input->data.dptr), static_cast<T *>(output->data.dptr), nullptr,
            nullptr, nullptr, capacity, hidden, params, stream, sizes, input->num_tensors);
      });
}

}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DISPATCH_GATED_CUH_
