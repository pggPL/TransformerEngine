/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file activation_template.h
 *  \brief Activation functions template.
 */

#ifndef TRANSFORMER_ENGINE_ACTIVATION_TEMPLATE_H_
#define TRANSFORMER_ENGINE_ACTIVATION_TEMPLATE_H_

#include <cuda_runtime.h>
#include <transformer_engine/activation.h>

#include "../common.h"
#include "../util/cast_kernels.cuh"
#include "../util/vectorized_pointwise.h"
#include "../util/math.h"

namespace transformer_engine {

template <typename ComputeType, typename Param, ComputeType (*OP)(ComputeType, const Param &)>
void act_fn(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = true;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor activation_input = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, OP>
      (input, activation_input, nullptr, output, dbias, workspace, stream);
}

template <typename ComputeType, typename Param, ComputeType (*OP)(ComputeType, const Param &)>
void dact_fn(const NVTETensor grad, const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, OP>
      (input, grad, nullptr, output, dbias, workspace, stream);
}

template <typename ComputeType, typename Param, ComputeType (*OP)(ComputeType, const Param &)>
void gated_act_fn(const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(input, "gated_act_input");
  CheckOutputTensor(*output, "gated_act_output");
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output->data.shape.size() == 2, "Output must have 2 dimensions.");
  NVTE_CHECK(input.data.shape[0] == output->data.shape[0],
             "Input shape[0] must be equal to output shape[0].");
  NVTE_CHECK(input.data.shape[1] == output->data.shape[1] * 2,
             "Input shape[1] must be 2x larger than output shape[1].");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->data.dtype, OType,
          if (!is_fp8_dtype(output->data.dtype) ||
              is_delayed_tensor_scaling(output->scaling_mode)) {
            constexpr int nvec = 32 / sizeof(IType);
            GatedActivationKernelLauncher<nvec, ComputeType, Param, OP>(
                reinterpret_cast<const IType *>(input.data.dptr),
                reinterpret_cast<OType *>(output->data.dptr),
                reinterpret_cast<const ComputeType *>(output->scale.dptr),
                reinterpret_cast<ComputeType *>(output->amax.dptr),
                reinterpret_cast<ComputeType *>(output->scale_inv.dptr), output->data.shape[0],
                output->data.shape[1], {}, stream);
          } else {
            NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
          });  // NOLINT(*)
  );           // NOLINT(*)
}

template <typename ComputeType, typename Param, ComputeType (*OP1)(ComputeType, const Param &),
          ComputeType (*OP2)(ComputeType, const Param &)>
void dgated_act_fn(const Tensor &grad, const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(grad, "dgated_act_grad");
  CheckInputTensor(input, "dgated_act_input");
  CheckOutputTensor(*output, "dgated_act_output");
  NVTE_CHECK(grad.data.shape.size() == 2, "Grad must have 2 dimensions.");
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output->data.shape.size() == 2, "Output must have 2 dimensions.");
  NVTE_CHECK(output->data.shape[0] == grad.data.shape[0],
             "Output shape[0] must be equal to grad shape[0].");
  NVTE_CHECK(output->data.shape[1] == grad.data.shape[1] * 2,
             "Output shape[1] must be 2x larger than grad shape[1].");
  NVTE_CHECK(input.data.shape == output->data.shape, "Input and output shapes must match.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->data.dtype, OType,
          if (!is_fp8_dtype(output->data.dtype) ||
              is_delayed_tensor_scaling(output->scaling_mode)) {
            constexpr int nvec = 32 / sizeof(IType);
            DGatedActivationKernelLauncher<nvec, ComputeType, Param, OP1, OP2>(
                reinterpret_cast<const IType *>(grad.data.dptr),
                reinterpret_cast<const IType *>(input.data.dptr),
                reinterpret_cast<OType *>(output->data.dptr), grad.data.shape[0],
                grad.data.shape[1], {}, stream);
          } else {
            NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
          });  // NOLINT(*)
  );           // NOLINT(*)
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_ACTIVATION_TEMPLATE_H_
