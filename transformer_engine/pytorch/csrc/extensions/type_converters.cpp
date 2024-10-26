/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "pybind.h"
#include "common/util/logging.h"
#include <transformer_engine/transformer_engine.h>

namespace transformer_engine::pytorch::detail {

TensorWrapper NVTETensorFromFloat8Tensor(py::handle tensor, py::handle quantization_params) {
  NVTE_CHECK(quantization_params.is_none() ||
             IsFloat8QParams(quantization_params.ptr()),
             "Expected quantization params of type Float8Params.");
  const at::Tensor &data = tensor.attr("_data").cast<at::Tensor>();
  const at::Tensor &scale_inv = tensor.attr("_scale_inv").cast<at::Tensor>();
  float *scale_inv_dptr = reinterpret_cast<float*>(scale_inv.data_ptr());
  const DType dtype = tensor.attr("_fp8_dtype").cast<DType>();

  const auto& shape = getTensorShape(data);

  float *scale_dptr = nullptr;
  float *amax_dptr = nullptr;

  if (!quantization_params.is_none()) {
    const at::Tensor &scale = quantization_params.attr("scale").cast<at::Tensor>();
    const at::Tensor &amax = quantization_params.attr("amax").cast<at::Tensor>();
    scale_dptr = reinterpret_cast<float*>(scale.data_ptr());
    amax_dptr = reinterpret_cast<float*>(amax.data_ptr());
  }

  return TensorWrapper(data.data_ptr(), shape, dtype,
                       amax_dptr, scale_dptr, scale_inv_dptr);
}

}  // namespace transformer_engine::pytorch::detail
