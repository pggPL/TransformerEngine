/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "pybind.h"
#include <transformer_engine/transformer_engine.h>

namespace transformer_engine::pytorch::detail {

TensorWrapper NVTETensorFromFloat8Tensor(py::handle tensor, QuantizationParams* quantization_params) {
  const at::Tensor &data = tensor.attr("_data").cast<at::Tensor>();
  const at::Tensor &scale_inv = tensor.attr("_scale_inv").cast<at::Tensor>();
  float *scale_inv_dptr = reinterpret_cast<float*>(scale_inv.data_ptr());
  const DType dtype = tensor.attr("_fp8_dtype").cast<DType>();

  const auto& shape = getShape(getTensorShape(data));

  bool transpose_valid = !tensor.attr("_transpose_invalid").cast<bool>();
  std::optional<at::Tensor> transpose = std::nullopt;
  if (transpose_valid) {
    transpose = tensor.attr("_transpose").cast<std::optional<at::Tensor>>();
  }

  auto ret = TensorWrapper(quantization_params->get_scaling_mode());
  ret.set_rowwise_data(data.data_ptr(), dtype, shape);
  if (transpose_valid && transpose != std::nullopt) {
    const auto& transpose_shape = getShape(getTensorShape(*transpose));
    ret.set_columnwise_data(transpose->data_ptr(), dtype, transpose_shape);
  }

  const auto scale_inv_dtype = GetTransformerEngineDType(scale_inv.scalar_type());
  const auto scale_inv_shape = getShape(getTensorShape(scale_inv));
  ret.set_rowwise_scale_inv(scale_inv_dptr,
                            scale_inv_dtype,
                            scale_inv_shape);
  ret.set_columnwise_scale_inv(scale_inv_dptr,
                               scale_inv_dtype,
                               scale_inv_shape);
  quantization_params->set_quantization_params(&ret);
  return ret;
}

std::unique_ptr<QuantizationParams> CreateFloat8Params(const py::handle params) {
  auto ret = std::make_unique<Float8Params>();

  const at::Tensor &scale = params.attr("scale").cast<at::Tensor>();
  const at::Tensor &amax = params.attr("amax").cast<at::Tensor>();
  const DType type = params.attr("dtype").cast<DType>();

  ret->amax = amax;
  ret->scale = scale;
  ret->dtype = type;

  return ret;
}

}  // namespace transformer_engine::pytorch::detail
