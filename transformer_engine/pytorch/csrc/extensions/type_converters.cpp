/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "pybind.h"

namespace transformer_engine::pytorch::detail {

TensorWrapper NVTETensorFromFloat8Tensor(py::handle tensor, QuantizationParams* quantization_params) {
  const at::Tensor &data = tensor.attr("_data").cast<at::Tensor>();
  const at::Tensor &scale_inv = tensor.attr("_scale_inv").cast<at::Tensor>();
  float *scale_inv_dptr = reinterpret_cast<float*>(scale_inv.data_ptr());
  const DType dtype = tensor.attr("_fp8_dtype").cast<DType>();

  const auto& shape = getTensorShape(data);

  bool transpose_valid = !tensor.attr("_transpose_invalid").cast<bool>();
  std::optional<at::Tensor> transpose = std::nullopt;
  if (transpose_valid) {
    transpose = tensor.attr("_transpose").cast<std::optional<at::Tensor>>();
  }

  auto ret = TensorWrapper();
  ret.set_rowwise_data(data.data_ptr(), dtype, shape);
  if (transpose_valid && transpose != std::nullopt) {
    const auto& transpose_shape = getTensorShape(*transpose);
    ret.set_columnwise_data(transpose->data_ptr(), dtype, transpose_shape);
  }

  const auto scale_inv_dtype = GetTransformerEngineDType(scale_inv.scalar_type());
  const auto scale_inv_shape = getTensorShape(scale_inv);
  ret.set_rowwise_scale_inv(scale_inv_dptr,
                            scale_inv_dtype,
                            scale_inv_shape);
  ret.set_columnwise_scale_inv(scale_inv_dptr,
                               scale_inv_dtype,
                               scale_inv_shape);
  quantization_params->set_quantization_params(&ret);
  return ret;
}

TensorWrapper NVTETensorFromMXFP8Tensor(py::handle tensor, QuantizationParams* quantization_params) {
  const DType dtype = tensor.attr("_fp8_dtype").cast<DType>();
  auto ret = TensorWrapper();
  auto [rowwise_usage, columnwise_usage] = quantization_params->get_usage();

  if (rowwise_usage) {
    const at::Tensor &data_rowwise = tensor.attr("_data_rowwise").cast<at::Tensor>();
    const at::Tensor &scale_inv_rowwise = tensor.attr("_scale_inv_rowwise").cast<at::Tensor>();
    float *scale_inv_rowwise_dptr = reinterpret_cast<float*>(scale_inv_rowwise.data_ptr());
    const auto& shape = getTensorShape(data_rowwise);
    ret.set_rowwise_data(data_rowwise.data_ptr(), dtype, shape);

    const auto scale_inv_rowwise_dtype = GetTransformerEngineDType(scale_inv_rowwise.scalar_type());
    const auto scale_inv_rowwise_shape = getTensorShape(scale_inv_rowwise);
    ret.set_rowwise_scale_inv(scale_inv_rowwise_dptr,
                              scale_inv_rowwise_dtype,
                              scale_inv_rowwise_shape);
  }

  if (columnwise_usage) {
    const at::Tensor &data_colwise = tensor.attr("_data_colwise").cast<at::Tensor>();
    const at::Tensor &scale_inv_colwise = tensor.attr("_scale_inv_colwise").cast<at::Tensor>();
    float *scale_inv_colwise_dptr = reinterpret_cast<float*>(scale_inv_colwise.data_ptr());
    const auto& shape = getTensorShape(data_colwise);
    ret.set_columnwise_data(data_colwise.data_ptr(), dtype, shape);

    const auto scale_inv_colwise_dtype = GetTransformerEngineDType(scale_inv_colwise.scalar_type());
    const auto scale_inv_colwise_shape = getTensorShape(scale_inv_colwise);
    ret.set_columnwise_scale_inv(scale_inv_colwise_dptr,
                                 scale_inv_colwise_dtype,
                                 scale_inv_colwise_shape);
  }

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

std::unique_ptr<QuantizationParams> CreateMXFP8Params(const py::handle params) {
  auto ret = std::make_unique<MXFP8Params>();

  const DType type = params.attr("dtype").cast<DType>();
  const bool rowwise_usage = params.attr("rowwise_usage").cast<bool>();
  const bool columnwise_usage = params.attr("columnwise_usage").cast<bool>();

  ret->dtype = type;
  ret->rowwise_usage = rowwise_usage;
  ret->columnwise_usage = columnwise_usage;

  return ret;
}

}  // namespace transformer_engine::pytorch::detail
