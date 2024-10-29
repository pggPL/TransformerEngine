/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "torch/torch.h"
#include <pybind.h>
#include "pybind.h"

namespace transformer_engine::pytorch {

py::handle NoneQuantizationParams::create_tensor(const std::vector<size_t>& shape,
                                                 DType dtype) const {
    at::TensorOptions opts;
    opts = opts.dtype(GetATenDType(dtype)).device(c10::kCUDA);
    at::Tensor ret = at::empty(shape, opts);
    return py::cast(ret);
}

void Float8Params::set_quantization_params(TensorWrapper* tensor) const {
  tensor->set_scale(scale.data_ptr(),
                    GetTransformerEngineDType(scale.scalar_type()),
                    getShape(getTensorShape(scale)));
  tensor->set_amax(amax.data_ptr(),
                   GetTransformerEngineDType(amax.scalar_type()),
                   getShape(getTensorShape(amax)));
  auto rowwise_data = tensor->get_rowwise_data();
  rowwise_data.dtype = dtype;

  auto columnwise_data = tensor->get_columnwise_data();
  columnwise_data.dtype = dtype;

  tensor->set_rowwise_data(rowwise_data.data_ptr,
                           rowwise_data.dtype,
                           rowwise_data.shape);
  tensor->set_columnwise_data(columnwise_data.data_ptr,
                              columnwise_data.dtype,
                              columnwise_data.shape);

  return *tensor;
}

py::handle Float8Params::create_tensor(const std::vector<size_t>& shape,
                                       DType dtype) const {
  using namespace pybind11::literals;
  std::vector<int64_t> torch_shape;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
  }
  at::TensorOptions opts;
  opts = opts.dtype(torch::kUInt8).device(c10::kCUDA);
  at::Tensor data = at::empty(torch_shape, opts);
  opts = opts.dtype(torch::kFloat32);
  at::Tensor scale_inv = at::empty({1}, opts);
  auto ret = Float8TensorClass("data"_a=data,
                               "fp8_scale_inv"_a=scale_inv,
                               "fp8_dtype"_a=this->dtype,
                               "dtype"_a=dtype);
  return ret;
}

}  // namespace transformer_engine::pytorch
