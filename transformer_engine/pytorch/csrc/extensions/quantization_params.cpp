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

std::pair<TensorWrapper, py::object> NoneQuantizationParams::create_tensor(
    const std::vector<size_t>& shape,
    DType dtype) const {
  at::TensorOptions opts;
  opts = opts.dtype(GetATenDType(dtype)).device(torch::kCUDA);
  std::vector<int64_t> torch_shape;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
  }
  at::Tensor ret = at::empty(torch_shape, opts);
  TensorWrapper tensor;
  tensor.set_rowwise_data(ret.data_ptr(),
                          dtype,
                          shape);
  return {std::move(tensor), py::cast(ret)};
}

void Float8Params::set_quantization_params(TensorWrapper* tensor) const {
  tensor->set_scale(scale.data_ptr(),
                    GetTransformerEngineDType(scale.scalar_type()),
                    getTensorShape(scale));
  tensor->set_amax(amax.data_ptr(),
                   GetTransformerEngineDType(amax.scalar_type()),
                   getTensorShape(amax));
  auto rowwise_data = tensor->get_rowwise_data();
  rowwise_data.dtype = static_cast<NVTEDType>(dtype);

  auto columnwise_data = tensor->get_columnwise_data();
  columnwise_data.dtype = static_cast<NVTEDType>(dtype);

  tensor->set_rowwise_data(rowwise_data.data_ptr,
                           static_cast<DType>(rowwise_data.dtype),
                           rowwise_data.shape);
  tensor->set_columnwise_data(columnwise_data.data_ptr,
                              static_cast<DType>(columnwise_data.dtype),
                              columnwise_data.shape);
}

std::pair<TensorWrapper, py::object> Float8Params::create_tensor(const std::vector<size_t>& shape,
                                                                 DType dtype) const {
  using namespace pybind11::literals;
  std::vector<int64_t> torch_shape;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
  }
  at::TensorOptions opts;
  opts = opts.dtype(torch::kUInt8).device(torch::kCUDA);
  at::Tensor data = at::empty(torch_shape, opts);
  opts = opts.dtype(torch::kFloat32);
  at::Tensor scale_inv = at::empty({1}, opts);
  py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorPythonClass));
  auto ret = Float8TensorClass("data"_a=data,
                               "fp8_scale_inv"_a=scale_inv,
                               "fp8_dtype"_a=this->dtype,
                               "dtype"_a=dtype);
  TensorWrapper tensor(this->get_scaling_mode());
  tensor.set_rowwise_data(data.data_ptr(),
                          this->dtype,
                          shape);
  tensor.set_rowwise_scale_inv(scale_inv.data_ptr(),
                               DType::kFloat32,
                               std::vector<size_t>{1});
  this->set_quantization_params(&tensor);
  return {std::move(tensor), std::move(ret)};
}

void MXFP8Params::set_quantization_params(TensorWrapper* tensor) const {
  tensor->set_scale(scale.data_ptr(),
                    GetTransformerEngineDType(scale.scalar_type()),
                    getTensorShape(scale));
  tensor->set_amax(amax.data_ptr(),
                   GetTransformerEngineDType(amax.scalar_type()),
                   getTensorShape(amax));
  auto rowwise_data = tensor->get_rowwise_data();
  rowwise_data.dtype = static_cast<NVTEDType>(dtype);

  auto columnwise_data = tensor->get_columnwise_data();
  columnwise_data.dtype = static_cast<NVTEDType>(dtype);

  tensor->set_rowwise_data(rowwise_data.data_ptr,
                           static_cast<DType>(rowwise_data.dtype),
                           rowwise_data.shape);
  tensor->set_columnwise_data(columnwise_data.data_ptr,
                              static_cast<DType>(columnwise_data.dtype),
                              columnwise_data.shape);
}

std::pair<TensorWrapper, py::object> MXFP8Params::create_tensor(const std::vector<size_t>& shape,
                                                                DType dtype) const {
  using namespace pybind11::literals;
  std::vector<int64_t> torch_shape;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
  }
  at::TensorOptions opts;
  opts = opts.dtype(torch::kUInt8).device(torch::kCUDA);
  at::Tensor data = at::empty(torch_shape, opts);
  opts = opts.dtype(torch::kFloat32);
  at::Tensor scale_inv = at::empty({1}, opts);
  py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorPythonClass));
  auto ret = Float8TensorClass("data"_a=data,
                               "fp8_scale_inv"_a=scale_inv,
                               "fp8_dtype"_a=this->dtype,
                               "dtype"_a=dtype);
  TensorWrapper tensor(this->get_scaling_mode());
  tensor.set_rowwise_data(data.data_ptr(),
                          this->dtype,
                          shape);
  tensor.set_rowwise_scale_inv(scale_inv.data_ptr(),
                               DType::kFloat32,
                               std::vector<size_t>{1});
  this->set_quantization_params(&tensor);
  return {std::move(tensor), std::move(ret)};
}

}  // namespace transformer_engine::pytorch
