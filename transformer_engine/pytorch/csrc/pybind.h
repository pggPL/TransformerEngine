/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "transformer_engine/transformer_engine.h"
#include "common.h"

namespace transformer_engine::pytorch {

extern PyTypeObject *Float8TensorPythonClass;
extern PyTypeObject *Float8TensorBasePythonClass;
extern PyTypeObject *Float8QParamsClass;

void init_extension();

namespace detail {

inline bool IsFloat8QParams(PyObject *obj) {
  return Py_TYPE(obj) == Float8QParamsClass;
}

inline bool IsFloat8Tensor(PyObject *obj) {
  return Py_TYPE(obj) == Float8TensorPythonClass ||
         Py_TYPE(obj) == Float8TensorBasePythonClass;
}

TensorWrapper NVTETensorFromFloat8Tensor(py::handle tensor, QuantizationParams* quantization_params);

std::unique_ptr<QuantizationParams> CreateFloat8Params(const py::handle params);

inline bool IsFloatingPointType(at::ScalarType type) {
  return type == at::kFloat ||
         type == at::kHalf  ||
         type == at::kBFloat16;
}

constexpr std::array custom_types_converters = {
  std::make_tuple(IsFloat8Tensor,
                  IsFloat8QParams,
                  NVTETensorFromFloat8Tensor,
                  CreateFloat8Params)
};

}  // namespace detail

}  // namespace transformer_engine::pytorch

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_
