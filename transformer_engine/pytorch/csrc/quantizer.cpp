/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime_api.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/python/interop.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "common.h"
#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "pybind.h"

namespace nb = nanobind;
namespace ts = torch::stable;

namespace transformer_engine::pytorch {

namespace {

using torch::headeronly::ScalarType;

constexpr ScalarType kU8 = ScalarType::Byte;
constexpr ScalarType kF32 = ScalarType::Float;
constexpr ScalarType kI64 = ScalarType::Long;

// ---------------------------------------------------------------------------
// Python <-> stable Tensor helpers
// ---------------------------------------------------------------------------

/*! @brief Wrap a stable Tensor as a new Python tensor object (steals new ref). */
nb::object tensor_to_py(const ts::Tensor& t) {
  return nb::steal<nb::object>(nb::handle(static_cast<PyObject*>(ts::to_pyobject(t))));
}

/*! @brief Read a Python tensor (nb handle) into a stable Tensor. */
ts::Tensor tensor_from_py(nb::handle h) { return ts::from_pyobject(h.ptr()); }

nb::object maybe_tensor_to_py(const std::optional<ts::Tensor>& tensor) {
  return tensor ? tensor_to_py(*tensor) : nb::object(nb::none());
}

/*! @brief Read a Python ``DType`` (IntEnum) attribute as a C++ ``DType``.
 *
 * The Python-side dtype is an ``IntEnum`` so it converts to ``int`` natively;
 * this avoids needing a nanobind type-caster for ``transformer_engine::DType``.
 */
DType read_dtype(nb::handle h) { return static_cast<DType>(nb::cast<int>(h)); }

/*! @brief Convert a TE ``DType`` to a Python ``torch.dtype`` object. */
nb::object dtype_to_py(DType dtype) {
  // TODO(stable-abi): needs ScalarType -> Python torch.dtype conversion
  //                   (pybind11 had a registered caster for at::ScalarType).
  return nb::steal<nb::object>(
      nb::handle(static_cast<PyObject*>(ts::scalartype_to_pyobject(GetATenDType(dtype)))));
}

// ---------------------------------------------------------------------------
// Device / allocation helpers (concentrate the TensorOptions -> stable gap)
// ---------------------------------------------------------------------------

ts::Device cuda_device() {
  // TODO(stable-abi): needs current CUDA device as a torch::stable::Device
  return ts::Device(torch::headeronly::DeviceType::CUDA, ts::accelerator::getCurrentDeviceIndex());
}

ts::Device device_of(const ts::Tensor& t) {
  // TODO(stable-abi): needs torch::stable::Tensor -> torch::stable::Device
  return ts::Device(torch::headeronly::DeviceType::CUDA, t.get_device());
}

nb::object device_to_py(const ts::Device& device) {
  // TODO(stable-abi): needs torch::stable::Device -> Python torch.device conversion
  return nb::steal<nb::object>(
      nb::handle(static_cast<PyObject*>(ts::device_to_pyobject(device))));
}

ts::Tensor stable_empty(const std::vector<int64_t>& shape, ScalarType dtype,
                        const ts::Device& device, bool pin_memory = false) {
  // TODO(stable-abi): needs torch::stable::empty(IntArrayRef, ScalarType, Device, bool pin_memory)
  return ts::empty(shape, dtype, device, pin_memory);
}

ts::Tensor stable_empty_cuda(const std::vector<int64_t>& shape, ScalarType dtype) {
  return stable_empty(shape, dtype, cuda_device());
}

cudaStream_t current_cuda_stream() {
  // TODO(stable-abi): needs cudaStream_t from torch::stable::accelerator stream
  return static_cast<cudaStream_t>(ts::accelerator::getCurrentStream().stream());
}

/*! @brief Resolve an optional device to a concrete CUDA device
 *
 * If no device is provided, uses the current CUDA device.
 */
ts::Device resolve_device(std::optional<ts::Device> device,
                          const std::optional<ts::Tensor>& data = std::nullopt) {
  if (device.has_value() && data.has_value()) {
    // Ensure that they are the same
    const auto provided_device = *device;
    const auto data_device = device_of(*data);
    // TODO(stable-abi): needs torch::stable::Device operator==
    NVTE_CHECK(provided_device == data_device,
               "Provided device and the device of the provided data tensor are not the same.");
    return provided_device;
  }
  if (device.has_value()) {
    return *device;
  }
  if (data.has_value()) {
    return device_of(*data);
  }
  return cuda_device();
}

/*! @brief Transposed tensor shape
 *
 * The tensor is interpreted as a 2D matrix by flattening all but the
 * last dimension, and then transposed.
 */
template <typename T = size_t, typename S = T>
std::vector<T> make_transpose_shape(const std::vector<S>& shape) {
  std::vector<T> ret;
  if (shape.size() > 0) {
    ret.push_back(shape.back());
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      ret.push_back(shape[i]);
    }
  }
  return ret;
}

/*! @brief Calculate stride from shape for contiguous tensors */
template <typename T>
std::vector<T> stride_from_shape(const std::vector<T>& shape) {
  std::vector<T> stride;
  if (shape.empty()) {
    return stride;
  }
  std::vector<T> rstride;
  rstride.reserve(shape.size());
  rstride.push_back(static_cast<T>(1));
  for (size_t i = shape.size(); i > 1; --i) {
    rstride.push_back(rstride.back() * shape[i - 1]);
  }
  stride.assign(rstride.rbegin(), rstride.rend());
  return stride;
}

/*! @brief Convert shape for FP4 data by dividing the last dimension by 2 */
template <typename T = size_t>
std::vector<T> convert_shape_for_fp4(const std::vector<T>& shape) {
  std::vector<T> ret;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    ret.push_back(shape[i]);
  }
  ret.push_back(shape.back() / 2);
  return ret;
}

/*! @brief Validate an optional 1D int64 CUDA grouped-tensor metadata tensor
 *         (first_dims / last_dims / tensor_offsets) of a given expected length. */
void check_grouped_metadata_tensor(const ts::Tensor& metadata_tensor, const char* metadata_name,
                                   const size_t expected_len) {
  // TODO(stable-abi): needs torch::stable::Tensor::is_cuda() / is_contiguous()
  NVTE_CHECK(metadata_tensor.is_cuda(), metadata_name, " must be on CUDA.");
  NVTE_CHECK(metadata_tensor.scalar_type() == ScalarType::Long, metadata_name,
             " must have dtype int64.");
  NVTE_CHECK(metadata_tensor.is_contiguous(), metadata_name, " must be contiguous.");
  NVTE_CHECK(static_cast<size_t>(metadata_tensor.numel()) == expected_len, metadata_name,
             " must have length ", expected_len, ".");
}

std::optional<ts::Tensor> build_grouped_tensor_offsets(const size_t num_tensors,
                                                       const std::optional<ts::Tensor>& first_dims,
                                                       const std::optional<ts::Tensor>& last_dims,
                                                       const size_t logical_first_dim,
                                                       const size_t logical_last_dim) {
  if (!first_dims.has_value() && !last_dims.has_value()) {
    return std::nullopt;
  }

  // Validate dims before the splits-to-offsets kernel reads them.
  if (first_dims.has_value()) {
    check_grouped_metadata_tensor(*first_dims, "first_dims", num_tensors);
  }
  if (last_dims.has_value()) {
    check_grouped_metadata_tensor(*last_dims, "last_dims", num_tensors);
  }

  const ts::Device offsets_device = device_of(first_dims.has_value() ? *first_dims : *last_dims);
  auto tensor_offsets =
      stable_empty({static_cast<int64_t>(num_tensors) + 1}, kI64, offsets_device);
  if (first_dims.has_value() && last_dims.has_value()) {
    auto first_dims_nvte = makeTransformerEngineTensor(*first_dims);
    auto last_dims_nvte = makeTransformerEngineTensor(*last_dims);
    auto tensor_offsets_nvte = makeTransformerEngineTensor(tensor_offsets);
    NVTE_SCOPED_GIL_RELEASE({
      nvte_splits_to_offsets_2d(first_dims_nvte.data(), last_dims_nvte.data(),
                                tensor_offsets_nvte.data(), current_cuda_stream());
    });
  } else if (first_dims.has_value()) {
    NVTE_SCOPED_GIL_RELEASE({
      nvte_splits_to_offsets(static_cast<const int64_t*>(first_dims->data_ptr()),
                             static_cast<int64_t*>(tensor_offsets.data_ptr()), num_tensors,
                             static_cast<int64_t>(logical_last_dim), current_cuda_stream());
    });
  } else {
    NVTE_SCOPED_GIL_RELEASE({
      nvte_splits_to_offsets(static_cast<const int64_t*>(last_dims->data_ptr()),
                             static_cast<int64_t*>(tensor_offsets.data_ptr()), num_tensors,
                             static_cast<int64_t>(logical_first_dim), current_cuda_stream());
    });
  }
  return tensor_offsets;
}

/*! @brief Validate grouped-tensor offset metadata and resolve the final offsets,
 *         whether they are precomputed or built from first_dims/last_dims. */
std::optional<ts::Tensor> resolve_grouped_tensor_offsets(
    const size_t num_tensors, const std::optional<ts::Tensor>& first_dims,
    const std::optional<ts::Tensor>& last_dims,
    const std::optional<ts::Tensor>& precomputed_tensor_offsets, const size_t logical_first_dim,
    const size_t logical_last_dim) {
  // Precomputed offsets take priority; otherwise build them from first_dims/last_dims
  // (which validates the dims internally).
  if (precomputed_tensor_offsets.has_value()) {
    // tensor_offsets uses a CSR-style prefix-sum layout with num_tensors+1 entries.
    check_grouped_metadata_tensor(*precomputed_tensor_offsets, "tensor_offsets", num_tensors + 1);
    return precomputed_tensor_offsets;
  }
  return build_grouped_tensor_offsets(num_tensors, first_dims, last_dims, logical_first_dim,
                                      logical_last_dim);
}

ScalarType grouped_tensor_data_dtype(const DType dtype) { return GetATenDType(dtype); }

nb::handle grouped_tensor_python_class(const bool internal) {
  PyTypeObject* cls = internal ? GroupedTensorStoragePythonClass : GroupedTensorPythonClass;
  return nb::handle(reinterpret_cast<PyObject*>(cls));
}

}  // namespace

constexpr size_t NVFP4_BLOCK_SIZE = 16;
constexpr size_t MXFP8_BLOCK_SIZE = 32;

Quantizer::Quantizer(const nb::handle& quantizer) {
  if (quantizer.is_none()) {
    this->rowwise_usage = true;
    this->columnwise_usage = true;
    this->internal = false;
    this->optimize_for_gemm = false;
  } else {
    this->rowwise_usage = nb::cast<bool>(quantizer.attr("rowwise_usage"));
    this->columnwise_usage = nb::cast<bool>(quantizer.attr("columnwise_usage"));
    this->internal = nb::cast<bool>(quantizer.attr("internal"));
    this->optimize_for_gemm = nb::cast<bool>(quantizer.attr("optimize_for_gemm"));
    this->quantizer = quantizer;
  }
}

Float8Quantizer::Float8Quantizer(const nb::handle& quantizer) : Quantizer(quantizer) {
  const ts::Tensor scale = tensor_from_py(quantizer.attr("scale"));
  const ts::Tensor amax = tensor_from_py(quantizer.attr("amax"));
  const DType type = read_dtype(quantizer.attr("dtype"));

  this->amax = amax;
  this->scale = scale;
  this->dtype = type;
}

std::pair<TensorWrapper, nb::object> NoneQuantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, std::optional<ts::Device> device_opt,
    bool pin_memory) const {
  const auto device = resolve_device(device_opt);
  const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
  return create_tensor(shape, dtype, stable_empty(shape_int64, GetATenDType(dtype), device,
                                                  pin_memory));
}

std::pair<TensorWrapper, nb::object> NoneQuantizer::create_tensor(const std::vector<size_t>& shape,
                                                                  DType dtype,
                                                                  ts::Tensor data) const {
  TensorWrapper out_cpp;
  out_cpp.set_rowwise_data(data.data_ptr(), dtype, shape);
  set_quantization_params(&out_cpp);
  return {std::move(out_cpp), tensor_to_py(data)};
}

std::pair<GroupedTensorWrapper, nb::object> NoneQuantizer::create_grouped_tensor(
    const size_t num_tensors, const std::vector<size_t>& logical_shape, const DType dtype,
    nb::object quantizer, const std::optional<ts::Tensor>& first_dims,
    const std::optional<ts::Tensor>& last_dims,
    const std::optional<ts::Tensor>& precomputed_tensor_offsets, const size_t logical_first_dim,
    const size_t logical_last_dim) const {
  const auto tensor_offsets =
      resolve_grouped_tensor_offsets(num_tensors, first_dims, last_dims, precomputed_tensor_offsets,
                                     logical_first_dim, logical_last_dim);
  const int64_t total_elements =
      static_cast<int64_t>(logical_first_dim) * static_cast<int64_t>(logical_last_dim);

  std::optional<ts::Tensor> rowwise_data;
  std::optional<ts::Tensor> columnwise_data;
  const bool with_rowwise_data = rowwise_usage;
  const bool with_columnwise_data = columnwise_usage;
  if (with_rowwise_data) {
    rowwise_data = stable_empty_cuda({total_elements}, grouped_tensor_data_dtype(dtype));
  }
  if (with_columnwise_data) {
    columnwise_data = stable_empty_cuda({total_elements}, grouped_tensor_data_dtype(dtype));
  }

  GroupedTensorWrapper out_cpp(num_tensors, logical_shape, this->get_scaling_mode());
  if (with_rowwise_data) {
    out_cpp.set_rowwise_data(rowwise_data->data_ptr(), dtype, getTensorShape(*rowwise_data));
  }
  if (with_columnwise_data) {
    out_cpp.set_columnwise_data(columnwise_data->data_ptr(), dtype,
                                getTensorShape(*columnwise_data));
  }
  if (first_dims.has_value()) {
    out_cpp.set_first_dims(first_dims->data_ptr(), DType::kInt64, getTensorShape(*first_dims));
  }
  if (last_dims.has_value()) {
    out_cpp.set_last_dims(last_dims->data_ptr(), DType::kInt64, getTensorShape(*last_dims));
  }
  if (tensor_offsets.has_value()) {
    out_cpp.set_tensor_offsets(tensor_offsets->data_ptr(), DType::kInt64,
                               getTensorShape(*tensor_offsets));
  }

  nb::handle GroupedTensorClass = grouped_tensor_python_class(this->internal);
  nb::dict kwargs;
  nb::tuple args = nb::tuple();
  const std::vector<int64_t> grouped_shape = {static_cast<int64_t>(logical_first_dim),
                                              static_cast<int64_t>(logical_last_dim)};
  const std::vector<int64_t> grouped_stride = stride_from_shape(grouped_shape);
  kwargs["shape"] = nb::cast(grouped_shape);
  kwargs["stride"] = nb::cast(grouped_stride);
  kwargs["dtype"] = dtype_to_py(dtype);
  kwargs["num_tensors"] = nb::cast(num_tensors);
  kwargs["quantizer"] = quantizer;
  kwargs["data"] = maybe_tensor_to_py(rowwise_data);
  kwargs["columnwise_data"] = maybe_tensor_to_py(columnwise_data);
  kwargs["scale_inv"] = nb::none();
  kwargs["columnwise_scale_inv"] = nb::none();
  kwargs["amax"] = nb::none();
  kwargs["columnwise_amax"] = nb::none();
  kwargs["scale"] = nb::none();
  kwargs["first_dims"] = first_dims.has_value() ? tensor_to_py(*first_dims) : nb::object(nb::none());
  kwargs["last_dims"] = last_dims.has_value() ? tensor_to_py(*last_dims) : nb::object(nb::none());
  kwargs["tensor_offsets"] =
      tensor_offsets.has_value() ? tensor_to_py(*tensor_offsets) : nb::object(nb::none());
  kwargs["with_gemm_swizzled_scales"] = nb::cast(false);
  PyObject* result = PyObject_Call(GroupedTensorClass.ptr(), args.ptr(), kwargs.ptr());
  if (result == nullptr) {
    PyErr_Print();
  }
  NVTE_CHECK(result != nullptr, "Failed to create GroupedTensor instance");
  nb::object out_py = nb::steal<nb::object>(result);

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<TensorWrapper, nb::object> NoneQuantizer::convert_and_update_tensor(
    nb::object tensor) const {
  auto tensor_pyt = tensor_from_py(tensor);
  TensorWrapper out_cpp;
  out_cpp.set_rowwise_data(tensor_pyt.data_ptr(),
                           GetTransformerEngineDType(tensor_pyt.scalar_type()),
                           getTensorShape(tensor_pyt));
  set_quantization_params(&out_cpp);
  return {std::move(out_cpp), std::move(tensor)};
}

void NoneQuantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                             const std::optional<TensorWrapper>& noop_flag) {
  NVTE_ERROR("NoneQuantizer does not support quantization");
}

void Float8Quantizer::set_quantization_params(TensorWrapper* tensor) const {
  tensor->set_scale(scale.data_ptr(), GetTransformerEngineDType(scale.scalar_type()),
                    getTensorShape(scale));
  tensor->set_amax(amax.data_ptr(), GetTransformerEngineDType(amax.scalar_type()),
                   getTensorShape(amax));
}

std::pair<TensorWrapper, nb::object> Float8Quantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, std::optional<ts::Device> device_opt,
    bool pin_memory) const {
  const auto device = resolve_device(device_opt);
  ts::Tensor scale_inv = stable_empty(std::vector<int64_t>{1}, kF32, device, pin_memory);
  return create_tensor(shape, dtype, std::nullopt, std::nullopt, std::move(scale_inv), device,
                       pin_memory);
}

std::pair<TensorWrapper, nb::object> Float8Quantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, std::optional<ts::Tensor> data,
    std::optional<ts::Tensor> transpose, std::optional<ts::Tensor> scale_inv,
    std::optional<ts::Device> device_opt, bool pin_memory) const {
  const auto device = resolve_device(device_opt, data);
  int is_non_tn_fp8_gemm_supported = nvte_is_non_tn_fp8_gemm_supported();
  // Initialize data tensor
  const bool with_data = rowwise_usage || is_non_tn_fp8_gemm_supported;
  if (with_data && !data) {
    const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    data = stable_empty(shape_int64, kU8, device, pin_memory);
  } else if (!with_data && data) {
    data.reset();
  }
  nb::object data_py = with_data ? tensor_to_py(*data) : nb::object(nb::none());

  // Initialize transpose tensor
  const bool with_transpose = columnwise_usage && !is_non_tn_fp8_gemm_supported;
  if (with_transpose && !transpose) {
    const auto transpose_shape = make_transpose_shape<int64_t>(shape);
    transpose = stable_empty(transpose_shape, kU8, device, pin_memory);
  } else if (!with_transpose && transpose) {
    transpose.reset();
  }
  nb::object transpose_py = with_transpose ? tensor_to_py(*transpose) : nb::object(nb::none());
  // Initialize scale-inverse tensor
  if (!scale_inv) {
    // TODO(stable-abi): needs torch::stable::reciprocal(Tensor)
    scale_inv = ts::reciprocal(scale);
  }
  nb::object scale_inv_py = tensor_to_py(*scale_inv);
  // Construct Python FP8 tensor
  nb::object out_py;
  if (internal) {
    // Use direct C API call bypassing pybind11 overhead
    nb::dict kwargs;
    nb::tuple args = nb::tuple();
    kwargs["data"] = data_py;
    kwargs["fp8_scale_inv"] = scale_inv_py;
    kwargs["fp8_dtype"] = MakePythonDType(this->dtype);
    kwargs["data_transpose"] = transpose_py;
    kwargs["quantizer"] = this->quantizer;
    kwargs["fake_dtype"] = dtype_to_py(dtype);

    PyObject* result = PyObject_Call(reinterpret_cast<PyObject*>(Float8TensorStoragePythonClass),
                                     args.ptr(), kwargs.ptr());
    if (result == nullptr) {
      PyErr_Print();
    }
    NVTE_CHECK(result != nullptr, "Failed to create Float8TensorStorage instance");
    out_py = nb::steal<nb::object>(result);
  } else {
    const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    const auto stride_int64 = stride_from_shape(shape_int64);

    // Use direct C API call bypassing pybind11 overhead
    nb::dict kwargs;
    nb::tuple args = nb::tuple();
    kwargs["shape"] = nb::cast(shape_int64);
    kwargs["stride"] = nb::cast(stride_int64);
    kwargs["dtype"] = dtype_to_py(dtype);
    kwargs["data"] = data_py;
    kwargs["fp8_scale_inv"] = scale_inv_py;
    kwargs["fp8_dtype"] = MakePythonDType(this->dtype);
    kwargs["data_transpose"] = transpose_py;
    kwargs["quantizer"] = this->quantizer;
    kwargs["device"] = device_to_py(device);
    PyObject* result = PyObject_Call(reinterpret_cast<PyObject*>(Float8TensorPythonClass),
                                     args.ptr(), kwargs.ptr());
    if (result == nullptr) {
      PyErr_Print();
    }

    NVTE_CHECK(result != nullptr, "Failed to create Float8Tensor instance");
    out_py = nb::steal<nb::object>(result);
  }

  // Construct C++ FP8 tensor
  TensorWrapper out_cpp(this->get_scaling_mode());
  if (with_data) {
    out_cpp.set_rowwise_data(data->data_ptr(), this->dtype, shape);
    out_cpp.set_rowwise_scale_inv(scale_inv->data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  }
  if (with_transpose) {
    const auto transpose_shape = make_transpose_shape(shape);
    out_cpp.set_columnwise_data(transpose->data_ptr(), this->dtype, transpose_shape);
    out_cpp.set_columnwise_scale_inv(scale_inv->data_ptr(), DType::kFloat32,
                                     std::vector<size_t>{1});
  }
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<GroupedTensorWrapper, nb::object> Float8Quantizer::create_grouped_tensor(
    const size_t num_tensors, const std::vector<size_t>& logical_shape, const DType dtype,
    nb::object quantizer, const std::optional<ts::Tensor>& first_dims,
    const std::optional<ts::Tensor>& last_dims,
    const std::optional<ts::Tensor>& precomputed_tensor_offsets, const size_t logical_first_dim,
    const size_t logical_last_dim) const {
  const auto tensor_offsets =
      resolve_grouped_tensor_offsets(num_tensors, first_dims, last_dims, precomputed_tensor_offsets,
                                     logical_first_dim, logical_last_dim);
  const int64_t total_elements =
      static_cast<int64_t>(logical_first_dim) * static_cast<int64_t>(logical_last_dim);

  std::optional<ts::Tensor> rowwise_data;
  std::optional<ts::Tensor> columnwise_data;
  std::optional<ts::Tensor> rowwise_scale_inv;
  std::optional<ts::Tensor> columnwise_scale_inv;
  ts::Tensor amax = stable_empty_cuda({static_cast<int64_t>(num_tensors)}, kF32);

  if (rowwise_usage) {
    rowwise_data = stable_empty_cuda({total_elements}, kU8);
    rowwise_scale_inv = stable_empty_cuda({static_cast<int64_t>(num_tensors)}, kF32);
  }
  if (columnwise_usage) {
    columnwise_data = stable_empty_cuda({total_elements}, kU8);
    columnwise_scale_inv = stable_empty_cuda({static_cast<int64_t>(num_tensors)}, kF32);
  }

  GroupedTensorWrapper out_cpp(num_tensors, logical_shape, this->get_scaling_mode());
  if (rowwise_usage) {
    out_cpp.set_rowwise_data(rowwise_data->data_ptr(), this->dtype, getTensorShape(*rowwise_data));
    out_cpp.set_rowwise_scale_inv(rowwise_scale_inv->data_ptr(), DType::kFloat32,
                                  getTensorShape(*rowwise_scale_inv));
  }
  if (columnwise_usage) {
    out_cpp.set_columnwise_data(columnwise_data->data_ptr(), this->dtype,
                                getTensorShape(*columnwise_data));
    out_cpp.set_columnwise_scale_inv(columnwise_scale_inv->data_ptr(), DType::kFloat32,
                                     getTensorShape(*columnwise_scale_inv));
  }
  out_cpp.set_amax(amax.data_ptr(), DType::kFloat32, getTensorShape(amax));
  if (first_dims.has_value()) {
    out_cpp.set_first_dims(first_dims->data_ptr(), DType::kInt64, getTensorShape(*first_dims));
  }
  if (last_dims.has_value()) {
    out_cpp.set_last_dims(last_dims->data_ptr(), DType::kInt64, getTensorShape(*last_dims));
  }
  if (tensor_offsets.has_value()) {
    out_cpp.set_tensor_offsets(tensor_offsets->data_ptr(), DType::kInt64,
                               getTensorShape(*tensor_offsets));
  }

  nb::handle GroupedTensorClass = grouped_tensor_python_class(this->internal);
  nb::dict kwargs;
  nb::tuple args = nb::tuple();
  const std::vector<int64_t> grouped_shape = {static_cast<int64_t>(logical_first_dim),
                                              static_cast<int64_t>(logical_last_dim)};
  const std::vector<int64_t> grouped_stride = stride_from_shape(grouped_shape);
  kwargs["shape"] = nb::cast(grouped_shape);
  kwargs["stride"] = nb::cast(grouped_stride);
  kwargs["dtype"] = dtype_to_py(dtype);
  kwargs["num_tensors"] = nb::cast(num_tensors);
  kwargs["quantizer"] = quantizer;
  kwargs["data"] = maybe_tensor_to_py(rowwise_data);
  kwargs["columnwise_data"] = maybe_tensor_to_py(columnwise_data);
  kwargs["scale_inv"] = maybe_tensor_to_py(rowwise_scale_inv);
  kwargs["columnwise_scale_inv"] = maybe_tensor_to_py(columnwise_scale_inv);
  kwargs["amax"] = tensor_to_py(amax);
  kwargs["columnwise_amax"] = nb::none();
  kwargs["scale"] = nb::none();
  kwargs["first_dims"] = first_dims.has_value() ? tensor_to_py(*first_dims) : nb::object(nb::none());
  kwargs["last_dims"] = last_dims.has_value() ? tensor_to_py(*last_dims) : nb::object(nb::none());
  kwargs["tensor_offsets"] =
      tensor_offsets.has_value() ? tensor_to_py(*tensor_offsets) : nb::object(nb::none());
  kwargs["with_gemm_swizzled_scales"] = nb::cast(false);
  PyObject* result = PyObject_Call(GroupedTensorClass.ptr(), args.ptr(), kwargs.ptr());
  if (result == nullptr) {
    PyErr_Print();
  }
  NVTE_CHECK(result != nullptr, "Failed to create GroupedTensor instance");
  nb::object out_py = nb::steal<nb::object>(result);

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<TensorWrapper, nb::object> Float8Quantizer::convert_and_update_tensor(
    nb::object tensor) const {
  NVTE_CHECK(detail::IsFloat8Tensor(tensor.ptr()), "Float8Quantizer must output to Float8Tensor.");
  int is_non_tn_fp8_gemm_supported = nvte_is_non_tn_fp8_gemm_supported();
  // Expected buffers
  const bool need_data = rowwise_usage || is_non_tn_fp8_gemm_supported;
  const bool need_transpose = columnwise_usage && !is_non_tn_fp8_gemm_supported;
  NVTE_CHECK(need_data || need_transpose, "Invalid usages for Float8Quantizer.");

  // Extract buffers from Python tensor
  auto data_py = tensor.attr("_data");
  auto transpose_py = tensor.attr("_transpose");
  const bool has_data = !data_py.is_none();
  const bool has_transpose = !transpose_py.is_none();
  NVTE_CHECK(has_data || has_transpose, "Float8Tensor has no data.");
  std::optional<ts::Tensor> data_tensor, transpose_tensor;
  if (has_data) {
    data_tensor = tensor_from_py(data_py);
  }
  if (has_transpose) {
    transpose_tensor = tensor_from_py(transpose_py);
  }
  ts::Tensor scale_inv_tensor = tensor_from_py(tensor.attr("_scale_inv"));

  // Tensor dimensions
  std::vector<size_t> shape;
  if (has_transpose) {
    const auto transpose_shape = getTensorShape(*transpose_tensor);
    if (transpose_shape.size() > 0) {
      for (size_t i = 1; i < transpose_shape.size(); ++i) {
        shape.push_back(transpose_shape[i]);
      }
      shape.push_back(transpose_shape.front());
    }
    if (has_data) {
      auto expected_shape = getTensorShape(*data_tensor);
      NVTE_CHECK(shape == expected_shape, "FP8 data (shape=", expected_shape,
                 ") and transpose (shape=", transpose_shape, ") do not match");
    }
  } else {  // Already checked has_data == true
    shape = getTensorShape(*data_tensor);
  }

  // Coerce data tensor
  if (has_data && !need_data) {
    data_tensor.reset();
    tensor.attr("_data") = nb::none();
  } else if (!has_data && need_data) {
    const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    data_tensor = stable_empty_cuda(shape_int64, kU8);
    tensor.attr("_data") = tensor_to_py(*data_tensor);
  }

  // Coerce transpose tensor
  if (has_transpose && !need_transpose) {
    transpose_tensor.reset();
    tensor.attr("_transpose") = nb::none();
  } else if (!has_transpose && need_transpose) {
    const auto transpose_shape = make_transpose_shape<int64_t>(shape);
    transpose_tensor = stable_empty_cuda(transpose_shape, kU8);
    tensor.attr("_transpose") = tensor_to_py(*transpose_tensor);
  }
  tensor.attr("_transpose_invalid") = !need_transpose;

  // Coerce other attrs
  tensor.attr("_fp8_dtype") = MakePythonDType(dtype);

  // Construct C++ FP8 tensor
  TensorWrapper out_cpp;
  if (data_tensor) {
    out_cpp.set_rowwise_data(data_tensor->data_ptr(), this->dtype, shape);
    out_cpp.set_rowwise_scale_inv(scale_inv_tensor.data_ptr(), DType::kFloat32,
                                  std::vector<size_t>{1});
  }
  if (transpose_tensor) {
    const auto transpose_shape = make_transpose_shape(shape);
    out_cpp.set_columnwise_data(transpose_tensor->data_ptr(), this->dtype, transpose_shape);
    out_cpp.set_columnwise_scale_inv(scale_inv_tensor.data_ptr(), DType::kFloat32,
                                     std::vector<size_t>{1});
  }
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(tensor)};
}

void Float8Quantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                               const std::optional<TensorWrapper>& noop_flag) {
  if (input.numel() == 0) {
    return;
  }
  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_v2(input.data(), out.data(), quant_config, current_cuda_stream());
  });
}

Float8CurrentScalingQuantizer::Float8CurrentScalingQuantizer(const nb::handle& quantizer)
    : Quantizer(quantizer) {
  this->dtype = read_dtype(quantizer.attr("dtype"));

  // Get amax reduction group if needed
  const bool with_amax_reduction = nb::cast<bool>(quantizer.attr("with_amax_reduction"));
  this->with_amax_reduction = with_amax_reduction;
  if (with_amax_reduction) {
    auto group = quantizer.attr("_canonicalized_amax_reduction_group")();
    NVTE_CHECK(!group.is_none(),
               "Float8CurrentScalingQuantizer could not canonicalize amax reduction group");
    this->amax_reduction_group = torch::stable::processgroup_from_pyobject(group.ptr());
  }

  // fp8 current scaling specific quantization params
  this->force_pow_2_scales = nb::cast<bool>(quantizer.attr("force_pow_2_scales"));
  this->amax_epsilon = nb::cast<float>(quantizer.attr("amax_epsilon"));
}

void Float8CurrentScalingQuantizer::set_quantization_params(TensorWrapper* tensor) const {}

std::pair<TensorWrapper, nb::object> Float8CurrentScalingQuantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, std::optional<ts::Device> device_opt,
    bool pin_memory) const {
  const auto device = resolve_device(device_opt);

  // Initialize data tensor
  ts::Tensor data_tensor;
  int is_non_tn_fp8_gemm_supported = nvte_is_non_tn_fp8_gemm_supported();
  const bool with_data = rowwise_usage || is_non_tn_fp8_gemm_supported;
  if (with_data) {
    const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    data_tensor = stable_empty(shape_int64, kU8, device, pin_memory);
  }

  // Initialize transpose tensor
  ts::Tensor transpose_tensor;
  const bool with_transpose = columnwise_usage && !is_non_tn_fp8_gemm_supported;
  if (with_transpose) {
    const auto transpose_shape = make_transpose_shape<int64_t>(shape);
    transpose_tensor = stable_empty(transpose_shape, kU8, device, pin_memory);
  }
  // Initialize scale-inverse tensor
  ts::Tensor scale_inv_tensor;
  {
    const std::vector<int64_t> scale_inv_shape = {1};
    scale_inv_tensor = stable_empty(scale_inv_shape, kF32, device, pin_memory);
  }
  // Construct Python FP8 tensor
  nb::object out_py;
  nb::object scale_inv_py = tensor_to_py(scale_inv_tensor);
  nb::object data_py = with_data ? tensor_to_py(data_tensor) : nb::object(nb::none());
  nb::object transpose_py =
      with_transpose ? tensor_to_py(transpose_tensor) : nb::object(nb::none());
  if (internal) {
    // Use direct C API call bypassing pybind11 overhead
    nb::dict kwargs;
    kwargs["data"] = data_py;
    kwargs["fp8_scale_inv"] = scale_inv_py;
    kwargs["fp8_dtype"] = MakePythonDType(this->dtype);
    kwargs["data_transpose"] = transpose_py;
    kwargs["quantizer"] = this->quantizer;
    kwargs["fake_dtype"] = dtype_to_py(dtype);

    nb::tuple args = nb::tuple();
    PyObject* result = PyObject_Call(reinterpret_cast<PyObject*>(Float8TensorStoragePythonClass),
                                     args.ptr(), kwargs.ptr());
    if (result == nullptr) {
      PyErr_Print();
    }
    NVTE_CHECK(result != nullptr, "Failed to create Float8TensorStorage instance");
    out_py = nb::steal<nb::object>(result);
  } else {
    const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    const auto stride_int64 = stride_from_shape(shape_int64);
    // Use direct C API call bypassing pybind11 overhead
    nb::dict kwargs;
    kwargs["shape"] = nb::cast(shape_int64);
    kwargs["stride"] = nb::cast(stride_int64);
    kwargs["dtype"] = dtype_to_py(dtype);
    kwargs["data"] = data_py;
    kwargs["fp8_scale_inv"] = scale_inv_py;
    kwargs["fp8_dtype"] = MakePythonDType(this->dtype);
    kwargs["data_transpose"] = transpose_py;
    kwargs["quantizer"] = this->quantizer;
    kwargs["device"] = device_to_py(device);
    nb::tuple args = nb::tuple();
    PyObject* result = PyObject_Call(reinterpret_cast<PyObject*>(Float8TensorPythonClass),
                                     args.ptr(), kwargs.ptr());
    if (result == nullptr) {
      PyErr_Print();
    }

    NVTE_CHECK(result != nullptr, "Failed to create Float8Tensor instance");
    out_py = nb::steal<nb::object>(result);
  }

  // Construct C++ FP8 tensor
  TensorWrapper out_cpp(this->get_scaling_mode());
  if (with_data) {
    out_cpp.set_rowwise_data(data_tensor.data_ptr(), this->dtype, shape);
    out_cpp.set_rowwise_scale_inv(scale_inv_tensor.data_ptr(), DType::kFloat32,
                                  std::vector<size_t>{1});
  }
  if (with_transpose) {
    const auto transpose_shape = make_transpose_shape(shape);
    out_cpp.set_columnwise_data(transpose_tensor.data_ptr(), this->dtype, transpose_shape);
    out_cpp.set_columnwise_scale_inv(scale_inv_tensor.data_ptr(), DType::kFloat32,
                                     std::vector<size_t>{1});
  }
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<GroupedTensorWrapper, nb::object> Float8CurrentScalingQuantizer::create_grouped_tensor(
    const size_t num_tensors, const std::vector<size_t>& logical_shape, const DType dtype,
    nb::object quantizer, const std::optional<ts::Tensor>& first_dims,
    const std::optional<ts::Tensor>& last_dims,
    const std::optional<ts::Tensor>& precomputed_tensor_offsets, const size_t logical_first_dim,
    const size_t logical_last_dim) const {
  // Group Quantize is not implemented for varying both dims yet.
  NVTE_CHECK(!(first_dims.has_value() && last_dims.has_value()),
             "FP8 current-scaling grouped quantization does not support varying both "
             "first and last dimensions.");

  const auto tensor_offsets =
      resolve_grouped_tensor_offsets(num_tensors, first_dims, last_dims, precomputed_tensor_offsets,
                                     logical_first_dim, logical_last_dim);
  const int64_t total_elements =
      static_cast<int64_t>(logical_first_dim) * static_cast<int64_t>(logical_last_dim);

  std::optional<ts::Tensor> rowwise_data;
  std::optional<ts::Tensor> columnwise_data;
  std::optional<ts::Tensor> rowwise_scale_inv;
  std::optional<ts::Tensor> columnwise_scale_inv;
  ts::Tensor scale = stable_empty_cuda({static_cast<int64_t>(num_tensors)}, kF32);
  ts::Tensor amax = stable_empty_cuda({static_cast<int64_t>(num_tensors)}, kF32);
  const bool is_non_tn_fp8_gemm_supported = nvte_is_non_tn_fp8_gemm_supported();
  const bool with_rowwise_data = rowwise_usage || is_non_tn_fp8_gemm_supported;
  const bool with_columnwise_data = columnwise_usage && !is_non_tn_fp8_gemm_supported;

  // FP8 current scaling has a single per-tensor scale
  std::optional<ts::Tensor> scale_inv;
  if (with_rowwise_data || with_columnwise_data) {
    scale_inv = stable_empty_cuda({static_cast<int64_t>(num_tensors)}, kF32);
  }
  if (with_rowwise_data) {
    rowwise_data = stable_empty_cuda({total_elements}, kU8);
    rowwise_scale_inv = scale_inv;
  }
  if (with_columnwise_data) {
    columnwise_data = stable_empty_cuda({total_elements}, kU8);
    columnwise_scale_inv = scale_inv;
  }

  GroupedTensorWrapper out_cpp(num_tensors, logical_shape, this->get_scaling_mode());
  if (with_rowwise_data) {
    out_cpp.set_rowwise_data(rowwise_data->data_ptr(), this->dtype, getTensorShape(*rowwise_data));
    out_cpp.set_rowwise_scale_inv(rowwise_scale_inv->data_ptr(), DType::kFloat32,
                                  getTensorShape(*rowwise_scale_inv));
  }
  if (with_columnwise_data) {
    out_cpp.set_columnwise_data(columnwise_data->data_ptr(), this->dtype,
                                getTensorShape(*columnwise_data));
    out_cpp.set_columnwise_scale_inv(columnwise_scale_inv->data_ptr(), DType::kFloat32,
                                     getTensorShape(*columnwise_scale_inv));
  }
  out_cpp.set_scale(scale.data_ptr(), DType::kFloat32, getTensorShape(scale));
  out_cpp.set_amax(amax.data_ptr(), DType::kFloat32, getTensorShape(amax));
  if (first_dims.has_value()) {
    out_cpp.set_first_dims(first_dims->data_ptr(), DType::kInt64, getTensorShape(*first_dims));
  }
  if (last_dims.has_value()) {
    out_cpp.set_last_dims(last_dims->data_ptr(), DType::kInt64, getTensorShape(*last_dims));
  }
  if (tensor_offsets.has_value()) {
    out_cpp.set_tensor_offsets(tensor_offsets->data_ptr(), DType::kInt64,
                               getTensorShape(*tensor_offsets));
  }

  nb::handle GroupedTensorClass = grouped_tensor_python_class(this->internal);
  nb::dict kwargs;
  nb::tuple args = nb::tuple();
  const std::vector<int64_t> grouped_shape = {static_cast<int64_t>(logical_first_dim),
                                              static_cast<int64_t>(logical_last_dim)};
  const std::vector<int64_t> grouped_stride = stride_from_shape(grouped_shape);
  kwargs["shape"] = nb::cast(grouped_shape);
  kwargs["stride"] = nb::cast(grouped_stride);
  kwargs["dtype"] = dtype_to_py(dtype);
  kwargs["num_tensors"] = nb::cast(num_tensors);
  kwargs["quantizer"] = quantizer;
  kwargs["data"] = maybe_tensor_to_py(rowwise_data);
  kwargs["columnwise_data"] = maybe_tensor_to_py(columnwise_data);
  kwargs["scale_inv"] = maybe_tensor_to_py(rowwise_scale_inv);
  kwargs["columnwise_scale_inv"] = maybe_tensor_to_py(columnwise_scale_inv);
  kwargs["amax"] = tensor_to_py(amax);
  kwargs["columnwise_amax"] = nb::none();
  kwargs["scale"] = tensor_to_py(scale);
  kwargs["first_dims"] = first_dims.has_value() ? tensor_to_py(*first_dims) : nb::object(nb::none());
  kwargs["last_dims"] = last_dims.has_value() ? tensor_to_py(*last_dims) : nb::object(nb::none());
  kwargs["tensor_offsets"] =
      tensor_offsets.has_value() ? tensor_to_py(*tensor_offsets) : nb::object(nb::none());
  kwargs["offsets"] = nb::none();
  kwargs["with_gemm_swizzled_scales"] = nb::cast(false);
  PyObject* result = PyObject_Call(GroupedTensorClass.ptr(), args.ptr(), kwargs.ptr());
  if (result == nullptr) {
    PyErr_Print();
  }
  NVTE_CHECK(result != nullptr, "Failed to create GroupedTensor instance");
  nb::object out_py = nb::steal<nb::object>(result);

  return {std::move(out_cpp), std::move(out_py)};
}

std::tuple<TensorWrapper, nb::object, ts::Tensor>
Float8CurrentScalingQuantizer::create_unquantized_tensor_with_amax(const std::vector<size_t>& shape,
                                                                   DType dtype,
                                                                   std::optional<ts::Tensor> data) {
  // TODO(stable-abi): needs torch::stable::zeros(IntArrayRef, ScalarType, Device)
  ts::Tensor amax_buf = ts::zeros({1}, kF32, cuda_device());
  auto out = data.has_value() ? NoneQuantizer(nb::none()).create_tensor(shape, dtype, data.value())
                              : NoneQuantizer(nb::none()).create_tensor(shape, dtype);
  TensorWrapper out_cpp = std::move(out.first);
  nb::object out_py = std::move(out.second);
  out_cpp.set_amax(amax_buf.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  return {std::move(out_cpp), std::move(out_py), std::move(amax_buf)};
}

std::pair<TensorWrapper, nb::object> Float8CurrentScalingQuantizer::convert_and_update_tensor(
    nb::object tensor) const {
  NVTE_CHECK(detail::IsFloat8Tensor(tensor.ptr()),
             "Float8CurrentScalingQuantizer must output to Float8Tensor.");
  int is_non_tn_fp8_gemm_supported = nvte_is_non_tn_fp8_gemm_supported();
  // Expected buffers
  const bool need_data = rowwise_usage || is_non_tn_fp8_gemm_supported;
  const bool need_transpose = columnwise_usage && !is_non_tn_fp8_gemm_supported;
  NVTE_CHECK(need_data || need_transpose, "Invalid quantizer usages.");

  // Extract buffers from Python tensor
  auto data_py = tensor.attr("_data");
  auto transpose_py = tensor.attr("_transpose");
  const bool has_data = !data_py.is_none();
  const bool has_transpose = !transpose_py.is_none();
  NVTE_CHECK(has_data || has_transpose, "Tensor has no data.");
  std::optional<ts::Tensor> data_tensor, transpose_tensor;
  if (has_data) {
    data_tensor = tensor_from_py(data_py);
  }
  if (has_transpose) {
    transpose_tensor = tensor_from_py(transpose_py);
  }
  ts::Tensor scale_inv_tensor = tensor_from_py(tensor.attr("_scale_inv"));

  // Tensor dimensions
  std::vector<size_t> shape;
  if (has_transpose) {
    const auto transpose_shape = getTensorShape(*transpose_tensor);
    if (transpose_shape.size() > 0) {
      for (size_t i = 1; i < transpose_shape.size(); ++i) {
        shape.push_back(transpose_shape[i]);
      }
      shape.push_back(transpose_shape.front());
    }
    if (has_data) {
      auto expected_shape = getTensorShape(*data_tensor);
      NVTE_CHECK(shape == expected_shape, "FP8 data (shape=", expected_shape,
                 ") and transpose (shape=", transpose_shape, ") do not match");
    }
  } else {  // Already checked has_data == true
    shape = getTensorShape(*data_tensor);
  }

  // Coerce data tensor in Python tensor
  if (has_data && !need_data) {
    data_tensor.reset();
    tensor.attr("_data") = nb::none();
  } else if (!has_data && need_data) {
    const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    data_tensor = stable_empty_cuda(shape_int64, kU8);
    tensor.attr("_data") = tensor_to_py(*data_tensor);
  }

  // Coerce transpose tensor
  if (has_transpose && !need_transpose) {
    transpose_tensor.reset();
    tensor.attr("_transpose") = nb::none();
  } else if (!has_transpose && need_transpose) {
    const auto transpose_shape = make_transpose_shape<int64_t>(shape);
    transpose_tensor = stable_empty_cuda(transpose_shape, kU8);
    tensor.attr("_transpose") = tensor_to_py(*transpose_tensor);
  }
  tensor.attr("_transpose_invalid") = !need_transpose;

  // Coerce other attrs
  tensor.attr("_fp8_dtype") = MakePythonDType(dtype);

  // Construct C++ FP8 tensor
  TensorWrapper out_cpp;
  if (data_tensor) {
    out_cpp.set_rowwise_data(data_tensor->data_ptr(), this->dtype, shape);
    out_cpp.set_rowwise_scale_inv(scale_inv_tensor.data_ptr(), DType::kFloat32,
                                  std::vector<size_t>{1});
  }
  if (transpose_tensor) {
    const auto transpose_shape = make_transpose_shape(shape);
    out_cpp.set_columnwise_data(transpose_tensor->data_ptr(), this->dtype, transpose_shape);
    out_cpp.set_columnwise_scale_inv(scale_inv_tensor.data_ptr(), DType::kFloat32,
                                     std::vector<size_t>{1});
  }
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(tensor)};
}

void Float8CurrentScalingQuantizer::quantize_impl(const TensorWrapper& input, TensorWrapper& out,
                                                  const std::optional<TensorWrapper>& noop_flag,
                                                  bool compute_amax, ts::Tensor amax_buf,
                                                  ts::Tensor scale_buf) {
  out.set_amax(amax_buf.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  out.set_scale(scale_buf.data_ptr(), DType::kFloat32, std::vector<size_t>{1});

  auto stream = current_cuda_stream();

  // Nothing to be done if input is empty
  if (input.numel() == 0) {
    // Clear amax/scale pointers defensively: amax_buf/scale_buf are caller-owned
    // locals that may be released right after this call, leaving dangling raw
    // pointers in `out`.
    out.set_amax(nullptr, DType::kFloat32, out.defaultShape);
    out.set_scale(nullptr, DType::kFloat32, out.defaultShape);
    return;
  }

  // Quantization configs
  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  quant_config.set_force_pow_2_scales(force_pow_2_scales);
  quant_config.set_amax_epsilon(amax_epsilon);

  // Compute amax
  if (compute_amax) {
    NVTE_SCOPED_GIL_RELEASE(
        { nvte_compute_amax_with_config(input.data(), out.data(), quant_config, stream); });
  }

  // Perform amax reduction if needed
  if (with_amax_reduction) {
    // allreduce amax tensor
    std::vector<ts::Tensor> tensors = {amax_buf};
    NVTE_SCOPED_GIL_RELEASE(
        { amax_reduction_group->allreduce(tensors, ts::ReduceOp::MAX).wait(); });
  }

  // Compute scaling factor
  NVTE_SCOPED_GIL_RELEASE({ nvte_compute_scale_from_amax(out.data(), quant_config, stream); });

  // Cast to FP8
  out.set_amax(nullptr, DType::kFloat32, out.defaultShape);  // Avoid atomic amax updates
  NVTE_SCOPED_GIL_RELEASE({ nvte_quantize_v2(input.data(), out.data(), quant_config, stream); });

  // Clear scale pointer defensively: amax_buf/scale_buf are caller-owned locals
  // that may be released right after this call, leaving a dangling raw pointer in `out`.
  out.set_scale(nullptr, DType::kFloat32, out.defaultShape);
}

void Float8CurrentScalingQuantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                                             const std::optional<TensorWrapper>& noop_flag) {
  ts::Tensor amax_and_scale = stable_empty_cuda({2}, kF32);
  // TODO(stable-abi): needs torch::stable::select(Tensor, dim, index) (was operator[]).
  this->quantize_impl(input, out, noop_flag, true, ts::select(amax_and_scale, 0, 0),
                      ts::select(amax_and_scale, 0, 1));
}

void Float8CurrentScalingQuantizer::quantize_with_amax(
    TensorWrapper& input, TensorWrapper& out, ts::Tensor amax,
    const std::optional<TensorWrapper>& noop_flag) {
  input.set_amax(nullptr, DType::kFloat32, input.defaultShape);
  this->quantize_impl(input, out, noop_flag, false, std::move(amax), stable_empty_cuda({1}, kF32));
}

Float8BlockQuantizer::Float8BlockQuantizer(const nb::handle& quantizer) : Quantizer(quantizer) {
  this->dtype = read_dtype(quantizer.attr("dtype"));
  this->block_scaling_dim = nb::cast<int>(quantizer.attr("block_scaling_dim"));
  this->force_pow_2_scales = nb::cast<bool>(quantizer.attr("force_pow_2_scales"));
  this->amax_epsilon = nb::cast<float>(quantizer.attr("amax_epsilon"));
  NVTE_CHECK(this->block_scaling_dim == 1 || this->block_scaling_dim == 2,
             "Unsupported block scaling dim.");
}

void Float8BlockQuantizer::set_quantization_params(TensorWrapper* tensor) const {}

std::pair<TensorWrapper, nb::object> Float8BlockQuantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, std::optional<ts::Device> device_opt,
    bool pin_memory) const {
  const auto device = resolve_device(device_opt);
  std::vector<int64_t> torch_shape;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
  }

  TensorWrapper tensor(this->get_scaling_mode());
  ts::Tensor data_rowwise, data_colwise, scale_inv_rowwise, scale_inv_colwise;

  if (rowwise_usage) {
    data_rowwise = stable_empty(torch_shape, kU8, device, pin_memory);
    auto scale_shape = get_scale_shape(shape, false);
    size_t sinv0 = scale_shape[0];
    size_t sinv1 = scale_shape[1];
    scale_inv_rowwise = stable_empty({static_cast<int64_t>(sinv0), static_cast<int64_t>(sinv1)},
                                     kF32, device, pin_memory);
    tensor.set_rowwise_data(data_rowwise.data_ptr(), this->dtype, shape);
    tensor.set_rowwise_scale_inv(scale_inv_rowwise.data_ptr(), DType::kFloat32,
                                 std::vector<size_t>{sinv0, sinv1});
  }

  if (columnwise_usage) {
    std::vector<int64_t> torch_columnwise_shape;
    std::vector<size_t> columnwise_shape;
    NVTE_CHECK(torch_shape.size() == shape.size(), "Shape expected to match torch shape. Shape ",
               columnwise_shape, " torch shape: ", torch_columnwise_shape);
    if (torch_shape.size() > 0) {
      torch_columnwise_shape.reserve(torch_shape.size());
      columnwise_shape.reserve(shape.size());
      torch_columnwise_shape.push_back(torch_shape[torch_shape.size() - 1]);
      columnwise_shape.push_back(shape[shape.size() - 1]);
      for (size_t i = 0; i < torch_shape.size() - 1; ++i) {
        torch_columnwise_shape.push_back(torch_shape[i]);
        columnwise_shape.push_back(shape[i]);
      }
    }
    auto scale_shape = get_scale_shape(shape, true);
    size_t sinv0 = scale_shape[0];
    size_t sinv1 = scale_shape[1];
    data_colwise = stable_empty(torch_columnwise_shape, kU8, device, pin_memory);
    scale_inv_colwise = stable_empty({static_cast<int64_t>(sinv0), static_cast<int64_t>(sinv1)},
                                     kF32, device, pin_memory);

    tensor.set_columnwise_data(data_colwise.data_ptr(), this->dtype, columnwise_shape);
    tensor.set_columnwise_scale_inv(scale_inv_colwise.data_ptr(), DType::kFloat32,
                                    std::vector<size_t>{sinv0, sinv1});
  }
  this->set_quantization_params(&tensor);

  nb::object ret;
  if (internal) {
    // Use direct C API call bypassing pybind11 overhead
    nb::dict kwargs;
    kwargs["rowwise_data"] = tensor_to_py(data_rowwise);
    kwargs["columnwise_data"] = tensor_to_py(data_colwise);
    kwargs["rowwise_scale_inv"] = tensor_to_py(scale_inv_rowwise);
    kwargs["columnwise_scale_inv"] = tensor_to_py(scale_inv_colwise);
    kwargs["fp8_dtype"] = MakePythonDType(this->dtype);
    kwargs["quantizer"] = this->quantizer;
    kwargs["is_2D_scaled"] = nb::cast(block_scaling_dim == 2);
    kwargs["fake_dtype"] = dtype_to_py(dtype);

    nb::tuple args = nb::tuple();
    PyObject* result =
        PyObject_Call(reinterpret_cast<PyObject*>(Float8BlockwiseQTensorStoragePythonClass),
                      args.ptr(), kwargs.ptr());
    if (result == nullptr) {
      PyErr_Print();
    }

    NVTE_CHECK(result != nullptr, "Failed to create Float8BlockwiseQTensorStorage instance");
    ret = nb::steal<nb::object>(result);
  } else {
    // Use direct C API call bypassing pybind11 overhead
    nb::dict kwargs;
    const auto stride_int64 = stride_from_shape(torch_shape);
    kwargs["shape"] = nb::cast(torch_shape);
    kwargs["stride"] = nb::cast(stride_int64);
    kwargs["dtype"] = dtype_to_py(dtype);
    kwargs["rowwise_data"] = tensor_to_py(data_rowwise);
    kwargs["columnwise_data"] = tensor_to_py(data_colwise);
    kwargs["rowwise_scale_inv"] = tensor_to_py(scale_inv_rowwise);
    kwargs["columnwise_scale_inv"] = tensor_to_py(scale_inv_colwise);
    kwargs["fp8_dtype"] = MakePythonDType(this->dtype);
    kwargs["quantizer"] = this->quantizer;
    kwargs["is_2D_scaled"] = nb::cast(block_scaling_dim == 2);
    kwargs["device"] = device_to_py(device);

    nb::tuple args = nb::tuple();
    PyObject* result = PyObject_Call(reinterpret_cast<PyObject*>(Float8BlockwiseQTensorPythonClass),
                                     args.ptr(), kwargs.ptr());
    if (result == nullptr) {
      PyErr_Print();
    }
    NVTE_CHECK(result != nullptr, "Failed to create Float8BlockwiseQTensor instance");
    ret = nb::steal<nb::object>(result);
  }

  return {std::move(tensor), std::move(ret)};
}

std::pair<GroupedTensorWrapper, nb::object> Float8BlockQuantizer::create_grouped_tensor(
    const size_t num_tensors, const std::vector<size_t>& logical_shape, const DType dtype,
    nb::object quantizer, const std::optional<ts::Tensor>& first_dims,
    const std::optional<ts::Tensor>& last_dims,
    const std::optional<ts::Tensor>& precomputed_tensor_offsets, const size_t logical_first_dim,
    const size_t logical_last_dim) const {
  const auto tensor_offsets =
      resolve_grouped_tensor_offsets(num_tensors, first_dims, last_dims, precomputed_tensor_offsets,
                                     logical_first_dim, logical_last_dim);
  const int64_t total_elements =
      static_cast<int64_t>(logical_first_dim) * static_cast<int64_t>(logical_last_dim);

  std::optional<ts::Tensor> rowwise_data;
  std::optional<ts::Tensor> columnwise_data;
  std::optional<ts::Tensor> rowwise_scale_inv;
  std::optional<ts::Tensor> columnwise_scale_inv;
  const std::vector<size_t> logical_shape_vec = {logical_first_dim, logical_last_dim};

  if (rowwise_usage) {
    rowwise_data = stable_empty_cuda({total_elements}, kU8);
    const auto scale_shape = get_scale_shape(logical_shape_vec, false);
    const int64_t total_scale_elements = static_cast<int64_t>(product(scale_shape));
    rowwise_scale_inv = stable_empty_cuda({total_scale_elements}, kF32);
  }

  if (columnwise_usage) {
    columnwise_data = stable_empty_cuda({total_elements}, kU8);
    const auto scale_shape = get_scale_shape(logical_shape_vec, true);
    const int64_t total_scale_elements = static_cast<int64_t>(product(scale_shape));
    columnwise_scale_inv = stable_empty_cuda({total_scale_elements}, kF32);
  }

  GroupedTensorWrapper out_cpp(num_tensors, logical_shape, this->get_scaling_mode());
  if (rowwise_usage) {
    out_cpp.set_rowwise_data(rowwise_data->data_ptr(), this->dtype, getTensorShape(*rowwise_data));
    out_cpp.set_rowwise_scale_inv(rowwise_scale_inv->data_ptr(), DType::kFloat32,
                                  getTensorShape(*rowwise_scale_inv));
  }
  if (columnwise_usage) {
    out_cpp.set_columnwise_data(columnwise_data->data_ptr(), this->dtype,
                                getTensorShape(*columnwise_data));
    out_cpp.set_columnwise_scale_inv(columnwise_scale_inv->data_ptr(), DType::kFloat32,
                                     getTensorShape(*columnwise_scale_inv));
  }
  if (first_dims.has_value()) {
    out_cpp.set_first_dims(first_dims->data_ptr(), DType::kInt64, getTensorShape(*first_dims));
  }
  if (last_dims.has_value()) {
    out_cpp.set_last_dims(last_dims->data_ptr(), DType::kInt64, getTensorShape(*last_dims));
  }
  if (tensor_offsets.has_value()) {
    out_cpp.set_tensor_offsets(tensor_offsets->data_ptr(), DType::kInt64,
                               getTensorShape(*tensor_offsets));
  }

  nb::handle GroupedTensorClass = grouped_tensor_python_class(this->internal);
  nb::dict kwargs;
  nb::tuple args = nb::tuple();
  const std::vector<int64_t> grouped_shape = {static_cast<int64_t>(logical_first_dim),
                                              static_cast<int64_t>(logical_last_dim)};
  const std::vector<int64_t> grouped_stride = stride_from_shape(grouped_shape);
  kwargs["shape"] = nb::cast(grouped_shape);
  kwargs["stride"] = nb::cast(grouped_stride);
  kwargs["dtype"] = dtype_to_py(dtype);
  kwargs["num_tensors"] = nb::cast(num_tensors);
  kwargs["quantizer"] = quantizer;
  kwargs["data"] = maybe_tensor_to_py(rowwise_data);
  kwargs["columnwise_data"] = maybe_tensor_to_py(columnwise_data);
  kwargs["scale_inv"] = maybe_tensor_to_py(rowwise_scale_inv);
  kwargs["columnwise_scale_inv"] = maybe_tensor_to_py(columnwise_scale_inv);
  kwargs["amax"] = nb::none();
  kwargs["columnwise_amax"] = nb::none();
  kwargs["scale"] = nb::none();
  kwargs["first_dims"] = first_dims.has_value() ? tensor_to_py(*first_dims) : nb::object(nb::none());
  kwargs["last_dims"] = last_dims.has_value() ? tensor_to_py(*last_dims) : nb::object(nb::none());
  kwargs["tensor_offsets"] =
      tensor_offsets.has_value() ? tensor_to_py(*tensor_offsets) : nb::object(nb::none());
  kwargs["with_gemm_swizzled_scales"] = nb::cast(false);
  PyObject* result = PyObject_Call(GroupedTensorClass.ptr(), args.ptr(), kwargs.ptr());
  if (result == nullptr) {
    PyErr_Print();
  }
  NVTE_CHECK(result != nullptr, "Failed to create GroupedTensor instance");
  nb::object out_py = nb::steal<nb::object>(result);

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<TensorWrapper, nb::object> Float8BlockQuantizer::convert_and_update_tensor(
    nb::object tensor) const {
  const DType dtype = read_dtype(tensor.attr("_fp8_dtype"));
  bool is_2D_scaled = nb::cast<bool>(tensor.attr("_is_2D_scaled"));
  const bool with_gemm_swizzled_scales = true;

  // Extract buffers from Python tensor
  auto get_tensor = [&tensor](const char* name) -> std::optional<ts::Tensor> {
    auto attr_py = tensor.attr(name);
    if (attr_py.is_none()) {
      return std::nullopt;
    }
    return tensor_from_py(attr_py);
  };
  auto rowwise_data = get_tensor("_rowwise_data");
  auto rowwise_scale_inv = get_tensor("_rowwise_scale_inv");
  auto columnwise_data = get_tensor("_columnwise_data");
  auto columnwise_scale_inv = get_tensor("_columnwise_scale_inv");
  NVTE_CHECK(rowwise_data || columnwise_data, "FP8BlockwiseTensor has no data.");

  auto get_columnwise_shape = [&columnwise_data]() -> std::vector<size_t> {
    if (!columnwise_data) {
      return std::vector<size_t>();
    }
    std::vector<size_t> shape = getTensorShape(*columnwise_data);
    std::vector<size_t> shape_transposed(shape.size());
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
      shape_transposed[i] = shape[i + 1];
    }
    if (shape.size() > 0) {
      shape_transposed[shape.size() - 1] = shape[0];
    }
    return shape_transposed;
  };
  std::vector<size_t> shape;
  if (rowwise_data) {
    shape = getTensorShape(*rowwise_data);
    if (columnwise_data) {
      auto expected_shape = get_columnwise_shape();
      NVTE_CHECK(shape == expected_shape, "BlockwiseFP8 row-wise data (shape=", shape,
                 ") and column-wise data (shape=", expected_shape, ") do not match");
    }
  } else {
    shape = get_columnwise_shape();
  }
  std::vector<int64_t> torch_shape;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
  }

  // Coerce row-wise data
  if (rowwise_usage) {
    if (!rowwise_data) {
      rowwise_data = stable_empty_cuda(torch_shape, kU8);
      tensor.attr("_rowwise_data") = tensor_to_py(*rowwise_data);
    }
    if (!rowwise_scale_inv) {
      auto scale_shape = get_scale_shape(shape, false);
      size_t sinv0 = scale_shape[0];
      size_t sinv1 = scale_shape[1];
      rowwise_scale_inv =
          stable_empty_cuda({static_cast<int64_t>(sinv0), static_cast<int64_t>(sinv1)}, kF32);
      tensor.attr("_rowwise_scale_inv") = tensor_to_py(*rowwise_scale_inv);
    }
  } else {  // rowwise_usage == false
    if (rowwise_data) {
      rowwise_data.reset();
      tensor.attr("_rowwise_data") = nb::none();
    }
    if (rowwise_scale_inv) {
      rowwise_scale_inv.reset();
      tensor.attr("_rowwise_scale_inv") = nb::none();
    }
  }

  // Coerce column-wise data
  if (columnwise_usage) {
    std::vector<size_t> columnwise_shape;
    std::vector<int64_t> torch_columnwise_shape;
    if (torch_shape.size() > 0) {
      torch_columnwise_shape.reserve(torch_shape.size());
      columnwise_shape.reserve(shape.size());
      torch_columnwise_shape.push_back(torch_shape[torch_shape.size() - 1]);
      columnwise_shape.push_back(shape[shape.size() - 1]);
      for (size_t i = 0; i < torch_shape.size() - 1; ++i) {
        torch_columnwise_shape.push_back(torch_shape[i]);
        columnwise_shape.push_back(shape[i]);
      }
    }
    if (!columnwise_data) {
      columnwise_data = stable_empty_cuda(torch_columnwise_shape, kU8);
      tensor.attr("_columnwise_data") = tensor_to_py(*columnwise_data);
    }
    if (!columnwise_scale_inv) {
      auto scale_shape = get_scale_shape(shape, true);
      size_t sinv0 = scale_shape[0];
      size_t sinv1 = scale_shape[1];
      columnwise_scale_inv =
          stable_empty_cuda({static_cast<int64_t>(sinv0), static_cast<int64_t>(sinv1)}, kF32);
      tensor.attr("_columnwise_scale_inv") = tensor_to_py(*columnwise_scale_inv);
    }
  } else {  // columnwise_usage == false
    if (columnwise_data) {
      columnwise_data.reset();
      tensor.attr("_columnwise_data") = nb::none();
    }
    if (columnwise_scale_inv) {
      columnwise_scale_inv.reset();
      tensor.attr("_columnwise_scale_inv") = nb::none();
    }
  }

  auto ret = TensorWrapper(is_2D_scaled ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D);

  if (rowwise_usage) {
    const ts::Tensor data_rowwise = tensor_from_py(tensor.attr("_rowwise_data"));
    const ts::Tensor scale_inv_rowwise = tensor_from_py(tensor.attr("_rowwise_scale_inv"));
    void* scale_inv_rowwise_dptr = scale_inv_rowwise.data_ptr();
    const auto& rowwise_shape = getTensorShape(data_rowwise);
    ret.set_rowwise_data(data_rowwise.data_ptr(), dtype, rowwise_shape);
    const auto scale_inv_rowwise_shape = getTensorShape(scale_inv_rowwise);
    ret.set_rowwise_scale_inv(scale_inv_rowwise_dptr, DType::kFloat32, scale_inv_rowwise_shape);
  }
  if (columnwise_usage) {
    const ts::Tensor data_colwise = tensor_from_py(tensor.attr("_columnwise_data"));
    const ts::Tensor scale_inv_colwise = tensor_from_py(tensor.attr("_columnwise_scale_inv"));
    void* scale_inv_colwise_dptr = scale_inv_colwise.data_ptr();
    const auto& shape = getTensorShape(data_colwise);
    ret.set_columnwise_data(data_colwise.data_ptr(), dtype, shape);
    const auto scale_inv_colwise_shape = getTensorShape(scale_inv_colwise);
    ret.set_columnwise_scale_inv(scale_inv_colwise_dptr, DType::kFloat32, scale_inv_colwise_shape);
  }
  ret.set_with_gemm_swizzled_scales(with_gemm_swizzled_scales);
  set_quantization_params(&ret);
  return {std::move(ret), std::move(tensor)};
}

void Float8BlockQuantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                                    const std::optional<TensorWrapper>& noop_flag) {
  if (input.numel() == 0) {
    return;
  }
  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  quant_config.set_force_pow_2_scales(force_pow_2_scales);
  quant_config.set_amax_epsilon(amax_epsilon);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_v2(input.data(), out.data(), quant_config, current_cuda_stream());
  });
}

std::vector<size_t> Float8BlockQuantizer::get_scale_shape(const std::vector<size_t>& shape,
                                                          bool columnwise) const {
  size_t numel = 1;
  for (auto s : shape) {
    numel *= s;
  }

  size_t k_dim = shape.size() == 0 ? 1u : shape.back();
  size_t m_dim = numel / k_dim;
  constexpr size_t kBlockLen = 128;

  std::vector<size_t> scale_shape;

  bool rowwise_usage = !columnwise;

  if (rowwise_usage) {
    // rowwise scaling factor shape
    size_t sinv0 = 0;
    size_t sinv1 = 0;
    if (block_scaling_dim == 2) {
      sinv0 = ceildiv(m_dim, kBlockLen);
      sinv1 = roundup(ceildiv(k_dim, kBlockLen), 4);
    } else if (block_scaling_dim == 1) {
      // default rowwise scaling factor shape already transpose the scaling factor so it's GEMM_READY
      sinv0 = ceildiv(k_dim, kBlockLen);
      sinv1 = roundup(m_dim, 4);
    } else {
      NVTE_ERROR(
          "Unsupported block_scaling_dim in create_tensor rowwise."
          "Expected 1 or 2. Got ",
          block_scaling_dim);
    }
    scale_shape = {sinv0, sinv1};
  } else {
    // columnwise scaling factor shape
    size_t sinv0 = 0;
    size_t sinv1 = 0;
    if (block_scaling_dim == 2) {
      sinv0 = ceildiv(k_dim, kBlockLen);
      sinv1 = roundup(ceildiv(m_dim, kBlockLen), 4);
    } else if (block_scaling_dim == 1) {
      sinv0 = ceildiv(m_dim, kBlockLen);
      sinv1 = roundup(k_dim, 4);
    } else {
      NVTE_ERROR(
          "Unsupported block_scaling_dim in create_tensor columnwise."
          "Expected 1 or 2. Got ",
          block_scaling_dim);
    }
    scale_shape = {sinv0, sinv1};
  }
  return scale_shape;
}

MXFP8Quantizer::MXFP8Quantizer(const nb::handle& quantizer) : Quantizer(quantizer) {
  this->dtype = read_dtype(quantizer.attr("dtype"));
}

void MXFP8Quantizer::set_quantization_params(TensorWrapper* tensor) const {}

std::pair<TensorWrapper, nb::object> MXFP8Quantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, std::optional<ts::Device> device_opt,
    bool pin_memory) const {
  const auto device = resolve_device(device_opt);

  // Scaling factor format
  const bool with_gemm_swizzled_scales = this->optimize_for_gemm;

  // Tensor dimensions
  const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
  const auto [flat_first_dim, flat_last_dim] = get_2d_dims(shape);
  NVTE_CHECK(flat_first_dim % MXFP8_BLOCK_SIZE == 0 && flat_last_dim % MXFP8_BLOCK_SIZE == 0,
             "MXFP8 requires tensor dims that are divisible by ", MXFP8_BLOCK_SIZE,
             " (got shape=", shape, ")");
  const auto rowwise_scale_inv_shape = get_scale_shape(shape, false);
  const auto columnwise_scale_inv_shape = get_scale_shape(shape, true);

  // Allocate tensors
  ts::Tensor rowwise_data_tensor, rowwise_scale_inv_tensor;
  ts::Tensor columnwise_data_tensor, columnwise_scale_inv_tensor;
  if (rowwise_usage) {
    const std::vector<int64_t> scale_inv_shape_int64(rowwise_scale_inv_shape.begin(),
                                                     rowwise_scale_inv_shape.end());
    rowwise_data_tensor = stable_empty(shape_int64, kU8, device, pin_memory);
    rowwise_scale_inv_tensor = stable_empty(scale_inv_shape_int64, kU8, device, pin_memory);
  }
  if (columnwise_usage) {
    const std::vector<int64_t> scale_inv_shape_int64(columnwise_scale_inv_shape.begin(),
                                                     columnwise_scale_inv_shape.end());
    columnwise_data_tensor = stable_empty(shape_int64, kU8, device, pin_memory);
    columnwise_scale_inv_tensor = stable_empty(scale_inv_shape_int64, kU8, device, pin_memory);
  }

  // Convert tensors to Python
  auto py_cast = [](ts::Tensor& tensor, bool need_cast) -> nb::object {
    return need_cast ? tensor_to_py(tensor) : nb::object(nb::none());
  };
  auto rowwise_data_py = py_cast(rowwise_data_tensor, rowwise_usage);
  auto rowwise_scale_inv_py = py_cast(rowwise_scale_inv_tensor, rowwise_usage);
  auto columnwise_data_py = py_cast(columnwise_data_tensor, columnwise_usage);
  auto columnwise_scale_inv_py = py_cast(columnwise_scale_inv_tensor, columnwise_usage);

  // Construct Python MXFP8 tensor
  nb::object out_py;
  if (internal) {
    // Use direct C API call bypassing pybind11 overhead
    nb::dict kwargs;
    nb::tuple args = nb::tuple();
    kwargs["rowwise_data"] = rowwise_data_py;
    kwargs["columnwise_data"] = columnwise_data_py;
    kwargs["rowwise_scale_inv"] = rowwise_scale_inv_py;
    kwargs["columnwise_scale_inv"] = columnwise_scale_inv_py;
    kwargs["fp8_dtype"] = MakePythonDType(this->dtype);
    kwargs["quantizer"] = this->quantizer;
    kwargs["with_gemm_swizzled_scales"] = nb::cast(with_gemm_swizzled_scales);
    kwargs["fake_dtype"] = dtype_to_py(dtype);

    PyObject* result = PyObject_Call(reinterpret_cast<PyObject*>(MXFP8TensorStoragePythonClass),
                                     args.ptr(), kwargs.ptr());
    if (result == nullptr) {
      PyErr_Print();
    }

    NVTE_CHECK(result != nullptr, "Failed to create MXFP8TensorStorage instance");
    out_py = nb::steal<nb::object>(result);
  } else {
    // Use direct C API call bypassing pybind11 overhead
    nb::dict kwargs;
    const auto stride_int64 = stride_from_shape(shape_int64);
    kwargs["shape"] = nb::cast(shape_int64);
    kwargs["stride"] = nb::cast(stride_int64);
    kwargs["dtype"] = dtype_to_py(dtype);
    kwargs["rowwise_data"] = rowwise_data_py;
    kwargs["columnwise_data"] = columnwise_data_py;
    kwargs["rowwise_scale_inv"] = rowwise_scale_inv_py;
    kwargs["columnwise_scale_inv"] = columnwise_scale_inv_py;
    kwargs["fp8_dtype"] = MakePythonDType(this->dtype);
    kwargs["quantizer"] = this->quantizer;
    kwargs["with_gemm_swizzled_scales"] = nb::cast(with_gemm_swizzled_scales);
    kwargs["device"] = device_to_py(device);

    nb::tuple args = nb::tuple();
    PyObject* result = PyObject_Call(reinterpret_cast<PyObject*>(MXFP8TensorPythonClass),
                                     args.ptr(), kwargs.ptr());
    if (result == nullptr) {
      PyErr_Print();
    }

    NVTE_CHECK(result != nullptr, "Failed to create MXFP8Tensor instance");
    out_py = nb::steal<nb::object>(result);
  }

  // Construct C++ MXFP8 tensor
  TensorWrapper out_cpp(NVTE_MXFP8_1D_SCALING);
  if (rowwise_usage) {
    out_cpp.set_rowwise_data(rowwise_data_tensor.data_ptr(), this->dtype, shape);
    out_cpp.set_rowwise_scale_inv(rowwise_scale_inv_tensor.data_ptr(), DType::kFloat8E8M0,
                                  rowwise_scale_inv_shape);
  }
  if (columnwise_usage) {
    out_cpp.set_columnwise_data(columnwise_data_tensor.data_ptr(), this->dtype, shape);
    out_cpp.set_columnwise_scale_inv(columnwise_scale_inv_tensor.data_ptr(), DType::kFloat8E8M0,
                                     columnwise_scale_inv_shape);
  }
  out_cpp.set_with_gemm_swizzled_scales(with_gemm_swizzled_scales);
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<GroupedTensorWrapper, nb::object> MXFP8Quantizer::create_grouped_tensor(
    const size_t num_tensors, const std::vector<size_t>& logical_shape, const DType dtype,
    nb::object quantizer, const std::optional<ts::Tensor>& first_dims,
    const std::optional<ts::Tensor>& last_dims,
    const std::optional<ts::Tensor>& precomputed_tensor_offsets, const size_t logical_first_dim,
    const size_t logical_last_dim) const {
  const auto tensor_offsets =
      resolve_grouped_tensor_offsets(num_tensors, first_dims, last_dims, precomputed_tensor_offsets,
                                     logical_first_dim, logical_last_dim);
  const int64_t total_elements =
      static_cast<int64_t>(logical_first_dim) * static_cast<int64_t>(logical_last_dim);

  std::optional<ts::Tensor> rowwise_data;
  std::optional<ts::Tensor> columnwise_data;
  std::optional<ts::Tensor> rowwise_scale_inv;
  std::optional<ts::Tensor> columnwise_scale_inv;
  const std::vector<size_t> logical_shape_vec = {logical_first_dim, logical_last_dim};

  // For VARYING_BOTH_DIMS each tensor in the group must have its first and last
  // dim be a multiple of 128 (grouped-kernel tile alignment). MXFP8 stores one
  // E8M0 scale per MXFP8_BLOCK_SIZE (32) elements, so the 128-alignment
  // guarantees the per-group scale counts tile evenly and the total scale count
  // is simply scale_inv_numel = data_numel / MXFP8_BLOCK_SIZE (i.e. / 32).
  const bool is_varying_both = first_dims.has_value() && last_dims.has_value();

  if (rowwise_usage) {
    rowwise_data = stable_empty_cuda({total_elements}, kU8);
    int64_t total_scale_elements;
    if (is_varying_both) {
      total_scale_elements = total_elements / static_cast<int64_t>(MXFP8_BLOCK_SIZE);
    } else {
      const auto scale_shape = get_scale_shape(logical_shape_vec, false);
      total_scale_elements = static_cast<int64_t>(product(scale_shape));
    }
    rowwise_scale_inv = stable_empty_cuda({total_scale_elements}, kU8);
  }

  if (columnwise_usage) {
    columnwise_data = stable_empty_cuda({total_elements}, kU8);
    int64_t total_scale_elements;
    if (is_varying_both) {
      total_scale_elements = total_elements / static_cast<int64_t>(MXFP8_BLOCK_SIZE);
    } else {
      const auto scale_shape = get_scale_shape(logical_shape_vec, true);
      total_scale_elements = static_cast<int64_t>(product(scale_shape));
    }
    columnwise_scale_inv = stable_empty_cuda({total_scale_elements}, kU8);
  }

  GroupedTensorWrapper out_cpp(num_tensors, logical_shape, this->get_scaling_mode());
  if (rowwise_usage) {
    out_cpp.set_rowwise_data(rowwise_data->data_ptr(), this->dtype, getTensorShape(*rowwise_data));
    out_cpp.set_rowwise_scale_inv(rowwise_scale_inv->data_ptr(), DType::kFloat8E8M0,
                                  getTensorShape(*rowwise_scale_inv));
  }
  if (columnwise_usage) {
    out_cpp.set_columnwise_data(columnwise_data->data_ptr(), this->dtype,
                                getTensorShape(*columnwise_data));
    out_cpp.set_columnwise_scale_inv(columnwise_scale_inv->data_ptr(), DType::kFloat8E8M0,
                                     getTensorShape(*columnwise_scale_inv));
  }
  if (first_dims.has_value()) {
    out_cpp.set_first_dims(first_dims->data_ptr(), DType::kInt64, getTensorShape(*first_dims));
  }
  if (last_dims.has_value()) {
    out_cpp.set_last_dims(last_dims->data_ptr(), DType::kInt64, getTensorShape(*last_dims));
  }
  if (tensor_offsets.has_value()) {
    out_cpp.set_tensor_offsets(tensor_offsets->data_ptr(), DType::kInt64,
                               getTensorShape(*tensor_offsets));
  }

  out_cpp.set_with_gemm_swizzled_scales(this->optimize_for_gemm);

  nb::handle GroupedTensorClass = grouped_tensor_python_class(this->internal);
  nb::dict kwargs;
  nb::tuple args = nb::tuple();
  const std::vector<int64_t> grouped_shape = {static_cast<int64_t>(logical_first_dim),
                                              static_cast<int64_t>(logical_last_dim)};
  const std::vector<int64_t> grouped_stride = stride_from_shape(grouped_shape);
  kwargs["shape"] = nb::cast(grouped_shape);
  kwargs["stride"] = nb::cast(grouped_stride);
  kwargs["dtype"] = dtype_to_py(dtype);
  kwargs["num_tensors"] = nb::cast(num_tensors);
  kwargs["quantizer"] = quantizer;
  kwargs["data"] = maybe_tensor_to_py(rowwise_data);
  kwargs["columnwise_data"] = maybe_tensor_to_py(columnwise_data);
  kwargs["scale_inv"] = maybe_tensor_to_py(rowwise_scale_inv);
  kwargs["columnwise_scale_inv"] = maybe_tensor_to_py(columnwise_scale_inv);
  kwargs["amax"] = nb::none();
  kwargs["columnwise_amax"] = nb::none();
  kwargs["scale"] = nb::none();
  kwargs["first_dims"] = first_dims.has_value() ? tensor_to_py(*first_dims) : nb::object(nb::none());
  kwargs["last_dims"] = last_dims.has_value() ? tensor_to_py(*last_dims) : nb::object(nb::none());
  kwargs["tensor_offsets"] =
      tensor_offsets.has_value() ? tensor_to_py(*tensor_offsets) : nb::object(nb::none());
  kwargs["with_gemm_swizzled_scales"] = nb::cast(this->optimize_for_gemm);
  PyObject* result = PyObject_Call(GroupedTensorClass.ptr(), args.ptr(), kwargs.ptr());
  if (result == nullptr) {
    PyErr_Print();
  }
  NVTE_CHECK(result != nullptr, "Failed to create GroupedTensor instance");
  nb::object out_py = nb::steal<nb::object>(result);

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<TensorWrapper, nb::object> MXFP8Quantizer::convert_and_update_tensor(
    nb::object tensor) const {
  NVTE_CHECK(detail::IsMXFP8Tensor(tensor.ptr()), "MXFP8Quantizer must output to MXFP8Tensor.");

  // Scaling factor format
  const bool with_gemm_swizzled_scales = this->optimize_for_gemm;

  // Extract buffers from Python tensor
  auto get_tensor = [&tensor](const char* name) -> std::optional<ts::Tensor> {
    auto attr_py = tensor.attr(name);
    if (attr_py.is_none()) {
      return std::nullopt;
    }
    return tensor_from_py(attr_py);
  };
  auto rowwise_data = get_tensor("_rowwise_data");
  auto rowwise_scale_inv = get_tensor("_rowwise_scale_inv");
  auto columnwise_data = get_tensor("_columnwise_data");
  auto columnwise_scale_inv = get_tensor("_columnwise_scale_inv");
  NVTE_CHECK(rowwise_data || columnwise_data, "MXFP8Tensor has no data.");

  // Tensor dimensions
  std::vector<size_t> shape;
  if (columnwise_data) {
    shape = getTensorShape(*columnwise_data);
    if (rowwise_data) {
      auto expected_shape = getTensorShape(*rowwise_data);
      NVTE_CHECK(shape == expected_shape, "MXFP8 row-wise data (shape=", expected_shape,
                 ") and column-wise data (shape=", shape, ") do not match");
    }
  } else {  // Already checked columnwise_data_tensor == true
    shape = getTensorShape(*rowwise_data);
  }

  // Coerce row-wise data
  if (rowwise_usage) {
    if (!rowwise_data) {
      const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
      rowwise_data = stable_empty_cuda(shape_int64, kU8);
      tensor.attr("_rowwise_data") = tensor_to_py(*rowwise_data);
    }
    if (!rowwise_scale_inv) {
      const auto scale_inv_shape = get_scale_shape(shape, false);
      const std::vector<int64_t> scale_inv_shape_int64(scale_inv_shape.begin(),
                                                       scale_inv_shape.end());
      rowwise_scale_inv = stable_empty_cuda(scale_inv_shape_int64, kU8);
      tensor.attr("_rowwise_scale_inv") = tensor_to_py(*rowwise_scale_inv);
    }
  } else {  // rowwise_usage == false
    if (rowwise_data) {
      rowwise_data.reset();
      tensor.attr("_rowwise_data") = nb::none();
    }
    if (rowwise_scale_inv) {
      rowwise_scale_inv.reset();
      tensor.attr("_rowwise_scale_inv") = nb::none();
    }
  }

  // Coerce column-wise data
  if (columnwise_usage) {
    if (!columnwise_data) {
      const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
      columnwise_data = stable_empty_cuda(shape_int64, kU8);
      tensor.attr("_columnwise_data") = tensor_to_py(*columnwise_data);
    }
    if (!columnwise_scale_inv) {
      const auto scale_inv_shape = get_scale_shape(shape, true);
      const std::vector<int64_t> scale_inv_shape_int64(scale_inv_shape.begin(),
                                                       scale_inv_shape.end());
      columnwise_scale_inv = stable_empty_cuda(scale_inv_shape_int64, kU8);
      tensor.attr("_columnwise_scale_inv") = tensor_to_py(*columnwise_scale_inv);
    }
  } else {  // columnwise_usage == false
    if (columnwise_data) {
      columnwise_data.reset();
      tensor.attr("_columnwise_data") = nb::none();
    }
    if (columnwise_scale_inv) {
      columnwise_scale_inv.reset();
      tensor.attr("_columnwise_scale_inv") = nb::none();
    }
  }

  // Coerce other attrs
  tensor.attr("_fp8_dtype") = MakePythonDType(dtype);
  tensor.attr("_with_gemm_swizzled_scales") = with_gemm_swizzled_scales;

  // Construct C++ MXFP8 tensor
  TensorWrapper out_cpp(NVTE_MXFP8_1D_SCALING);
  if (rowwise_usage) {
    out_cpp.set_rowwise_data(rowwise_data->data_ptr(), dtype, shape);
    out_cpp.set_rowwise_scale_inv(rowwise_scale_inv->data_ptr(), DType::kFloat8E8M0,
                                  getTensorShape(*rowwise_scale_inv));
  }
  if (columnwise_usage) {
    out_cpp.set_columnwise_data(columnwise_data->data_ptr(), dtype, shape);
    out_cpp.set_columnwise_scale_inv(columnwise_scale_inv->data_ptr(), DType::kFloat8E8M0,
                                     getTensorShape(*columnwise_scale_inv));
  }
  out_cpp.set_with_gemm_swizzled_scales(with_gemm_swizzled_scales);
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(tensor)};
}

void MXFP8Quantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                              const std::optional<TensorWrapper>& noop_flag) {
  if (input.numel() == 0) {
    return;
  }
  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_v2(input.data(), out.data(), quant_config, current_cuda_stream());
  });
}

std::vector<size_t> MXFP8Quantizer::get_scale_shape(const std::vector<size_t>& shape,
                                                    bool columnwise) const {
  size_t numel = 1;
  for (auto s : shape) {
    numel *= s;
  }

  auto last_dim = shape.back();

  NVTE_CHECK(last_dim % MXFP8_BLOCK_SIZE == 0 && (numel / last_dim) % MXFP8_BLOCK_SIZE == 0,
             "MXFP8 requires tensor dims that are divisible by ", MXFP8_BLOCK_SIZE,
             " (got shape=", shape, ")");

  std::vector<size_t> scale_shape;

  bool rowwise_usage = !columnwise;

  if (rowwise_usage) {
    // rowwise scaling factor shape
    size_t sinv0 = roundup(numel / last_dim, 128);
    size_t sinv1 = roundup(last_dim / MXFP8_BLOCK_SIZE, 4);
    scale_shape = {sinv0, sinv1};
  } else {
    // columnwise scaling factor shape
    size_t sinv0 = roundup(numel / (last_dim * MXFP8_BLOCK_SIZE), 4);
    size_t sinv1 = roundup(last_dim, 128);
    scale_shape = {sinv0, sinv1};
  }
  return scale_shape;
}

NVFP4Quantizer::NVFP4Quantizer(const nb::handle& quantizer) : Quantizer(quantizer) {
  this->dtype = read_dtype(quantizer.attr("dtype"));
  this->with_rht = nb::cast<bool>(quantizer.attr("with_rht"));
  this->with_post_rht_amax = nb::cast<bool>(quantizer.attr("with_post_rht_amax"));
  this->with_2d_quantization = nb::cast<bool>(quantizer.attr("with_2d_quantization"));
  this->stochastic_rounding = nb::cast<bool>(quantizer.attr("stochastic_rounding"));
  const bool nvfp4_use_4over6 = nb::cast<bool>(quantizer.attr("nvfp4_use_4over6"));
  this->nvfp4_e4m3_max = nb::cast<int>(quantizer.attr("nvfp4_e4m3_max"));
  NVTE_CHECK(this->nvfp4_e4m3_max == 448 || this->nvfp4_e4m3_max == 256,
             "Unsupported NVFP4 E4M3 max: ", this->nvfp4_e4m3_max);
  const auto nvfp4_4over6_err_mode =
      nb::cast<std::string>(quantizer.attr("nvfp4_4over6_err_mode"));
  if (!nvfp4_use_4over6) {
    this->nvfp4_4over6_mode = kNVTENVFP44Over6Disabled;
  } else if (nvfp4_4over6_err_mode == "MAE") {
    this->nvfp4_4over6_mode = kNVTENVFP44Over6MinMAE;
  } else if (nvfp4_4over6_err_mode == "MSE") {
    this->nvfp4_4over6_mode = kNVTENVFP44Over6MinMSE;
  } else {
    NVTE_ERROR("Unsupported NVFP4 4over6 error mode: ", nvfp4_4over6_err_mode);
  }
  this->row_scaled_nvfp4 = nb::cast<bool>(quantizer.attr("row_scaled_nvfp4"));

  // Get amax reduction group if needed for NVFP4 AG
  const bool with_amax_reduction = nb::cast<bool>(quantizer.attr("with_amax_reduction"));
  this->with_amax_reduction = with_amax_reduction;
  if (with_amax_reduction) {
    auto group = quantizer.attr("_canonicalized_amax_reduction_group")();
    NVTE_CHECK(!group.is_none(), "NVFP4Quantizer could not canonicalize amax reduction group");
    this->amax_reduction_group = torch::stable::processgroup_from_pyobject(group.ptr());
  }

  this->rht_matrix_random_sign_mask_t =
      nb::cast<int>(quantizer.attr("rht_matrix_random_sign_mask_t"));
  this->rht_matrix = tensor_from_py(quantizer.attr("rht_matrix"));
}

void NVFP4Quantizer::set_quantization_params(TensorWrapper* tensor) const {
  // set dtype for rowwise and columnwise data in tensor wrapper
  auto rowwise_data = tensor->get_rowwise_data();
  rowwise_data.dtype = static_cast<NVTEDType>(this->dtype);

  auto columnwise_data = tensor->get_columnwise_data();
  columnwise_data.dtype = static_cast<NVTEDType>(this->dtype);

  tensor->set_rowwise_data(rowwise_data.data_ptr, static_cast<DType>(rowwise_data.dtype),
                           rowwise_data.shape);
  tensor->set_columnwise_data(columnwise_data.data_ptr, static_cast<DType>(columnwise_data.dtype),
                              columnwise_data.shape);
}

bool NVFP4Quantizer::is_eligible_for_rht_cast_fusion(const std::vector<size_t>& shape,
                                                     bool for_grouped_kernel) {
  const auto [rows, cols] = get_2d_dims(shape);
  const size_t row_align = for_grouped_kernel ? 128 : 64;
  return rows % row_align == 0 && cols % 128 == 0 && transformer_engine::cuda::sm_arch() >= 100 &&
         transformer_engine::cuda::sm_arch() <= 110;
}

std::pair<TensorWrapper, nb::object> NVFP4Quantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, std::optional<ts::Device> device_opt,
    bool pin_memory) const {
  const auto device = resolve_device(device_opt);

  // Tensor dimensions
  const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
  const auto [flat_first_dim, flat_last_dim] = get_2d_dims(shape);

  // Swizzled SF is only valid when the RHT cast-fusion path runs;
  // other quantize paths reject it.
  const bool with_gemm_swizzled_scales = this->optimize_for_gemm && this->with_rht &&
                                         NVFP4Quantizer::is_eligible_for_rht_cast_fusion(shape);
  NVTE_CHECK(flat_first_dim % NVFP4_BLOCK_SIZE == 0, "First dim for NVFP4 must be divisible by ",
             NVFP4_BLOCK_SIZE, " (got shape=", shape, ")");
  NVTE_CHECK(flat_last_dim % NVFP4_BLOCK_SIZE == 0,
             "NVFP4 requires tensor dims that are divisible by ", NVFP4_BLOCK_SIZE,
             " (got shape=", shape, ")");
  const bool row_scaled_nvfp4 = this->row_scaled_nvfp4;
  const bool nvfp4_use_4over6 = this->nvfp4_4over6_mode != kNVTENVFP44Over6Disabled;
  const int nvfp4_e4m3_max = this->nvfp4_e4m3_max;
  if (row_scaled_nvfp4) {
    NVTE_CHECK(rowwise_usage, "Row-scaled NVFP4 quantization requires rowwise usage.");
    NVTE_CHECK(!columnwise_usage,
               "Row-scaled NVFP4 quantization does not support columnwise usage.");
  }
  const auto rowwise_scale_inv_shape = get_scale_shape(shape, false);
  const auto columnwise_scale_inv_shape = get_scale_shape(shape, true);

  // Allocate tensors
  ts::Tensor rowwise_data_tensor, rowwise_scale_inv_tensor, amax_rowwise;
  ts::Tensor columnwise_data_tensor, columnwise_scale_inv_tensor, amax_columnwise;
  if (rowwise_usage) {
    const std::vector<int64_t> scale_inv_shape_int64(rowwise_scale_inv_shape.begin(),
                                                     rowwise_scale_inv_shape.end());
    rowwise_data_tensor =
        stable_empty(convert_shape_for_fp4(shape_int64), kU8, device, pin_memory);
    rowwise_scale_inv_tensor = stable_empty(scale_inv_shape_int64, kU8, device, pin_memory);
    const int64_t amax_rows = row_scaled_nvfp4 ? static_cast<int64_t>(flat_first_dim) : 1;
    // hadamard amax kernel will zero out pointer with ZeroAmaxKernel
    // nvte_compute_amax_with_config will zero out the pointer if needed
    amax_rowwise = stable_empty({amax_rows}, kF32, device, pin_memory);
  }
  if (columnwise_usage) {
    const std::vector<int64_t> scale_inv_shape_int64(columnwise_scale_inv_shape.begin(),
                                                     columnwise_scale_inv_shape.end());
    // enforce 2D shape to avoid [S, B, H] shape and B and be 1
    // and the transposed shape is [H, S, B], so divide last dim by 2 gives zero
    std::vector<int64_t> shape_int64_2d = {static_cast<int64_t>(flat_first_dim),
                                           static_cast<int64_t>(flat_last_dim)};
    const auto transpose_shape_int64 = make_transpose_shape<int64_t>(shape_int64_2d);
    columnwise_data_tensor =
        stable_empty(convert_shape_for_fp4(transpose_shape_int64), kU8, device, pin_memory);
    columnwise_scale_inv_tensor = stable_empty(scale_inv_shape_int64, kU8, device, pin_memory);
    // hadamard amax kernel will zero out pointer with ZeroAmaxKernel
    // nvte_compute_amax_with_config will zero out the pointer if needed
    amax_columnwise = stable_empty({1}, kF32, device, pin_memory);
  }

  // Convert tensors to Python
  auto py_cast = [](ts::Tensor& tensor, bool need_cast) -> nb::object {
    return need_cast ? tensor_to_py(tensor) : nb::object(nb::none());
  };
  auto rowwise_data_py = py_cast(rowwise_data_tensor, rowwise_usage);
  auto rowwise_scale_inv_py = py_cast(rowwise_scale_inv_tensor, rowwise_usage);
  auto columnwise_data_py = py_cast(columnwise_data_tensor, columnwise_usage);
  auto columnwise_scale_inv_py = py_cast(columnwise_scale_inv_tensor, columnwise_usage);
  auto amax_rowwise_py = py_cast(amax_rowwise, rowwise_usage);
  auto amax_columnwise_py = py_cast(amax_columnwise, columnwise_usage);

  // Construct Python NVFP4 tensor
  nb::object out_py;
  if (internal) {
    // Use direct C API call bypassing pybind11 overhead
    nb::dict kwargs;
    kwargs["rowwise_data"] = rowwise_data_py;
    kwargs["columnwise_data"] = columnwise_data_py;
    kwargs["rowwise_scale_inv"] = rowwise_scale_inv_py;
    kwargs["columnwise_scale_inv"] = columnwise_scale_inv_py;
    kwargs["amax_rowwise"] = amax_rowwise_py;
    kwargs["amax_columnwise"] = amax_columnwise_py;
    kwargs["fp4_dtype"] = MakePythonDType(this->dtype);
    kwargs["quantizer"] = this->quantizer;
    kwargs["with_gemm_swizzled_scales"] = nb::cast(with_gemm_swizzled_scales);
    kwargs["row_scaled_nvfp4"] = nb::cast(row_scaled_nvfp4);
    kwargs["nvfp4_use_4over6"] = nb::cast(nvfp4_use_4over6);
    kwargs["nvfp4_e4m3_max"] = nb::cast(nvfp4_e4m3_max);
    kwargs["fake_dtype"] = dtype_to_py(dtype);

    nb::tuple args = nb::tuple();

    PyObject* result = PyObject_Call(reinterpret_cast<PyObject*>(NVFP4TensorStoragePythonClass),
                                     args.ptr(), kwargs.ptr());
    if (result == nullptr) {
      PyErr_Print();
    }

    NVTE_CHECK(result != nullptr, "Failed to create NVFP4TensorStorage instance");
    out_py = nb::steal<nb::object>(result);
  } else {
    // Use direct C API call bypassing pybind11 overhead
    nb::dict kwargs;
    const auto stride_int64 = stride_from_shape(shape_int64);
    kwargs["shape"] = nb::cast(shape_int64);
    kwargs["stride"] = nb::cast(stride_int64);
    kwargs["dtype"] = dtype_to_py(dtype);
    kwargs["rowwise_data"] = rowwise_data_py;
    kwargs["columnwise_data"] = columnwise_data_py;
    kwargs["rowwise_scale_inv"] = rowwise_scale_inv_py;
    kwargs["columnwise_scale_inv"] = columnwise_scale_inv_py;
    kwargs["amax_rowwise"] = amax_rowwise_py;
    kwargs["amax_columnwise"] = amax_columnwise_py;
    kwargs["fp4_dtype"] = MakePythonDType(this->dtype);
    kwargs["quantizer"] = this->quantizer;
    kwargs["with_gemm_swizzled_scales"] = nb::cast(with_gemm_swizzled_scales);
    kwargs["device"] = device_to_py(device);
    kwargs["row_scaled_nvfp4"] = nb::cast(row_scaled_nvfp4);
    kwargs["nvfp4_use_4over6"] = nb::cast(nvfp4_use_4over6);
    kwargs["nvfp4_e4m3_max"] = nb::cast(nvfp4_e4m3_max);
    nb::tuple args = nb::tuple();
    PyObject* result = PyObject_Call(reinterpret_cast<PyObject*>(NVFP4TensorPythonClass),
                                     args.ptr(), kwargs.ptr());
    if (result == nullptr) {
      PyErr_Print();
    }

    NVTE_CHECK(result != nullptr, "Failed to create NVFP4Tensor instance");
    out_py = nb::steal<nb::object>(result);
  }

  // Construct C++ tensor
  TensorWrapper out_cpp(NVTE_NVFP4_1D_SCALING);
  if (rowwise_usage) {
    out_cpp.set_rowwise_data(rowwise_data_tensor.data_ptr(), DType::kFloat4E2M1, shape);
    out_cpp.set_rowwise_scale_inv(rowwise_scale_inv_tensor.data_ptr(), DType::kFloat8E4M3,
                                  rowwise_scale_inv_shape);
    out_cpp.set_amax(amax_rowwise.data_ptr(), DType::kFloat32, getTensorShape(amax_rowwise));
  }
  if (columnwise_usage) {
    // enforce 2D shape to avoid [S, B, H] shape and B and be 1
    // and the transposed shape is [H, S, B], so divide last dim by 2 gives zero
    std::vector<size_t> shape_2d = {flat_first_dim, flat_last_dim};
    auto col_data_shape_fp4 = make_transpose_shape<size_t>(shape_2d);
    out_cpp.set_columnwise_data(columnwise_data_tensor.data_ptr(), DType::kFloat4E2M1,
                                col_data_shape_fp4);
    out_cpp.set_columnwise_scale_inv(columnwise_scale_inv_tensor.data_ptr(), DType::kFloat8E4M3,
                                     columnwise_scale_inv_shape);
    out_cpp.set_columnwise_amax(amax_columnwise.data_ptr(), DType::kFloat32,
                                std::vector<size_t>{1});
  }
  out_cpp.set_with_gemm_swizzled_scales(with_gemm_swizzled_scales);
  out_cpp.set_row_scaled_nvfp4(row_scaled_nvfp4);
  out_cpp.set_nvfp4_e4m3_max(nvfp4_e4m3_max);
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<GroupedTensorWrapper, nb::object> NVFP4Quantizer::create_grouped_tensor(
    const size_t num_tensors, const std::vector<size_t>& logical_shape, const DType dtype,
    nb::object quantizer, const std::optional<ts::Tensor>& first_dims,
    const std::optional<ts::Tensor>& last_dims,
    const std::optional<ts::Tensor>& precomputed_tensor_offsets, const size_t logical_first_dim,
    const size_t logical_last_dim) const {
  const auto tensor_offsets =
      resolve_grouped_tensor_offsets(num_tensors, first_dims, last_dims, precomputed_tensor_offsets,
                                     logical_first_dim, logical_last_dim);
  const int64_t total_elements =
      static_cast<int64_t>(logical_first_dim) * static_cast<int64_t>(logical_last_dim);
  NVTE_CHECK(total_elements % 2 == 0, "NVFP4 data size must be divisible by 2.");

  std::optional<ts::Tensor> rowwise_data;
  std::optional<ts::Tensor> columnwise_data;
  std::optional<ts::Tensor> rowwise_scale_inv;
  std::optional<ts::Tensor> columnwise_scale_inv;
  std::optional<ts::Tensor> rowwise_amax;
  std::optional<ts::Tensor> columnwise_amax;
  const std::vector<size_t> logical_shape_vec = {logical_first_dim, logical_last_dim};
  const bool row_scaled_nvfp4 = this->row_scaled_nvfp4;
  const bool nvfp4_use_4over6 = this->nvfp4_4over6_mode != kNVTENVFP44Over6Disabled;
  const int nvfp4_e4m3_max = this->nvfp4_e4m3_max;
  if (row_scaled_nvfp4) {
    NVTE_CHECK(rowwise_usage, "Row-scaled NVFP4 grouped quantization requires rowwise usage.");
    NVTE_CHECK(!columnwise_usage,
               "Row-scaled NVFP4 grouped quantization does not support columnwise usage.");
  }

  const int64_t total_data_elements = total_elements / 2;

  if (rowwise_usage) {
    rowwise_data = stable_empty_cuda({total_data_elements}, kU8);
    const auto scale_shape = get_scale_shape(logical_shape_vec, false);
    const int64_t total_scale_elements = static_cast<int64_t>(product(scale_shape));
    rowwise_scale_inv = stable_empty_cuda({total_scale_elements}, kU8);
    const int64_t amax_elements = row_scaled_nvfp4 ? static_cast<int64_t>(logical_first_dim)
                                                   : static_cast<int64_t>(num_tensors);
    rowwise_amax = stable_empty_cuda({amax_elements}, kF32);
  }

  if (columnwise_usage) {
    columnwise_data = stable_empty_cuda({total_data_elements}, kU8);
    const auto scale_shape = get_scale_shape(logical_shape_vec, true);
    const int64_t total_scale_elements = static_cast<int64_t>(product(scale_shape));
    columnwise_scale_inv = stable_empty_cuda({total_scale_elements}, kU8);
    columnwise_amax = stable_empty_cuda({static_cast<int64_t>(num_tensors)}, kF32);
  }

  GroupedTensorWrapper out_cpp(num_tensors, logical_shape, this->get_scaling_mode());
  if (rowwise_usage) {
    out_cpp.set_rowwise_data(rowwise_data->data_ptr(), this->dtype, getTensorShape(*rowwise_data));
    out_cpp.set_rowwise_scale_inv(rowwise_scale_inv->data_ptr(), DType::kFloat8E4M3,
                                  getTensorShape(*rowwise_scale_inv));
    out_cpp.set_amax(rowwise_amax->data_ptr(), DType::kFloat32, getTensorShape(*rowwise_amax));
  }
  if (columnwise_usage) {
    out_cpp.set_columnwise_data(columnwise_data->data_ptr(), this->dtype,
                                getTensorShape(*columnwise_data));
    out_cpp.set_columnwise_scale_inv(columnwise_scale_inv->data_ptr(), DType::kFloat8E4M3,
                                     getTensorShape(*columnwise_scale_inv));
    out_cpp.set_columnwise_amax(columnwise_amax->data_ptr(), DType::kFloat32,
                                getTensorShape(*columnwise_amax));
  }
  if (first_dims.has_value()) {
    out_cpp.set_first_dims(first_dims->data_ptr(), DType::kInt64, getTensorShape(*first_dims));
  }
  if (last_dims.has_value()) {
    out_cpp.set_last_dims(last_dims->data_ptr(), DType::kInt64, getTensorShape(*last_dims));
  }
  if (tensor_offsets.has_value()) {
    out_cpp.set_tensor_offsets(tensor_offsets->data_ptr(), DType::kInt64,
                               getTensorShape(*tensor_offsets));
  }

  out_cpp.set_with_gemm_swizzled_scales(this->optimize_for_gemm);

  nb::handle GroupedTensorClass = grouped_tensor_python_class(this->internal);
  nb::dict kwargs;
  nb::tuple args = nb::tuple();
  const std::vector<int64_t> grouped_shape = {static_cast<int64_t>(logical_first_dim),
                                              static_cast<int64_t>(logical_last_dim)};
  const std::vector<int64_t> grouped_stride = stride_from_shape(grouped_shape);
  kwargs["shape"] = nb::cast(grouped_shape);
  kwargs["stride"] = nb::cast(grouped_stride);
  kwargs["dtype"] = dtype_to_py(dtype);
  kwargs["num_tensors"] = nb::cast(num_tensors);
  kwargs["quantizer"] = quantizer;
  kwargs["data"] = maybe_tensor_to_py(rowwise_data);
  kwargs["columnwise_data"] = maybe_tensor_to_py(columnwise_data);
  kwargs["scale_inv"] = maybe_tensor_to_py(rowwise_scale_inv);
  kwargs["columnwise_scale_inv"] = maybe_tensor_to_py(columnwise_scale_inv);
  kwargs["amax"] = maybe_tensor_to_py(rowwise_amax);
  kwargs["columnwise_amax"] = maybe_tensor_to_py(columnwise_amax);
  kwargs["scale"] = nb::none();
  kwargs["first_dims"] = first_dims.has_value() ? tensor_to_py(*first_dims) : nb::object(nb::none());
  kwargs["last_dims"] = last_dims.has_value() ? tensor_to_py(*last_dims) : nb::object(nb::none());
  kwargs["tensor_offsets"] =
      tensor_offsets.has_value() ? tensor_to_py(*tensor_offsets) : nb::object(nb::none());
  kwargs["with_gemm_swizzled_scales"] = nb::cast(this->optimize_for_gemm);
  kwargs["row_scaled_nvfp4"] = nb::cast(row_scaled_nvfp4);
  kwargs["nvfp4_use_4over6"] = nb::cast(nvfp4_use_4over6);
  kwargs["nvfp4_e4m3_max"] = nb::cast(nvfp4_e4m3_max);
  PyObject* result = PyObject_Call(GroupedTensorClass.ptr(), args.ptr(), kwargs.ptr());
  if (result == nullptr) {
    PyErr_Print();
  }
  NVTE_CHECK(result != nullptr, "Failed to create GroupedTensor instance");
  nb::object out_py = nb::steal<nb::object>(result);

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<TensorWrapper, nb::object> NVFP4Quantizer::create_unquantized_tensor_with_amax(
    TensorWrapper& quantized_tensor, DType dtype) {
  // Construct tensor
  auto shape = convertShape(quantized_tensor.shape());
  auto [out_cpp, out_py] = NoneQuantizer(nb::none()).create_tensor(shape, dtype);

  // Register amax pointer from quantized tensor
  auto rowwise_amax = quantized_tensor.get_amax();
  auto columnwise_amax = quantized_tensor.get_columnwise_amax();

  void* amax_ptr = rowwise_amax.data_ptr;
  std::vector<size_t> amax_shape = convertShape(rowwise_amax.shape);
  if (amax_ptr == nullptr) {
    amax_ptr = columnwise_amax.data_ptr;
    amax_shape = convertShape(columnwise_amax.shape);
  }
  NVTE_CHECK(amax_ptr != nullptr, "Could not extract amax pointer from NVFP4 tensor.");
  out_cpp.set_amax(amax_ptr, DType::kFloat32, amax_shape);

  // Zero out amax
  const size_t amax_numel = product(amax_shape);
  NVTE_CHECK_CUDA(
      cudaMemsetAsync(amax_ptr, 0, amax_numel * sizeof(float), current_cuda_stream()));

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<TensorWrapper, nb::object> NVFP4Quantizer::convert_and_update_tensor(
    nb::object tensor) const {
  NVTE_CHECK(detail::IsNVFP4Tensor(tensor.ptr()), "NVFP4Quantizer must output to IsNVFP4Tensor.");

  // Extract buffers from Python tensor
  auto get_tensor = [&tensor](const char* name) -> std::optional<ts::Tensor> {
    auto attr_py = tensor.attr(name);
    if (attr_py.is_none()) {
      return std::nullopt;
    }
    return tensor_from_py(attr_py);
  };
  auto rowwise_data = get_tensor("_rowwise_data");
  auto rowwise_scale_inv = get_tensor("_rowwise_scale_inv");
  auto columnwise_data = get_tensor("_columnwise_data");
  auto columnwise_scale_inv = get_tensor("_columnwise_scale_inv");
  auto amax_rowwise = get_tensor("_amax_rowwise");
  auto amax_columnwise = get_tensor("_amax_columnwise");
  NVTE_CHECK(rowwise_data || columnwise_data, "NVFP4Tensor has no data.");

  // Tensor dimensions, shape means original shape
  std::vector<size_t> shape;
  if (rowwise_data) {
    shape = convert_shape_back_from_fp4(getTensorShape(*rowwise_data), false);
    if (columnwise_data) {
      auto col_shape = convert_shape_back_from_fp4(getTensorShape(*columnwise_data), true);
      NVTE_CHECK(get_2d_dims(shape) == get_2d_dims(col_shape), "NVFP4 row-wise data (shape=", shape,
                 ") and column-wise data (shape=", col_shape, ") do not match");
    }
  } else {
    shape = convert_shape_back_from_fp4(getTensorShape(*columnwise_data), true);
  }

  const auto [flat_first_dim, flat_last_dim] = get_2d_dims(shape);

  // Swizzled SF is only valid when the RHT cast-fusion path runs;
  // other quantize paths reject it.
  const bool with_gemm_swizzled_scales = this->optimize_for_gemm && this->with_rht &&
                                         NVFP4Quantizer::is_eligible_for_rht_cast_fusion(shape);

  const bool row_scaled_nvfp4 = this->row_scaled_nvfp4;
  const bool nvfp4_use_4over6 = this->nvfp4_4over6_mode != kNVTENVFP44Over6Disabled;
  const int nvfp4_e4m3_max = this->nvfp4_e4m3_max;
  if (row_scaled_nvfp4) {
    NVTE_CHECK(rowwise_usage, "Row-scaled NVFP4 quantization requires rowwise usage.");
    NVTE_CHECK(!columnwise_usage,
               "Row-scaled NVFP4 quantization does not support columnwise usage.");
  }
  tensor.attr("_row_scaled_nvfp4") = row_scaled_nvfp4;
  tensor.attr("_with_gemm_swizzled_scales") = with_gemm_swizzled_scales;
  tensor.attr("_nvfp4_use_4over6") = nb::cast(nvfp4_use_4over6);
  tensor.attr("_nvfp4_e4m3_max") = nb::cast(nvfp4_e4m3_max);

  // Coerce row-wise data
  if (rowwise_usage) {
    if (!rowwise_data) {
      const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
      rowwise_data = stable_empty_cuda(convert_shape_for_fp4(shape_int64), kU8);
      tensor.attr("_rowwise_data") = tensor_to_py(*rowwise_data);
    }
    if (!rowwise_scale_inv) {
      const auto scale_inv_shape = get_scale_shape(shape, false);
      const std::vector<int64_t> scale_inv_shape_int64(scale_inv_shape.begin(),
                                                       scale_inv_shape.end());
      rowwise_scale_inv = stable_empty_cuda(scale_inv_shape_int64, kU8);
      tensor.attr("_rowwise_scale_inv") = tensor_to_py(*rowwise_scale_inv);
    }
    const int64_t amax_rows = row_scaled_nvfp4 ? static_cast<int64_t>(flat_first_dim) : 1;
    if (!amax_rowwise || amax_rowwise->numel() != amax_rows) {
      // hadamard amax kernel will zero out pointer with ZeroAmaxKernel
      // nvte_compute_amax_with_config will zero out the pointer if needed
      amax_rowwise = stable_empty_cuda({amax_rows}, kF32);
      tensor.attr("_amax_rowwise") = tensor_to_py(*amax_rowwise);
    }
  } else {  // rowwise_usage == false
    if (rowwise_data) {
      rowwise_data.reset();
      tensor.attr("_rowwise_data") = nb::none();
    }
    if (rowwise_scale_inv) {
      rowwise_scale_inv.reset();
      tensor.attr("_rowwise_scale_inv") = nb::none();
    }
    if (amax_rowwise) {
      amax_rowwise.reset();
      tensor.attr("_amax_rowwise") = nb::none();
    }
  }

  // Coerce column-wise data
  if (columnwise_usage) {
    if (!columnwise_data) {
      // enforce 2D shape to avoid [S, B, H] shape and B and be 1
      // and the transposed shape is [H, S, B], so divide last dim by 2 gives zero
      std::vector<int64_t> shape_int64_2d = {static_cast<int64_t>(flat_first_dim),
                                             static_cast<int64_t>(flat_last_dim)};
      const auto transpose_shape_int64 = make_transpose_shape<int64_t>(shape_int64_2d);
      columnwise_data = stable_empty_cuda(convert_shape_for_fp4(transpose_shape_int64), kU8);
      tensor.attr("_columnwise_data") = tensor_to_py(*columnwise_data);
    }
    if (!columnwise_scale_inv) {
      const auto scale_inv_shape = get_scale_shape(shape, true);
      const std::vector<int64_t> scale_inv_shape_int64(scale_inv_shape.begin(),
                                                       scale_inv_shape.end());
      columnwise_scale_inv = stable_empty_cuda(scale_inv_shape_int64, kU8);
      tensor.attr("_columnwise_scale_inv") = tensor_to_py(*columnwise_scale_inv);
    }
    if (!amax_columnwise) {
      // hadamard amax kernel will zero out pointer with ZeroAmaxKernel
      // nvte_compute_amax_with_config will zero out the pointer if needed
      amax_columnwise = stable_empty_cuda({1}, kF32);
      tensor.attr("_amax_columnwise") = tensor_to_py(*amax_columnwise);
    }
  } else {  // columnwise_usage == false
    if (columnwise_data) {
      columnwise_data.reset();
      tensor.attr("_columnwise_data") = nb::none();
    }
    if (columnwise_scale_inv) {
      columnwise_scale_inv.reset();
      tensor.attr("_columnwise_scale_inv") = nb::none();
    }
    if (amax_columnwise) {
      amax_columnwise.reset();
      tensor.attr("_amax_columnwise") = nb::none();
    }
  }

  // Construct C++ tensor
  TensorWrapper out_cpp(NVTE_NVFP4_1D_SCALING);
  if (rowwise_usage) {
    out_cpp.set_rowwise_data(rowwise_data->data_ptr(), DType::kFloat4E2M1, shape);
    out_cpp.set_rowwise_scale_inv(rowwise_scale_inv->data_ptr(), DType::kFloat8E4M3,
                                  getTensorShape(*rowwise_scale_inv));
    out_cpp.set_amax(amax_rowwise->data_ptr(), DType::kFloat32, getTensorShape(*amax_rowwise));
  }
  if (columnwise_usage) {
    // enforce 2D shape to avoid [S, B, H] shape and B and be 1
    // and the transposed shape is [H, S, B], so divide last dim by 2 gives zero
    std::vector<size_t> shape_2d = {flat_first_dim, flat_last_dim};
    auto col_data_shape_fp4 = make_transpose_shape<size_t>(shape_2d);
    out_cpp.set_columnwise_data(columnwise_data->data_ptr(), DType::kFloat4E2M1,
                                col_data_shape_fp4);
    out_cpp.set_columnwise_scale_inv(columnwise_scale_inv->data_ptr(), DType::kFloat8E4M3,
                                     getTensorShape(*columnwise_scale_inv));
    out_cpp.set_columnwise_amax(amax_columnwise->data_ptr(), DType::kFloat32,
                                std::vector<size_t>{1});
  }
  out_cpp.set_with_gemm_swizzled_scales(with_gemm_swizzled_scales);
  out_cpp.set_row_scaled_nvfp4(row_scaled_nvfp4);
  out_cpp.set_nvfp4_e4m3_max(nvfp4_e4m3_max);
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(tensor)};
}

void NVFP4Quantizer::quantize_with_rht_unfused_helper(
    const TensorWrapper& input, TensorWrapper& out, TensorWrapper& rht_output_t_cpp,
    QuantizationConfigWrapper& quant_config, QuantizationConfigWrapper& quant_config_columnwise,
    cudaStream_t stream) {
  // The kernels invoked below reject swizzled-SF output, so trip a clear
  // error here before reaching them.
  NVTE_CHECK(!out.get_with_gemm_swizzled_scales(),
             "NVFP4 RHT-unfused fallback path does not support "
             "with_gemm_swizzled_scales=True. Either disable optimize_for_gemm on the "
             "quantizer, or ensure the input shape is eligible for RHT cast-fusion "
             "(bf16 dtype + rows%64==0 + cols%128==0 + SM 100/110).");

  if (rowwise_usage) {
    // For rowwise usage, we need to quantize the input directly, but we need to avoid quantizing columnwise
    TensorWrapper out_identity(out.scaling_mode());
    auto out_identity_data = out.get_rowwise_data();
    auto out_identity_scale_inv = out.get_rowwise_scale_inv();
    auto out_identity_amax = out.get_amax();
    out_identity.set_rowwise_data(out_identity_data.data_ptr,
                                  static_cast<DType>(out_identity_data.dtype),
                                  out_identity_data.shape);
    out_identity.set_rowwise_scale_inv(out_identity_scale_inv.data_ptr,
                                       static_cast<DType>(out_identity_scale_inv.dtype),
                                       out_identity_scale_inv.shape);
    out_identity.set_amax(out_identity_amax.data_ptr, static_cast<DType>(out_identity_amax.dtype),
                          out_identity_amax.shape);

    NVTE_SCOPED_GIL_RELEASE(
        { nvte_quantize_v2(input.data(), out_identity.data(), quant_config, stream); });
  }

  if (columnwise_usage) {
    // Get the output columnwise data, scale_inv, and amax
    auto out_columnwise_data = out.get_columnwise_data();
    auto out_columnwise_scale_inv = out.get_columnwise_scale_inv();
    // NOTE: should already be populated.
    auto out_columnwise_amax = out.get_columnwise_amax();

    // Flatten column-wise data shape to 2D to avoid problems when
    // converting between FP4 tensor shape and byte tensor shape
    // (involves dividing last dim by 2).
    auto [flat_first_dim, flat_last_dim] = get_2d_dims(out_columnwise_data.shape, true);
    std::vector<size_t> colwise_data_shape_2d = {flat_first_dim, flat_last_dim};

    // Create a wrapper for the columnwise output, as the rowwise output.
    // The reason is due to the input `rht_output_t` is already in the transposed layout.
    // Thus, we only need a rowwise quantization to generate the columnwise output.
    TensorWrapper out_transpose(out.scaling_mode());
    out_transpose.set_rowwise_data(out_columnwise_data.data_ptr,
                                   static_cast<DType>(out_columnwise_data.dtype),
                                   colwise_data_shape_2d);
    out_transpose.set_rowwise_scale_inv(out_columnwise_scale_inv.data_ptr,
                                        static_cast<DType>(out_columnwise_scale_inv.dtype),
                                        out_columnwise_scale_inv.shape);
    out_transpose.set_amax(out_columnwise_amax.data_ptr,
                           static_cast<DType>(out_columnwise_amax.dtype),
                           out_columnwise_amax.shape);

    // Invoking fallback RHT kernel unfused.
    NVTE_SCOPED_GIL_RELEASE({
      // Perform the RHT(input.t), and write to rht_output_cpp.columnwise.
      nvte_hadamard_transform(input.data(), rht_output_t_cpp.data(), 0,
                              this->rht_matrix_random_sign_mask_t, stream);
    });

    // Quantize kernel will treat everything as rowwise input/output, which is
    // intended.
    NVTE_SCOPED_GIL_RELEASE({
      nvte_quantize_v2(rht_output_t_cpp.data(), out_transpose.data(), quant_config_columnwise,
                       stream);
    });
  }
}

void NVFP4Quantizer::quantize_impl(const TensorWrapper& input, TensorWrapper& out,
                                   const std::optional<TensorWrapper>& noop_flag,
                                   bool compute_amax) {
  auto reduce_amaxes = [&]() {
    if (!this->with_amax_reduction) {
      return;
    }

    std::vector<ts::Tensor> amax_tensors;
    auto make_amax_tensor = [](void* data_ptr) {
      NVTE_CHECK(data_ptr != nullptr, "Could not find amax pointer for NVFP4 amax reduction.");
      // Non-owning stable Tensor view over the amax scalar (contiguous 1-elem).
      return ts::from_blob(data_ptr, {1}, {1}, cuda_device(), kF32);
    };
    if (rowwise_usage) {
      amax_tensors.push_back(make_amax_tensor(out.get_amax().data_ptr));
    }
    if (columnwise_usage) {
      amax_tensors.push_back(make_amax_tensor(out.get_columnwise_amax().data_ptr));
    }
    if (amax_tensors.empty()) {
      return;
    }

    NVTE_SCOPED_GIL_RELEASE(
        { this->amax_reduction_group->allreduce_coalesced(amax_tensors, ts::ReduceOp::MAX).wait(); });
  };

  // Nothing to be done if input is empty
  if (input.numel() == 0) {
    if (!compute_amax) {
      reduce_amaxes();
    }
    return;
  }

  auto stream = current_cuda_stream();

  QuantizationConfigWrapper quant_config;
  QuantizationConfigWrapper quant_config_columnwise;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
    quant_config_columnwise.set_noop_tensor(noop_flag->data());
  }
  quant_config.set_nvfp4_2d_quantization(this->with_2d_quantization);
  quant_config.set_stochastic_rounding(this->stochastic_rounding);
  quant_config.set_nvfp4_4over6_mode(this->nvfp4_4over6_mode);
  quant_config_columnwise.set_nvfp4_4over6_mode(this->nvfp4_4over6_mode);

  if (this->nvfp4_4over6_mode != kNVTENVFP44Over6Disabled) {
    NVTE_CHECK(!this->with_rht, "NVFP4 4over6 quantization does not support RHT.");
    NVTE_CHECK(!this->stochastic_rounding,
               "NVFP4 4over6 quantization does not support stochastic rounding.");
  }

  // We only need RHT for columnwise usage.
  // flat first dim and last dim for multi dimensional input
  const auto [rows, cols] = get_2d_dims(input.shape());

  const bool row_scaled_nvfp4 = out.get_row_scaled_nvfp4();
  if (row_scaled_nvfp4) {
    NVTE_CHECK(!this->with_rht, "Row-scaled NVFP4 quantization does not support RHT.");
    NVTE_CHECK(!this->with_2d_quantization,
               "Row-scaled NVFP4 quantization does not support 2D quantization.");
    NVTE_CHECK(!this->stochastic_rounding,
               "Row-scaled NVFP4 quantization does not support stochastic rounding.");
    NVTE_CHECK(!this->with_amax_reduction,
               "Row-scaled NVFP4 quantization does not support amax reduction.");
    NVTE_CHECK(cols % 16 == 0, "Row-scaled NVFP4 quantization requires last dim divisible by 16.");
  }

  // Restriction for the RHT cast fusion kernel because we are using MMA hardware for computing RHT
  const bool eligible_for_rht_cast_fusion =
      input.dtype() == DType::kBFloat16 &&
      NVFP4Quantizer::is_eligible_for_rht_cast_fusion(convertShape(input.shape()));

  // Stochastic rounding
  // When both rowwise and columnwise quantization are used with RHT,
  // we need separate RNG states for each to ensure they use different random numbers.
  TensorWrapper te_rng_state;
  TensorWrapper te_rng_state_columnwise;

  // Only need a separate rng state when:
  // 1. Stochastic rounding is enabled
  // 2. RHT is enabled
  // 3. Columnwise usage is enabled
  // 4. Rowwise and columnwise quantization are not fused,
  //    because within a single kernel we can generate two different random numbers for rowwise and columnwise
  const bool need_separate_columnwise_rng = this->stochastic_rounding && this->with_rht &&
                                            this->columnwise_usage &&
                                            (!eligible_for_rht_cast_fusion);

  if (this->stochastic_rounding) {
    const size_t rng_elts_per_thread = 1024;  // Wild guess, probably can be tightened
    auto gen = get_cuda_generator(std::nullopt);

    // Generate RNG state for rowwise quantization
    auto philox_args = init_philox_state(gen, rng_elts_per_thread);
    auto rng_state = stable_empty_cuda({2}, kI64);
    philox_unpack(philox_args, static_cast<int64_t*>(rng_state.data_ptr()));
    te_rng_state = makeTransformerEngineTensor(rng_state);
    quant_config.set_rng_state(te_rng_state.data());

    // Generate separate RNG state for columnwise quantization
    if (need_separate_columnwise_rng) {
      auto philox_args_columnwise = init_philox_state(gen, rng_elts_per_thread);
      auto rng_state_columnwise = stable_empty_cuda({2}, kI64);
      philox_unpack(philox_args_columnwise, static_cast<int64_t*>(rng_state_columnwise.data_ptr()));
      te_rng_state_columnwise = makeTransformerEngineTensor(rng_state_columnwise);
      quant_config_columnwise.set_stochastic_rounding(true);
      quant_config_columnwise.set_rng_state(te_rng_state_columnwise.data());
      quant_config_columnwise.set_nvfp4_2d_quantization(this->with_2d_quantization);
    }
  }

  // Compute amax.
  if (this->with_rht) {
    if (input.dtype() != DType::kBFloat16) {
      NVTE_ERROR("RHT is only supported for bfloat16 input, got dtype enum value ",
                 static_cast<int>(input.dtype()));
    }
    if (this->with_post_rht_amax) {
      // We need:
      // 1. Rowwise amax = amax for input
      // 2. Columnwise amax = amax for RHT(input.t)
      if (compute_amax) {
        NVTE_SCOPED_GIL_RELEASE({
          nvte_hadamard_transform_amax(input.data(), out.data(), 0,
                                       this->rht_matrix_random_sign_mask_t, stream);
        });
      }
    } else {
      // raise error since it's not supported yet
      NVTE_ERROR(
          "Pre-RHT amax is not supported yet. "
          "Use with_post_rht_amax=true instead.");
    }
  } else {  // Without RHT
    if (compute_amax && !row_scaled_nvfp4) {
      // Amax pointers
      auto rowwise_amax_ptr = out.get_amax().data_ptr;
      auto columnwise_amax_ptr = out.get_columnwise_amax().data_ptr;
      void* amax_ptr = rowwise_amax_ptr != nullptr ? rowwise_amax_ptr : columnwise_amax_ptr;
      NVTE_CHECK(amax_ptr != nullptr, "Could not find amax pointer");

      // Compute amax of input tensor
      out.set_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
      NVTE_SCOPED_GIL_RELEASE(
          { nvte_compute_amax_with_config(input.data(), out.data(), quant_config, stream); });
      out.set_amax(rowwise_amax_ptr, DType::kFloat32, std::vector<size_t>{1});

      // Make sure row-wise and column-wise amaxes match
      if (rowwise_amax_ptr != amax_ptr && rowwise_amax_ptr != nullptr) {
        NVTE_CHECK_CUDA(cudaMemcpyAsync(rowwise_amax_ptr, amax_ptr, sizeof(float),
                                        cudaMemcpyDeviceToDevice, stream));
      }
      if (columnwise_amax_ptr != amax_ptr && columnwise_amax_ptr != nullptr) {
        NVTE_CHECK_CUDA(cudaMemcpyAsync(columnwise_amax_ptr, amax_ptr, sizeof(float),
                                        cudaMemcpyDeviceToDevice, stream));
      }
    }
  }

  reduce_amaxes();

  // Fast math toggle: RHT transform can be accelerated
  // What math is accelerated? Only the high precision math, so numerical impact is minimal
  // 1. replace 1 / x by reciprocal_approximate_ftz(x)
  // 2. when RHT cast fusion is available, fusion allows cast to be performed on FP32 data,
  //    this will essentially remove a round trip between FP32 to BF16 then FP32
  // NVFP4 4over6 candidate error math is controlled separately by
  // NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH.
  const auto use_fast_math = transformer_engine::getenv<bool>("NVTE_USE_FAST_MATH");
  if (use_fast_math && this->nvfp4_4over6_mode == kNVTENVFP44Over6Disabled) {
    quant_config.set_use_fast_math(true);
    quant_config_columnwise.set_use_fast_math(true);
  }

  const auto use_4over6_err_use_fast_math =
      transformer_engine::getenv<bool>("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH");
  if (use_4over6_err_use_fast_math) {
    quant_config.set_nvfp4_4over6_err_use_fast_math(true);
    quant_config_columnwise.set_nvfp4_4over6_err_use_fast_math(true);
  }

  if (this->with_rht) {
    if (eligible_for_rht_cast_fusion) {
      // fusion kernel requires passing in RHT matrix directly for maximum performance
      NVTE_CHECK(this->rht_matrix.defined() && this->rht_matrix.numel() > 0,
                 "RHT matrix is not available.");
      auto rht_matrix_nvte = makeTransformerEngineTensor(this->rht_matrix);
      // Fusion kernel that does the following:
      // 1. Rowwise quantization
      // 2. RHT followed by columnwise quantization & transpose
      NVTE_SCOPED_GIL_RELEASE({
        nvte_quantize_with_hadamard_transform(input.data(), out.data(), rht_matrix_nvte.data(),
                                              quant_config, stream);
      });
    } else {
      // Use separate RNG state for columnwise to ensure different random numbers than rowwise
      // This is only necessary because it's the unfused path where rowwise and columnwise
      // are separate kernel launches
      auto& columnwise_quant_config_to_use =
          need_separate_columnwise_rng ? quant_config_columnwise : quant_config;
      // unfused path also needs memory allocation for intermediate buffer for RHT output
      ts::Tensor rht_output_t;  // The RHT(x_t) output, in columnwise layout
      // This wrapper is going to be passed as input to the quantization kernel.
      TensorWrapper rht_output_t_cpp;  // Wrapper to contain the RHT(x) and RHT(x_t) outputs
      rht_output_t =
          allocateTorchTensor(static_cast<int>(cols), static_cast<int>(rows), input.dtype());
      // NOTE (frsun): This is non-intuitive, we are writing the
      // result of transposed RHT to the output of rowwise.
      rht_output_t_cpp.set_rowwise_data(rht_output_t.data_ptr(), input.dtype(),
                                        std::vector<size_t>{cols, rows});
      this->quantize_with_rht_unfused_helper(input, out, rht_output_t_cpp, quant_config,
                                             columnwise_quant_config_to_use, stream);
    }
  } else {
    NVTE_SCOPED_GIL_RELEASE({ nvte_quantize_v2(input.data(), out.data(), quant_config, stream); });
  }
}

void NVFP4Quantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                              const std::optional<TensorWrapper>& noop_flag) {
  this->quantize_impl(input, out, noop_flag, true);
}

void NVFP4Quantizer::quantize_with_amax(TensorWrapper& input, TensorWrapper& out) {
  NVTE_CHECK(!out.get_row_scaled_nvfp4(),
             "quantize_with_amax is not supported for row-scaled NVFP4 quantization.");
  // Update output tensor amaxes with input tensor amax
  auto input_amax_ptr = input.amax();
  auto output_rowwise_amax_ptr = out.get_amax().data_ptr;
  auto output_columnwise_amax_ptr = out.get_columnwise_amax().data_ptr;
  NVTE_CHECK(input_amax_ptr != nullptr ||
                 (output_rowwise_amax_ptr == nullptr && output_columnwise_amax_ptr == nullptr),
             "Input tensor does not have pre-computed amax");
  if (input_amax_ptr != output_rowwise_amax_ptr && input_amax_ptr != nullptr &&
      output_rowwise_amax_ptr != nullptr) {
    NVTE_CHECK_CUDA(cudaMemcpyAsync(output_rowwise_amax_ptr, input_amax_ptr, sizeof(float),
                                    cudaMemcpyDeviceToDevice, current_cuda_stream()));
  }
  if (input_amax_ptr != output_columnwise_amax_ptr && input_amax_ptr != nullptr &&
      output_columnwise_amax_ptr != nullptr) {
    NVTE_CHECK_CUDA(cudaMemcpyAsync(output_columnwise_amax_ptr, input_amax_ptr, sizeof(float),
                                    cudaMemcpyDeviceToDevice, current_cuda_stream()));
  }
  input.set_amax(nullptr, DType::kFloat32, input.defaultShape);

  // Perform quantization
  this->quantize_impl(input, out, std::nullopt, false);
}

std::vector<size_t> NVFP4Quantizer::get_scale_shape(const std::vector<size_t>& shape,
                                                    bool columnwise) const {
  const auto [flat_first_dim, last_dim] = get_2d_dims(shape);

  NVTE_CHECK(last_dim % NVFP4_BLOCK_SIZE == 0, "Last dim for NVFP4 must be divisible by ",
             NVFP4_BLOCK_SIZE, " (got dim=", last_dim, ")");
  NVTE_CHECK(flat_first_dim % NVFP4_BLOCK_SIZE == 0,
             "NVFP4 requires tensor dims that are divisible by ", NVFP4_BLOCK_SIZE,
             " (got shape=", shape, ")");

  std::vector<size_t> scale_shape;

  bool rowwise_usage = !columnwise;

  if (rowwise_usage) {
    // rowwise scaling factor shape
    size_t sinv0 = roundup(flat_first_dim, 128);
    size_t sinv1 = roundup(last_dim / NVFP4_BLOCK_SIZE, 4);
    scale_shape = {sinv0, sinv1};
  } else {
    // columnwise scaling factor shape
    size_t sinv0 = roundup(last_dim, 128);
    size_t sinv1 = roundup(flat_first_dim / NVFP4_BLOCK_SIZE, 4);
    scale_shape = {sinv0, sinv1};
  }
  return scale_shape;
}

}  // namespace transformer_engine::pytorch
