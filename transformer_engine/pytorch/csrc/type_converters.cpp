/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <nanobind/nanobind.h>
#include <transformer_engine/transformer_engine.h>

#include <torch/csrc/stable/python/interop.h>
#include <torch/csrc/stable/tensor.h>

#include "common.h"
#include "pybind.h"

namespace transformer_engine::pytorch {
namespace detail {

TensorWrapper NVTETensorFromFloat8Tensor(nb::handle tensor, Quantizer *quantizer) {
  auto ret = TensorWrapper(quantizer->get_scaling_mode());

  bool data_exists = !tensor.attr("_data").is_none();
  bool transpose_exists =
      !nb::cast<bool>(tensor.attr("_transpose_invalid")) && !tensor.attr("_transpose").is_none();

  NVTE_CHECK(data_exists || transpose_exists, "No data found for FP8 Tensor.");

  // FP8 data
  const DType fp8_dtype = static_cast<DType>(nb::cast<int>(tensor.attr("_fp8_dtype")));
  if (data_exists) {
    const auto &data = torch::stable::from_pyobject(tensor.attr("_data").ptr());
    ret.set_rowwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
  }

  // FP8 data transpose
  if (transpose_exists) {
    const auto &data_transpose = torch::stable::from_pyobject(tensor.attr("_transpose").ptr());
    ret.set_columnwise_data(data_transpose.data_ptr(), fp8_dtype, getTensorShape(data_transpose));
  }

  // Scale-inverse
  {
    const auto &scale_inv = torch::stable::from_pyobject(tensor.attr("_scale_inv").ptr());
    float *dptr = reinterpret_cast<float *>(scale_inv.data_ptr());
    const auto &dtype = GetTransformerEngineDType(scale_inv.scalar_type());
    const auto &shape = getTensorShape(scale_inv);
    ret.set_rowwise_scale_inv(dptr, dtype, shape);
    ret.set_columnwise_scale_inv(dptr, dtype, shape);
  }

  // Quantizer state
  quantizer->set_quantization_params(&ret);

  return ret;
}

TensorWrapper NVTETensorFromMXFP8Tensor(nb::handle tensor, Quantizer *quantizer) {
  auto ret = TensorWrapper(NVTE_MXFP8_1D_SCALING);

  const bool rowwise_usage = !(tensor.attr("_rowwise_data").is_none());
  const bool columnwise_usage = !(tensor.attr("_columnwise_data").is_none());
  const bool with_gemm_swizzled_scales = nb::cast<bool>(tensor.attr("_with_gemm_swizzled_scales"));

  NVTE_CHECK(rowwise_usage || columnwise_usage, "No data found for MXFP8 Tensor.");

  // Row-scaled data
  const DType fp8_dtype = static_cast<DType>(nb::cast<int>(tensor.attr("_fp8_dtype")));
  if (rowwise_usage) {
    const auto &data = torch::stable::from_pyobject(tensor.attr("_rowwise_data").ptr());
    const auto &scale_inv = torch::stable::from_pyobject(tensor.attr("_rowwise_scale_inv").ptr());
    ret.set_rowwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0, getTensorShape(scale_inv));
  }

  // Column-scaled data
  if (columnwise_usage) {
    const auto &data = torch::stable::from_pyobject(tensor.attr("_columnwise_data").ptr());
    const auto &scale_inv = torch::stable::from_pyobject(tensor.attr("_columnwise_scale_inv").ptr());
    ret.set_columnwise_data(data.data_ptr(), fp8_dtype, getTensorShape(data));
    ret.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0,
                                 getTensorShape(scale_inv));
  }

  // Scale layout
  ret.set_with_gemm_swizzled_scales(with_gemm_swizzled_scales);

  // Quantizer state
  quantizer->set_quantization_params(&ret);

  return ret;
}

TensorWrapper NVTETensorFromFloat8BlockwiseQTensor(nb::handle tensor, Quantizer *quantizer) {
  const DType dtype = static_cast<DType>(nb::cast<int>(tensor.attr("_fp8_dtype")));
  bool is_2D_scaled = nb::cast<bool>(tensor.attr("_is_2D_scaled"));

  bool rowwise_usage = !(tensor.attr("_rowwise_data").is_none());
  bool columnwise_usage = !(tensor.attr("_columnwise_data").is_none());

  auto ret = TensorWrapper(is_2D_scaled ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D);

  // Row-wise data
  if (rowwise_usage) {
    const torch::stable::Tensor &data_rowwise = torch::stable::from_pyobject(tensor.attr("_rowwise_data").ptr());
    const torch::stable::Tensor &scale_inv_rowwise = torch::stable::from_pyobject(tensor.attr("_rowwise_scale_inv").ptr());
    void *scale_inv_rowwise_dptr = scale_inv_rowwise.data_ptr();
    const auto &rowwise_shape = getTensorShape(data_rowwise);
    ret.set_rowwise_data(data_rowwise.data_ptr(), dtype, rowwise_shape);
    const auto scale_inv_rowwise_shape = getTensorShape(scale_inv_rowwise);
    ret.set_rowwise_scale_inv(scale_inv_rowwise_dptr, DType::kFloat32, scale_inv_rowwise_shape);
  }

  // Column-wise data
  if (columnwise_usage) {
    const torch::stable::Tensor &data_colwise = torch::stable::from_pyobject(tensor.attr("_columnwise_data").ptr());
    const torch::stable::Tensor &scale_inv_colwise = torch::stable::from_pyobject(tensor.attr("_columnwise_scale_inv").ptr());
    void *scale_inv_colwise_dptr = scale_inv_colwise.data_ptr();
    const auto &shape = getTensorShape(data_colwise);
    ret.set_columnwise_data(data_colwise.data_ptr(), dtype, shape);

    const auto scale_inv_colwise_shape = getTensorShape(scale_inv_colwise);
    ret.set_columnwise_scale_inv(scale_inv_colwise_dptr, DType::kFloat32, scale_inv_colwise_shape);
  }

  // Quantizer state
  quantizer->set_quantization_params(&ret);

  return ret;
}

TensorWrapper NVTETensorFromNVFP4Tensor(nb::handle tensor, Quantizer *quantizer) {
  const DType dtype = static_cast<DType>(nb::cast<int>(tensor.attr("_fp4_dtype")));

  auto ret = TensorWrapper(NVTE_NVFP4_1D_SCALING);

  const bool rowwise_usage = !(tensor.attr("_rowwise_data").is_none());
  const bool columnwise_usage = !(tensor.attr("_columnwise_data").is_none());
  const bool with_gemm_swizzled_scales = nb::cast<bool>(tensor.attr("_with_gemm_swizzled_scales"));
  const bool row_scaled_nvfp4 = nb::cast<bool>(tensor.attr("_row_scaled_nvfp4"));
  const int nvfp4_e4m3_max = nb::cast<int>(tensor.attr("_nvfp4_e4m3_max"));

  NVTE_CHECK(rowwise_usage || columnwise_usage, "No data found for NVFP4 Tensor.");

  // Row-scaled data
  if (rowwise_usage) {
    const auto &data = torch::stable::from_pyobject(tensor.attr("_rowwise_data").ptr());
    const auto &scale_inv = torch::stable::from_pyobject(tensor.attr("_rowwise_scale_inv").ptr());
    const auto &amax_rowwise = torch::stable::from_pyobject(tensor.attr("_amax_rowwise").ptr());
    ret.set_rowwise_data(data.data_ptr(), dtype,
                         convert_shape_back_from_fp4(getTensorShape(data), false));
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3, getTensorShape(scale_inv));
    ret.set_amax(amax_rowwise.data_ptr(), DType::kFloat32, getTensorShape(amax_rowwise));
  }

  // Column-scaled data
  if (columnwise_usage) {
    const auto &data = torch::stable::from_pyobject(tensor.attr("_columnwise_data").ptr());
    const auto &scale_inv = torch::stable::from_pyobject(tensor.attr("_columnwise_scale_inv").ptr());
    const auto &amax_columnwise = torch::stable::from_pyobject(tensor.attr("_amax_columnwise").ptr());
    ret.set_columnwise_data(data.data_ptr(), DType::kFloat4E2M1,
                            convert_shape_back_from_fp4(getTensorShape(data), false));
    ret.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3,
                                 getTensorShape(scale_inv));
    ret.set_columnwise_amax(amax_columnwise.data_ptr(), DType::kFloat32,
                            getTensorShape(amax_columnwise));
  }

  // Scale layout
  ret.set_with_gemm_swizzled_scales(with_gemm_swizzled_scales);
  ret.set_row_scaled_nvfp4(row_scaled_nvfp4);
  ret.set_nvfp4_e4m3_max(nvfp4_e4m3_max);

  // Quantizer state
  quantizer->set_quantization_params(&ret);

  return ret;
}

NVTEScalingMode ScalingModeFromQuantizer(nb::handle quantizer) {
  auto *quantizer_ptr = quantizer.ptr();
  if (IsMXFP8Quantizers(quantizer_ptr)) {
    return NVTE_MXFP8_1D_SCALING;
  }
  if (IsNVFP4Quantizers(quantizer_ptr)) {
    return NVTE_NVFP4_1D_SCALING;
  }
  if (IsFloat8BlockwiseQuantizers(quantizer_ptr)) {
    const int block_scaling_dim = nb::cast<int>(quantizer.attr("block_scaling_dim"));
    return (block_scaling_dim == 2) ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D;
  }
  return NVTE_DELAYED_TENSOR_SCALING;
}

DType GetTransformerEngineDTypeForScaleInv(nb::handle quantizer, torch::stable::Tensor scale_inv) {
  auto *quantizer_ptr = quantizer.ptr();
  if (IsMXFP8Quantizers(quantizer_ptr)) {
    return DType::kFloat8E8M0;
  }
  if (IsFloat8BlockwiseQuantizers(quantizer_ptr)) {
    return DType::kFloat32;
  }
  if (IsNVFP4Quantizers(quantizer_ptr)) {
    return DType::kFloat8E4M3;
  }
  return GetTransformerEngineDType(scale_inv.scalar_type());
}

GroupedTensorWrapper GroupedTensorFromPyTorchGroupedTensor(nb::handle tensor) {
  // Returns a GroupedTensorWrapper from a PyTorch GroupedTensor.
  const auto num_tensors = nb::cast<size_t>(tensor.attr("num_tensors"));
  const auto logical_shape = nb::cast<std::vector<size_t>>(tensor.attr("logical_shape"));
  nb::handle quantizer = nb::none();
  DType quantizer_dtype = DType::kNumTypes;
  NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
  if (!tensor.attr("quantizer").is_none()) {
    quantizer = tensor.attr("quantizer");
    if (!quantizer.is_none()) {
      scaling_mode = ScalingModeFromQuantizer(quantizer);
      quantizer_dtype = static_cast<DType>(nb::cast<int>(quantizer.attr("dtype")));
    }
  }
  auto ret = GroupedTensorWrapper(num_tensors, logical_shape, scaling_mode);

  // Rowwise data
  if (!tensor.attr("rowwise_data").is_none()) {
    const auto &data = torch::stable::from_pyobject(tensor.attr("rowwise_data").ptr());
    DType data_dtype =
        quantizer.is_none() ? GetTransformerEngineDType(data.scalar_type()) : quantizer_dtype;
    ret.set_rowwise_data(data.data_ptr(), data_dtype, getTensorShape(data));
  } else if (quantizer_dtype != DType::kNumTypes) {
    ret.set_rowwise_data(nullptr, quantizer_dtype, std::vector<size_t>{0});
  }

  // Columnwise data
  if (!tensor.attr("columnwise_data").is_none()) {
    const auto &data = torch::stable::from_pyobject(tensor.attr("columnwise_data").ptr());
    DType data_dtype =
        quantizer.is_none() ? GetTransformerEngineDType(data.scalar_type()) : quantizer_dtype;
    ret.set_columnwise_data(data.data_ptr(), data_dtype, getTensorShape(data));
  } else if (quantizer_dtype != DType::kNumTypes) {
    ret.set_columnwise_data(nullptr, quantizer_dtype, std::vector<size_t>{0});
  }

  // Scale
  if (!tensor.attr("scale").is_none()) {
    const auto &scale = torch::stable::from_pyobject(tensor.attr("scale").ptr());
    ret.set_scale(scale.data_ptr(), GetTransformerEngineDType(scale.scalar_type()),
                  getTensorShape(scale));
  }

  // Amax
  if (!tensor.attr("amax").is_none()) {
    const auto &amax = torch::stable::from_pyobject(tensor.attr("amax").ptr());
    ret.set_amax(amax.data_ptr(), GetTransformerEngineDType(amax.scalar_type()),
                 getTensorShape(amax));
  }
  if (!tensor.attr("columnwise_amax").is_none()) {
    const auto &amax = torch::stable::from_pyobject(tensor.attr("columnwise_amax").ptr());
    ret.set_columnwise_amax(amax.data_ptr(), GetTransformerEngineDType(amax.scalar_type()),
                            getTensorShape(amax));
  }

  // Scale inverse
  if (!tensor.attr("scale_inv").is_none()) {
    const auto &scale_inv = torch::stable::from_pyobject(tensor.attr("scale_inv").ptr());
    ret.set_rowwise_scale_inv(scale_inv.data_ptr(),
                              GetTransformerEngineDTypeForScaleInv(quantizer, scale_inv),
                              getTensorShape(scale_inv));
  }
  if (!tensor.attr("columnwise_scale_inv").is_none()) {
    const auto &scale_inv = torch::stable::from_pyobject(tensor.attr("columnwise_scale_inv").ptr());
    ret.set_columnwise_scale_inv(scale_inv.data_ptr(),
                                 GetTransformerEngineDTypeForScaleInv(quantizer, scale_inv),
                                 getTensorShape(scale_inv));
  }

  // Shape metadata
  if (!tensor.attr("first_dims").is_none()) {
    const auto &first_dims = torch::stable::from_pyobject(tensor.attr("first_dims").ptr());
    ret.set_first_dims(first_dims.data_ptr(), GetTransformerEngineDType(first_dims.scalar_type()),
                       getTensorShape(first_dims));
  }
  if (!tensor.attr("last_dims").is_none()) {
    const auto &last_dims = torch::stable::from_pyobject(tensor.attr("last_dims").ptr());
    ret.set_last_dims(last_dims.data_ptr(), GetTransformerEngineDType(last_dims.scalar_type()),
                      getTensorShape(last_dims));
  }
  if (!tensor.attr("tensor_offsets").is_none()) {
    const auto &tensor_offsets = torch::stable::from_pyobject(tensor.attr("tensor_offsets").ptr());
    ret.set_tensor_offsets(tensor_offsets.data_ptr(),
                           GetTransformerEngineDType(tensor_offsets.scalar_type()),
                           getTensorShape(tensor_offsets));
  }

  bool with_gemm_swizzled = false;
  if (nb::hasattr(tensor, "_with_gemm_swizzled_scales")) {
    with_gemm_swizzled = nb::cast<bool>(tensor.attr("_with_gemm_swizzled_scales"));
  }
  ret.set_with_gemm_swizzled_scales(with_gemm_swizzled);

  return ret;
}

}  // namespace detail

}  // namespace transformer_engine::pytorch
