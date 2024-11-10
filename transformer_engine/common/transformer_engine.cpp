/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>

#include "common.h"

namespace transformer_engine {

size_t typeToSize(const DType type) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(type, T,
                                     return TypeInfo<T>::size;);  // NOLINT(*)
}

bool is_fp8_dtype(const DType t) { return t == DType::kFloat8E4M3 || t == DType::kFloat8E5M2; }

std::string to_string(const DType type) {
  switch (type) {
    case DType::kByte:
      return "Byte";
    case DType::kBFloat16:
      return "BFloat16";
    case DType::kFloat16:
      return "Float16";
    case DType::kFloat32:
      return "Float32";
    case DType::kFloat8E4M3:
      return "Float8E4M3";
    case DType::kFloat8E5M2:
      return "Float8E5M2";
    case DType::kInt32:
      return "Int32";
    case DType::kInt64:
      return "Int64";
    default:
      return "Invalid type " + std::to_string(static_cast<int>(type));
  }
}

std::string to_string(const ScalingMode &mode) {
  return "{(" + std::to_string(mode.x) + ", " + std::to_string(mode.y) + "), " +
         std::to_string(static_cast<bool>(mode.delayed_scaling)) + "}";
}

void CheckScaleTensor(const Tensor *t) {
  // Need (4, 128) alignment even for e8 scaling factor
  auto block_alignment = std::vector<size_t>{4ul / typeToSize(t->scale_inv.dtype),
                                             128ul / typeToSize(t->scale_inv.dtype)};
  size_t expected_x, expected_y, alignment;
  if (t->scaling_mode.x == -1) {
    expected_x = 1;
  } else {
    NVTE_CHECK(t->data.shape.size() == 2,
               "Invalid shape of the tensor. Expected 2 dimensions for fine granularity scaling.");
    alignment = block_alignment[t->scaling_mode.x < t->scaling_mode.y];
    expected_x =
        DIVUP(DIVUP(t->data.shape.at(0), static_cast<size_t>(t->scaling_mode.x)), alignment) *
        alignment;
  }
  if (t->scaling_mode.y == -1) {
    expected_y = 1;
  } else {
    NVTE_CHECK(t->data.shape.size() == 2,
               "Invalid shape of the tensor. Expected 2 dimensions for fine granularity scaling.");
    alignment = block_alignment[t->scaling_mode.x > t->scaling_mode.y];
    expected_y =
        DIVUP(DIVUP(t->data.shape.at(1), static_cast<size_t>(t->scaling_mode.y)), alignment) *
        alignment;
  }
  if (expected_x == 1 && expected_y == 1) {
    // per-tensor scaling
    NVTE_CHECK(t->scale_inv.shape == std::vector<size_t>{1});
  } else {
    const auto &expected = std::vector<size_t>{expected_x, expected_y};
    NVTE_CHECK(t->scale_inv.shape == expected, "Tensor has invalid scale_inv shape. Expected (" +
                                                   to_string(expected) + ", while got " +
                                                   to_string(t->scale_inv.shape));
  }
}

void CheckInputTensor(const Tensor &t, const std::string &name) {
  const DType type = t.data.dtype;
  if (is_fp8_dtype(type)) {
    // FP8 input needs to have scale_inv
    NVTE_CHECK(t.scale_inv.dptr != nullptr,
               "FP8 scaling factor input " + name + "_scale_inverse" + " must be allocated.");
    NVTE_CHECK(t.scale_inv.dtype == DType::kFloat32 || t.scale_inv.dtype == DType::kByte,
               "Unsupported type of scaling factor input " + name + "_scale_inverse" +
                   ". Expected Float32 or Byte, got " + to_string(t.scale_inv.dtype));
  } else {
    NVTE_CHECK(t.scale.dptr == nullptr, "Scale is not supported for non-FP8 input " + name + ".");
    NVTE_CHECK(t.amax.dptr == nullptr, "Amax is not supported for non-FP8 input " + name + ".");
    NVTE_CHECK(t.scale_inv.dptr == nullptr,
               "Scale_inv is not supported for non-FP8 input " + name + ".");
  }
  NVTE_CHECK(t.data.dptr != nullptr, "Input " + name + " is not allocated!");
}

void CheckOutputTensor(const Tensor &t, const std::string &name, bool allow_empty) {
  const DType type = t.data.dtype;
  if (is_fp8_dtype(type)) {
    // FP8 output needs to have scale, scale_inv and (if delayed scaling) amax
    if (t.scaling_mode.delayed_scaling) {
      NVTE_CHECK(t.amax.dptr != nullptr, "FP8 output " + name + " must have amax tensor.");
      NVTE_CHECK(t.amax.dtype == DType::kFloat32);
      NVTE_CHECK(t.amax.shape == std::vector<size_t>{1});
    }
    NVTE_CHECK(t.scale_inv.dptr != nullptr,
               "FP8 scaling factor input " + name + "_scale_inverse" + " must be allocated.");
    NVTE_CHECK(t.scale_inv.dtype == DType::kFloat32 || t.scale_inv.dtype == DType::kByte,
               "Unsupported type of scaling factor input " + name + "_scale_inverse" +
                   ". Expected Float32 or Byte, got " + to_string(t.scale_inv.dtype));
  } else {
    NVTE_CHECK(t.scale.dptr == nullptr, "Scale is not supported for non-FP8 output " + name + ".");
    NVTE_CHECK(t.amax.dptr == nullptr, "Amax is not supported for non-FP8 output " + name + ".");
    NVTE_CHECK(t.scale_inv.dptr == nullptr,
               "Scale_inv is not supported for non-FP8 output " + name + ".");
  }

  if (!allow_empty) {
    NVTE_CHECK(t.data.dptr != nullptr, "Output " + name + " is not allocated!");
  }
}

bool is_block_scaling(const Tensor *t) {
  auto mode = t->scaling_mode;
  auto colwise_block_scaling = is_columnwise_block_scaling(mode);
  auto rowwise_block_scaling = is_rowwise_block_scaling(mode);
  int block_size = (colwise_block_scaling) ? mode.x : mode.y;

  auto nelem = t->numel();
  NVTE_CHECK(nelem % block_size == 0, "Incorrect number of inputs elements ", nelem, " for ",
             block_size, " block scaling.");
  NVTE_CHECK(block_size == 32, "Unsupported block size ", block_size, " provided.");

  return rowwise_block_scaling || colwise_block_scaling;
}


}  // namespace transformer_engine

NVTETensor nvte_create_tensor(NVTEScalingMode scaling_mode) {
  transformer_engine::Tensor *ret = new transformer_engine::Tensor;
  ret->scaling_mode = scaling_mode;
  return ret;
}

void nvte_destroy_tensor(NVTETensor tensor) {
  if (tensor == nullptr) return;
  auto *t = reinterpret_cast<transformer_engine::Tensor *>(tensor);
  delete t;
}

NVTEDType nvte_tensor_type(const NVTETensor tensor) {
  return static_cast<NVTEDType>(
      reinterpret_cast<const transformer_engine::Tensor *>(tensor)->data.dtype);
}

NVTEShape nvte_tensor_shape(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  NVTEShape ret;
  ret.data = t.data.shape.data();
  ret.ndim = t.data.shape.size();
  return ret;
}

size_t nvte_tensor_ndim(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.data.shape.size();
}

size_t nvte_tensor_size(const NVTETensor tensor, const size_t dim) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  NVTE_CHECK(dim >= 0 && dim < t.data.shape.size(), "Invalid dimension index: ", dim);
  return t.data.shape[dim];
}

size_t nvte_tensor_numel(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  size_t numel = 1;
  for (auto size : t.data.shape) {
    numel *= size;
  }
  return numel;
}

size_t nvte_tensor_element_size(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return transformer_engine::typeToSize(t.data.dtype);
}

void *nvte_tensor_data(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.data.dptr;
}

float *nvte_tensor_amax(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  NVTE_CHECK(t.amax.dtype == transformer_engine::DType::kFloat32,
             "Tensor's amax must have Float32 type!");
  return reinterpret_cast<float *>(t.amax.dptr);
}

float *nvte_tensor_scale(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  NVTE_CHECK(t.scale.dtype == transformer_engine::DType::kFloat32,
             "Tensor's scale must have Float32 type!");
  return reinterpret_cast<float *>(t.scale.dptr);
}

float *nvte_tensor_scale_inv(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  if (transformer_engine::is_tensor_scaling(t.scaling_mode))
    NVTE_CHECK(t.scale_inv.dtype == transformer_engine::DType::kFloat32,
               "Tensor's inverse of scale must have Float32 type!");
  else
    NVTE_CHECK(t.scale_inv.dtype == transformer_engine::DType::kByte,
               "Tensor's inverse of scale must have Byte type!");
  return reinterpret_cast<float *>(t.scale_inv.dptr);
}

NVTEShape nvte_tensor_scale_inv_shape(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  NVTEShape ret;
  ret.data = t.scale_inv.shape.data();
  ret.ndim = t.scale_inv.shape.size();
  return ret;
}

void nvte_set_tensor_param(NVTETensor* tensor,
                           NVTETensorParam param_name,
                           const NVTEBasicTensor* param) {
  auto &t = *reinterpret_cast<transformer_engine::Tensor *>(*tensor);
  switch (param_name) {
    case kNVTERowwiseData:
      t.data = *param;
      break;
    case kNVTEColumnwiseData:
      t.columnwise_data = *param;
      break;
    case kNVTEScale:
      t.scale = *param;
      break;
    case kNVTEAmax:
      t.amax = *param;
      break;
    case kNVTERowwiseScaleInv:
      t.scale_inv = *param;
      break;
    case kNVTEColumnwiseScaleInv:
      t.columnwise_scale_inv = *param;
      break;
    default:
      NVTE_ERROR("Unknown tensor parameter!");
  }
}

NVTEBasicTensor nvte_get_tensor_param(const NVTETensor tensor,
                                      NVTETensorParam param_name) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  switch (param_name) {
    case kNVTERowwiseData:
      return t.data;
    case kNVTEColumnwiseData:
      return t.columnwise_data;
    case kNVTEScale:
      return t.scale;
    case kNVTEAmax:
      return t.amax;
    case kNVTERowwiseScaleInv:
      return t.scale_inv;
    case kNVTEColumnwiseScaleInv:
      return t.columnwise_scale_inv;
    default:
      NVTE_ERROR("Unknown tensor parameter!");
  }
}

NVTEScalingMode nvte_tensor_scaling_mode(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.scaling_mode;
}

void nvte_tensor_pack_create(NVTETensorPack *pack) {
  for (int i = 0; i < pack->MAX_SIZE; i++) {
    pack->tensors[i] = reinterpret_cast<NVTETensor>(new transformer_engine::Tensor);
  }
}

void nvte_tensor_pack_destroy(NVTETensorPack *pack) {
  for (int i = 0; i < pack->MAX_SIZE; i++) {
    auto *t = reinterpret_cast<transformer_engine::Tensor *>(pack->tensors[i]);
    delete t;
  }
}
