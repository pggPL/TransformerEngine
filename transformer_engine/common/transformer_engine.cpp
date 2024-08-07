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

void CheckScaleTensor(const SimpleTensor &scale, const ScalingMode &mode, const SimpleTensor &data,
                      const std::string &name, const std::string &suffix) {
  NVTE_CHECK(scale.dptr != nullptr,
             "FP8 scaling factor input " + name + suffix + " must be allocated.");
  NVTE_CHECK(scale.dtype == DType::kFloat32 || scale.dtype == DType::kByte,
             "Unsupported type of scaling factor input " + name + suffix +
                 ". Expected Float32 or Byte, got " + to_string(scale.dtype));
  // Need 4B alignment even for e8 scaling factor
  size_t alignment = 4ul / typeToSize(scale.dtype);
  size_t expected_x;
  if (mode.x == -1) {
    expected_x = 1;
  } else {
    NVTE_CHECK(data.shape.size() == 2, "Invalid shape of the tensor " + name +
                                           ". Expected 2 dimensions for fine granularity scaling.");
    expected_x = DIVUP(DIVUP(data.shape.at(0), static_cast<size_t>(mode.x)), alignment);
  }
  size_t expected_y;
  if (mode.y == -1) {
    expected_y = 1;
  } else {
    NVTE_CHECK(data.shape.size() == 2, "Invalid shape of the tensor " + name +
                                           ". Expected 2 dimensions for fine granularity scaling.");
    expected_y = DIVUP(DIVUP(data.shape.at(1), static_cast<size_t>(mode.y)), alignment);
  }
  if (expected_x == 1 && expected_y == 1) {
    // per-tensor scaling
    NVTE_CHECK(scale.shape == std::vector<size_t>{1});
  } else {
    const auto &expected = std::vector<size_t>{expected_x, expected_y};
    NVTE_CHECK(scale.shape == expected, "Tensor " + name + suffix +
                                            " has invalid shape. Expected (" + to_string(expected) +
                                            ", while got " + to_string(scale.shape));
  }
}

void CheckInputTensor(const Tensor &t, const std::string &name) {
  const DType type = t.data.dtype;
  if (is_fp8_dtype(type)) {
    // FP8 input needs to have scale_inv
    CheckScaleTensor(t.scale_inv, t.scaling_mode, t.data, name, "_scale_inverse");
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
    CheckScaleTensor(t.scale_inv, t.scaling_mode, t.data, name, "_scale_inverse");
    CheckScaleTensor(t.scale, t.scaling_mode, t.data, name, "_scale");
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

bool is_columnwise_block_scaling(const Tensor *t) {
  ScalingMode mode = t->scaling_mode;
  auto block_size = mode.x;
  bool columnwise_block_scaling = mode.y == 1 && !(mode.delayed_scaling);
  if (columnwise_block_scaling) {
    auto nelem = t->numel();
    NVTE_CHECK(nelem % block_size == 0, "Incorrect number of inputs elements ", nelem,
               " for ", block_size, " block scaling.");
  }
  if (block_size != 32) {
    NVTE_ERROR("Block size not supported.");
  }
  return columnwise_block_scaling;
}

}  // namespace transformer_engine

NVTETensor nvte_create_tensor(void *dptr, const NVTEShape shape, const NVTEDType dtype, float *amax,
                              float *scale, float *scale_inv, NVTEScalingMode scaling_mode) {
  transformer_engine::Tensor *ret = new transformer_engine::Tensor;
  ret->data.dptr = dptr;
  ret->data.shape = std::vector<size_t>(shape.data, shape.data + shape.ndim);
  ret->data.dtype = static_cast<transformer_engine::DType>(dtype);
  ret->amax.dptr = amax;
  ret->scale.dptr = scale;
  ret->scale_inv.dptr = scale_inv;
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
  NVTE_CHECK(t.scale_inv.dtype == transformer_engine::DType::kFloat32,
             "Tensor's inverse of scale must have Float32 type!");
  return reinterpret_cast<float *>(t.scale_inv.dptr);
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
