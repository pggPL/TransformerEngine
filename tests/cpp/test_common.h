/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <transformer_engine/transformer_engine.h>
#include "util/logging.h"

namespace test {
using namespace transformer_engine;

template <size_t i>
struct BytesToType {};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
};

using byte = uint8_t;
using int32 = int32_t;
using int64 = int64_t;
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;

template <typename T>
struct TypeInfo{
    using types = std::tuple<byte,
                             int32,
                             int64,
                             fp32,
                             fp16,
                             bf16,
                             fp8e4m3,
                             fp8e5m2>;

    template <typename U, DType current>
    struct Helper {
        constexpr static DType getType() {
            constexpr int i = static_cast<int>(current);
            if (std::is_same<U, typename std::tuple_element<i, types>::type>::value) {
                return current;
            } else {
                return Helper<U, static_cast<DType>(i + 1)>::getType();
            }
        }
    };

    template <typename U>
    struct Helper<U, DType::kNumTypes> {
        constexpr static DType getType() {
            return DType::kNumTypes;
        }
    };

    template <typename U>
    constexpr static DType getType() {
        return Helper<U, DType::kByte>::getType();
    }

    constexpr static DType dtype = getType<T>();
    constexpr static size_t size = sizeof(T);
};

class Tensor {
 public:
  Tensor(const NVTEShape &shape, const DType type, const NVTEScalingMode &mode = {-1, -1, 1});

  Tensor(const std::vector<size_t> &shape, const DType type, const std::vector<int> &mode = {-1, -1, 1}) :
    Tensor(NVTEShape{shape.data(), shape.size()}, type, NVTEScalingMode{mode[0], mode[1], mode[2]}) {}

  Tensor() {}

  Tensor& operator=(const Tensor &other) = delete;
  Tensor(const Tensor &other) = delete;

  Tensor(Tensor &&other) = default;
  Tensor& operator=(Tensor &&other) = default;

  ~Tensor() {
    if (tensor_.dptr() != nullptr) {
      cudaFree(tensor_.dptr());
    }
  }
  NVTETensor data() const noexcept {
    return tensor_.data();
  }

  const NVTEShape shape() const noexcept {
    return tensor_.shape();
  }

  DType dtype() const noexcept {
    return tensor_.dtype();
  }

  void *dptr() const noexcept {
    return tensor_.dptr();
  }

  template <typename T>
  T *cpu_dptr() const {
    NVTE_CHECK(TypeInfo<T>::dtype == tensor_.dtype(), "Invalid type!");
    return reinterpret_cast<T *>(cpu_data_.get());
  }

  float amax() const {
    if(amax_cpu_data_) {
      to_cpu();
      return *amax_cpu_data_;
    } else {
      return 0;
    }
  }

  float scale() const {
    if(scale_cpu_data_) {
      NVTE_CHECK(tensor_.scaling_mode().x == -1 && tensor_.scaling_mode().y == -1, "Invalid scaling_mode!");
      to_cpu();
      return *scale_cpu_data_;
    } else {
      return 1;
    }
  }

  template <typename T>
  T *cpu_scale_inv_ptr(){
    if (tensor_.scaling_mode().x == -1 && tensor_.scaling_mode().y == -1){
      NVTE_CHECK(TypeInfo<T>::dtype == DType::kFloat32, "Invalid type!");
    } else {
      NVTE_CHECK(TypeInfo<T>::dtype == DType::kByte, "Invalid type!");
    }
    return reinterpret_cast<T*>(scale_inv_cpu_data_.get());
  }

  float scale_inv(){
    if(scale_inv_cpu_data_) {
      to_cpu();
      float scale_inv = cpu_scale_inv_ptr<float>()[0];
      return scale_inv;
    } else {
      return 1;
    }
  }

  void to_cpu() const;
  void from_cpu() const;
  void set_scale(float scale);
  void set_scale_inv(float scale_inv);
  void shareFP8Meta(const Tensor &other);

 private:
  TensorWrapper tensor_;
  std::unique_ptr<unsigned char[]> cpu_data_;
  std::shared_ptr<float> amax_cpu_data_;
  std::shared_ptr<float> scale_cpu_data_;
  std::unique_ptr<unsigned char[]> scale_inv_cpu_data_;
};

constexpr uint32_t FP32_EXPONENT_BIAS = 127;

template <typename T>
struct Numeric_Traits {
    static constexpr double minSubnorm = 1.0;
    static constexpr double maxSubnorm = 1.0;
    static constexpr double minNorm    = 1.0;
    static constexpr double maxNorm    = 1.0;
    static constexpr double artifInf   = 1.0;
    static constexpr int maxBiasedExponent = 1;
};

template <>
struct Numeric_Traits<fp8e4m3> {
    static constexpr double minSubnorm = 1.0   / static_cast<double>(1 << 9);   // std::pow(2.0, -9.0);
    static constexpr double maxSubnorm = 0.875 / static_cast<double>(1 << 6);   // std::pow(2.0, -6.0);
    static constexpr double minNorm    = 1.0   / static_cast<double>(1 << 6);   // std::pow(2.0, -6.0);
    static constexpr double maxNorm    = 448.0;
    static constexpr double artifInf   = 10.0 * maxNorm;                        // artificial Infinity
    static constexpr int maxBiasedExponentAsFP32 = 8 + FP32_EXPONENT_BIAS;
    static constexpr int maxUnbiasedExponentAsFP32 = 8;
};

template <>
struct Numeric_Traits<fp8e5m2> {
    static constexpr double minSubnorm = 1.0  / static_cast<double>(1 << 16);   // std::pow(2.0, -16.0);
    static constexpr double maxSubnorm = 0.75 / static_cast<double>(1 << 14);   // std::pow(2.0, -14.0);
    static constexpr double minNorm    = 1.0  / static_cast<double>(1 << 14);   // std::pow(2.0, -14.0);
    static constexpr double maxNorm    = 57344.0;
    static constexpr double artifInf   = 10.0 * maxNorm;                        // artificial Infinity
    static constexpr int maxBiasedExponentAsFP32 = 15 + FP32_EXPONENT_BIAS;
    static constexpr int maxUnbiasedExponentAsFP32 = 15;
};

template <>
struct Numeric_Traits<fp32> {
    static constexpr double minSubnorm = std::numeric_limits<fp32>::denorm_min();   // std::pow(2.0, -149.0);
    static constexpr double maxSubnorm = std::numeric_limits<fp32>::min()
                                         - std::numeric_limits<fp32>::denorm_min(); // minNormalized - minDenormalized
    static constexpr double minNorm    = std::numeric_limits<fp32>::min();          // std::pow(2.0, -126.0);
    static constexpr double maxNorm    = std::numeric_limits<fp32>::max();          // (1 - pow(2, -24)) * pow(2, 128)
    static constexpr double artifInf   = std::numeric_limits<fp32>::infinity();
    static constexpr int maxBiasedExponentAsFP32 = 255;
    static constexpr int maxUnbiasedExponentAsFP32 = 128;
};

template <typename T>
struct Quantized_Limits {
    static constexpr double ranges[]  = {
        0.0,
        Numeric_Traits<T>::minNorm,
        Numeric_Traits<T>::maxNorm,
        Numeric_Traits<T>::artifInf
    };
    static constexpr inline fp32 max() { return static_cast<fp32>(Numeric_Traits<T>::maxNorm); }
    static constexpr inline fp32 max_reciprocal() { return static_cast<fp32>(1.0 / max()); }
    static constexpr inline int max_norm_biased_exponent() { return Numeric_Traits<T>::maxBiasedExponentAsFP32; }
    static constexpr inline int max_norm_unbiased_exponent() { return Numeric_Traits<T>::maxUnbiasedExponentAsFP32; }
};

// Input data filling cases
// Considering normal and subnormal magnitudes of E4M3 and E5M2 formats
// with nearest to even rounding per OFP8 specification
enum InputsFillCase {
    zero_to_minNorm             = 0,    // [0, min_normal)
    minNorm_to_maxNorm          = 1,    // [min_normal, max_normal)
    maxNorm_to_inf              = 2,    // [max_normal, inf)
    zeros                       = 3,    // {0}
    uniform                     = 4,    // std::uniform_real_distribution<> dis(-2.0, 1.0)
};

size_t typeToSize(DType type);
size_t product(const NVTEShape &shape);

bool areShapesEqual(const NVTEShape &s1, const NVTEShape &s2);

void compareResults(const std::string &name, const Tensor &test, const void *ref,
                    double atol = 1e-5, double rtol = 1e-8);
void compareResults(const std::string &name, const float test, const float ref,
                    double atol = 1e-5, double rtol = 1e-8);
void compareResults(const std::string &name, const uint8_t *test, const uint8_t *ref,
                    size_t N);

std::pair<double, double> getTolerances(const DType type);

void fillUniform(Tensor *t);

template <typename InputEncoding>
void fillCase(Tensor *t, const InputsFillCase fill_case);

void setRandomScale(Tensor *t);
void setRandomScaleInv(Tensor *t);

constexpr int THREADS_PER_WARP = 32;

const std::string &typeName(DType type);
const std::string& caseName(InputsFillCase type);

extern std::vector<DType> all_fp_types;

bool isFp8Type(DType type);

}  // namespace test

#define TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(dtype, type, ...) \
    switch (dtype) { \
        using namespace transformer_engine; \
        case DType::kByte: \
            { \
                using type = byte; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kInt32: \
            { \
                using type = int32; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kInt64: \
            { \
                using type = int64; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat32: \
            { \
                using type = float; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat16: \
            { \
                using type = fp16; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kBFloat16: \
            { \
                using type = bf16; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat8E4M3: \
            { \
                using type = fp8e4m3; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat8E5M2: \
            { \
                using type = fp8e5m2; \
                {__VA_ARGS__} \
            } \
        break; \
        default: \
            NVTE_ERROR("Invalid type."); \
    }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(dtype, type, ...) \
    switch (dtype) { \
        using namespace transformer_engine; \
        case DType::kFloat8E4M3: \
            { \
                using type = fp8e4m3; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat8E5M2: \
            { \
                using type = fp8e5m2; \
                {__VA_ARGS__} \
            } \
        break; \
        default: \
            NVTE_ERROR("Invalid type."); \
    }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(dtype, type, ...) \
    switch (dtype) { \
        using namespace transformer_engine; \
        case DType::kFloat32: \
            { \
                using type = float; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat16: \
            { \
                using type = fp16; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kBFloat16: \
            { \
                using type = bf16; \
                {__VA_ARGS__} \
            } \
        break; \
        default: \
            NVTE_ERROR("Invalid type."); \
    }
