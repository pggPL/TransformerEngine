/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_

#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/python/interop.h>  // from_pyobject / to_pyobject (2.14+)
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/cast_transpose_noop.h>
#include <transformer_engine/comm_gemm_overlap.h>
#include <transformer_engine/fused_attn.h>
#include <transformer_engine/fused_rope.h>
#include <transformer_engine/fused_router.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/hadamard_transform.h>
#include <transformer_engine/multi_stream.h>
#include <transformer_engine/multi_tensor.h>
#include <transformer_engine/normalization.h>
#include <transformer_engine/padding.h>
#include <transformer_engine/permutation.h>
#include <transformer_engine/recipe.h>
#include <transformer_engine/softmax.h>
#include <transformer_engine/swizzle.h>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/transpose.h>
#include <transformer_engine/utils.h>

#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

// The RNG/Philox path uses torch::stable::philox_cuda_state_from_pyobject (from
// <torch/csrc/stable/python/interop.h>, included above) instead of ATen CUDA
// generator internals. torch.distributed ProcessGroup now uses the stable-ABI
// header-only wrapper (see dist_group_type below).
#include <torch/csrc/stable/c10d.h>

#include "common/util/logging.h"

namespace nb = nanobind;

namespace transformer_engine::pytorch {

// in python we have: dist_group_type = torch.distributed.ProcessGroup
// Stable-ABI header-only wrapper over c10d::ProcessGroup.
using dist_group_type = torch::stable::ProcessGroup;

// Each tensor here is shape (N, ) holding all scaling
// data for a single FP8 block, e.g. LayerNormLinear
class FP8TensorMeta {
 public:
  torch::stable::Tensor scale;
  torch::stable::Tensor scale_inv;
  torch::stable::Tensor amax_history;
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8FwdTensors {
  GEMM1_INPUT = 0,
  GEMM1_WEIGHT = 1,
  GEMM1_OUTPUT = 2,
  GEMM2_INPUT = 3,
  GEMM2_WEIGHT = 4,
  GEMM2_OUTPUT = 5,
  GEMM3_INPUT = 6,
  GEMM3_WEIGHT = 7,
  GEMM3_OUTPUT = 8
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8BwdTensors {
  GRAD_OUTPUT1 = 0,
  GRAD_INPUT1 = 1,
  GRAD_OUTPUT2 = 2,
  GRAD_INPUT2 = 3,
  GRAD_OUTPUT3 = 4,
  GRAD_INPUT3 = 5
};

class Quantizer {
 public:
  virtual NVTEScalingMode get_scaling_mode() const = 0;

  virtual void set_quantization_params(TensorWrapper* tensor) const = 0;

  /*! @brief Construct a tensor with uninitialized data */
  virtual std::pair<TensorWrapper, nb::object> create_tensor(
      const std::vector<size_t>& shape, DType dtype,
      std::optional<torch::stable::Device> device = std::nullopt,
      bool pin_memory = false) const = 0;

  /*! @brief Construct a grouped tensor with uninitialized data
   *
   * @param tensor_offsets If provided, the precomputed inclusive scan of
   *   ``first_dims * logical_last_dim`` with a leading zero, used to locate
   *   each per-group sub-tensor in the shared backing buffer. If null, the
   *   offsets are computed from ``first_dims`` on demand. Passing this in lets
   *   callers that already have the scan (e.g. from
   *   ``tex.splits_to_offsets_multi``) skip the redundant kernel launch.
   */
  virtual std::pair<GroupedTensorWrapper, nb::object> create_grouped_tensor(
      size_t num_tensors, const std::vector<size_t>& logical_shape, DType dtype,
      nb::object quantizer, const std::optional<torch::stable::Tensor>& first_dims,
      const std::optional<torch::stable::Tensor>& last_dims,
      const std::optional<torch::stable::Tensor>& precomputed_tensor_offsets,
      size_t logical_first_dim, size_t logical_last_dim) const = 0;

  /*! @brief Convert a PyTorch tensor into a Transformer Engine C++ tensor
   *
   * The PyTorch tensor's attributes are modified to match the
   * quantizer's configuration.
   */
  virtual std::pair<TensorWrapper, nb::object> convert_and_update_tensor(
      nb::object tensor) const = 0;

  /*! @brief Convert to a quantized data format */
  virtual void quantize(const TensorWrapper& input, TensorWrapper& out,
                        const std::optional<TensorWrapper>& noop_flag = std::nullopt) = 0;

  virtual ~Quantizer() = default;

  DType dtype = DType::kNumTypes;
  bool rowwise_usage = true;
  bool columnwise_usage = true;
  bool internal = false;
  bool optimize_for_gemm = false;
  nb::handle quantizer;

 protected:
  explicit Quantizer(const nb::handle& quantizer);
};

class NoneQuantizer : public Quantizer {
 public:
  explicit NoneQuantizer(const nb::handle& quantizer) : Quantizer(quantizer) {}

  NVTEScalingMode get_scaling_mode() const override { return NVTE_DELAYED_TENSOR_SCALING; }

  void set_quantization_params(TensorWrapper* tensor) const override {}

  std::pair<TensorWrapper, nb::object> create_tensor(
      const std::vector<size_t>& shape, DType dtype,
      std::optional<torch::stable::Device> device = std::nullopt,
      bool pin_memory = false) const override;

  std::pair<GroupedTensorWrapper, nb::object> create_grouped_tensor(
      size_t num_tensors, const std::vector<size_t>& logical_shape, DType dtype,
      nb::object quantizer, const std::optional<torch::stable::Tensor>& first_dims,
      const std::optional<torch::stable::Tensor>& last_dims,
      const std::optional<torch::stable::Tensor>& precomputed_tensor_offsets,
      size_t logical_first_dim, size_t logical_last_dim) const override;

  /*! @brief Construct a tensor with pre-initialized data */
  std::pair<TensorWrapper, nb::object> create_tensor(const std::vector<size_t>& shape, DType dtype,
                                                     torch::stable::Tensor data) const;

  std::pair<TensorWrapper, nb::object> convert_and_update_tensor(nb::object tensor) const override;

  void quantize(const TensorWrapper& input, TensorWrapper& out,
                const std::optional<TensorWrapper>& noop_flag = std::nullopt) override;
};

class Float8Quantizer : public Quantizer {
 public:
  torch::stable::Tensor scale;
  torch::stable::Tensor scale_inv;
  torch::stable::Tensor amax;

  explicit Float8Quantizer(const nb::handle& quantizer);

  NVTEScalingMode get_scaling_mode() const override { return NVTE_DELAYED_TENSOR_SCALING; }

  void set_quantization_params(TensorWrapper* tensor) const override;

  std::pair<TensorWrapper, nb::object> create_tensor(
      const std::vector<size_t>& shape, DType dtype,
      std::optional<torch::stable::Device> device = std::nullopt,
      bool pin_memory = false) const override;

  std::pair<GroupedTensorWrapper, nb::object> create_grouped_tensor(
      size_t num_tensors, const std::vector<size_t>& logical_shape, DType dtype,
      nb::object quantizer, const std::optional<torch::stable::Tensor>& first_dims,
      const std::optional<torch::stable::Tensor>& last_dims,
      const std::optional<torch::stable::Tensor>& precomputed_tensor_offsets,
      size_t logical_first_dim, size_t logical_last_dim) const override;

  /*! @brief Construct a tensor with pre-initialized data */
  std::pair<TensorWrapper, nb::object> create_tensor(
      const std::vector<size_t>& shape, DType dtype, std::optional<torch::stable::Tensor> data,
      std::optional<torch::stable::Tensor> transpose, std::optional<torch::stable::Tensor> scale_inv,
      std::optional<torch::stable::Device> device = std::nullopt, bool pin_memory = false) const;

  std::pair<TensorWrapper, nb::object> convert_and_update_tensor(nb::object shape) const override;

  void quantize(const TensorWrapper& input, TensorWrapper& out,
                const std::optional<TensorWrapper>& noop_flag = std::nullopt) override;
};

class Float8CurrentScalingQuantizer : public Quantizer {
 public:
  DType dtype;
  bool with_amax_reduction;
  std::optional<dist_group_type> amax_reduction_group;
  bool force_pow_2_scales = false;
  float amax_epsilon = 0.0;

  explicit Float8CurrentScalingQuantizer(const nb::handle& quantizer);

  NVTEScalingMode get_scaling_mode() const override { return NVTE_DELAYED_TENSOR_SCALING; }

  void set_quantization_params(TensorWrapper* tensor) const override;

  std::pair<TensorWrapper, nb::object> create_tensor(
      const std::vector<size_t>& shape, DType dtype,
      std::optional<torch::stable::Device> device = std::nullopt,
      bool pin_memory = false) const override;

  std::pair<GroupedTensorWrapper, nb::object> create_grouped_tensor(
      size_t num_tensors, const std::vector<size_t>& logical_shape, DType dtype,
      nb::object quantizer, const std::optional<torch::stable::Tensor>& first_dims,
      const std::optional<torch::stable::Tensor>& last_dims,
      const std::optional<torch::stable::Tensor>& precomputed_tensor_offsets,
      size_t logical_first_dim, size_t logical_last_dim) const override;

  /*! @brief Construct an unquantized tensor with a freshly allocated amax buffer.
   *
   * The amax is zeroed out. Most TE kernels that output amax expect
   * amax to be initialized to zero. The amax tensor is returned as
   * the third element to keep it alive in the caller's scope.
  */
  std::tuple<TensorWrapper, nb::object, torch::stable::Tensor> create_unquantized_tensor_with_amax(
      const std::vector<size_t>& shape, DType dtype, std::optional<torch::stable::Tensor> data = std::nullopt);

  std::pair<TensorWrapper, nb::object> convert_and_update_tensor(nb::object shape) const override;

  void quantize(const TensorWrapper& input, TensorWrapper& out,
                const std::optional<TensorWrapper>& noop_flag = std::nullopt) override;

  /*! @brief Quantize to FP8, skipping local amax computation
   *
   * The provided amax tensor is assumed to already hold the local
   * amax. The amax may still be reduced across the amax reduction
   * group.
   */
  void quantize_with_amax(TensorWrapper& input, TensorWrapper& out, torch::stable::Tensor amax,
                          const std::optional<TensorWrapper>& noop_flag = std::nullopt);

 private:
  void quantize_impl(const TensorWrapper& input, TensorWrapper& out,
                     const std::optional<TensorWrapper>& noop_flag, bool compute_amax,
                     torch::stable::Tensor amax_buf, torch::stable::Tensor scale_buf);
};

class Float8BlockQuantizer : public Quantizer {
 public:
  // Options about how to quantize the tensor
  // Quantization scales are rounded down to powers of 2.
  bool force_pow_2_scales = false;
  // Amax within quantization tile has a floor of epsilon.
  float amax_epsilon = 0.0;

 private:
  int block_scaling_dim = 2;

 public:
  // Initializes from a python handle to a Float8BlockQuantizer
  explicit Float8BlockQuantizer(const nb::handle& quantizer);

  NVTEScalingMode get_scaling_mode() const override {
    return (block_scaling_dim == 2) ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D;
  }

  // Gets rowwise and columnwise_data from tensor and sets them on wrapper
  void set_quantization_params(TensorWrapper* tensor) const override;

  // Create a python Float8BlockQuantized tensor and C++ wrapper
  // for the tensor. Should set quantized data, scales for rowwise
  // and optionally columnwise usage.
  std::pair<TensorWrapper, nb::object> create_tensor(
      const std::vector<size_t>& shape, DType dtype,
      std::optional<torch::stable::Device> device = std::nullopt,
      bool pin_memory = false) const override;

  std::pair<GroupedTensorWrapper, nb::object> create_grouped_tensor(
      size_t num_tensors, const std::vector<size_t>& logical_shape, DType dtype,
      nb::object quantizer, const std::optional<torch::stable::Tensor>& first_dims,
      const std::optional<torch::stable::Tensor>& last_dims,
      const std::optional<torch::stable::Tensor>& precomputed_tensor_offsets,
      size_t logical_first_dim, size_t logical_last_dim) const override;

  std::pair<TensorWrapper, nb::object> convert_and_update_tensor(nb::object shape) const override;

  void quantize(const TensorWrapper& input, TensorWrapper& out,
                const std::optional<TensorWrapper>& noop_flag = std::nullopt) override;

  std::vector<size_t> get_scale_shape(const std::vector<size_t>& shape, bool columnwise) const;
};

class MXFP8Quantizer : public Quantizer {
 public:
  explicit MXFP8Quantizer(const nb::handle& quantizer);

  NVTEScalingMode get_scaling_mode() const override { return NVTE_MXFP8_1D_SCALING; }

  void set_quantization_params(TensorWrapper* tensor) const override;

  std::pair<TensorWrapper, nb::object> create_tensor(
      const std::vector<size_t>& shape, DType dtype,
      std::optional<torch::stable::Device> device = std::nullopt,
      bool pin_memory = false) const override;

  std::pair<GroupedTensorWrapper, nb::object> create_grouped_tensor(
      size_t num_tensors, const std::vector<size_t>& logical_shape, DType dtype,
      nb::object quantizer, const std::optional<torch::stable::Tensor>& first_dims,
      const std::optional<torch::stable::Tensor>& last_dims,
      const std::optional<torch::stable::Tensor>& precomputed_tensor_offsets,
      size_t logical_first_dim, size_t logical_last_dim) const override;

  std::pair<TensorWrapper, nb::object> convert_and_update_tensor(nb::object shape) const override;

  void quantize(const TensorWrapper& input, TensorWrapper& out,
                const std::optional<TensorWrapper>& noop_flag = std::nullopt) override;

  std::vector<size_t> get_scale_shape(const std::vector<size_t>& shape, bool columnwise) const;
};

class NVFP4Quantizer : public Quantizer {
 public:
  // amax reduction for low precision FP4 AG
  bool with_amax_reduction;
  std::optional<dist_group_type> amax_reduction_group;
  // random hadamard transform
  bool with_rht;
  bool with_post_rht_amax;
  // 2D block scaling
  bool with_2d_quantization;
  bool stochastic_rounding;
  // 4over6 candidate-selection mode used when quantizing emitted NVFP4 tensors.
  NVTENVFP44Over6Mode nvfp4_4over6_mode;
  // Global E4M3 scale bound used by emitted NVFP4 tensors.
  int nvfp4_e4m3_max;
  // Whether tensors emitted by this quantizer use row-scaled NVFP4 metadata.
  bool row_scaled_nvfp4;

  int rht_matrix_random_sign_mask_t;
  torch::stable::Tensor rht_matrix;

  explicit NVFP4Quantizer(const nb::handle& quantizer);

  NVTEScalingMode get_scaling_mode() const override { return NVTE_NVFP4_1D_SCALING; }

  void set_quantization_params(TensorWrapper* tensor) const override;

  std::pair<TensorWrapper, nb::object> create_tensor(
      const std::vector<size_t>& shape, DType dtype,
      std::optional<torch::stable::Device> device = std::nullopt,
      bool pin_memory = false) const override;

  std::pair<GroupedTensorWrapper, nb::object> create_grouped_tensor(
      size_t num_tensors, const std::vector<size_t>& logical_shape, DType dtype,
      nb::object quantizer, const std::optional<torch::stable::Tensor>& first_dims,
      const std::optional<torch::stable::Tensor>& last_dims,
      const std::optional<torch::stable::Tensor>& precomputed_tensor_offsets,
      size_t logical_first_dim, size_t logical_last_dim) const override;

  /*! @brief Construct an unquantized tensor that shares NVFP4 tensor's amax pointer
   *
   * The amax is zeroed out. Most TE kernels that output amax expect
   * amax to be initialized to zero.
   */
  std::pair<TensorWrapper, nb::object> create_unquantized_tensor_with_amax(
      TensorWrapper& quantized_tensor, DType dtype);

  std::pair<TensorWrapper, nb::object> convert_and_update_tensor(nb::object shape) const override;

  void quantize(const TensorWrapper& input, TensorWrapper& out,
                const std::optional<TensorWrapper>& noop_flag = std::nullopt) override;
  void quantize_impl(const TensorWrapper& input, TensorWrapper& out,
                     const std::optional<TensorWrapper>& noop_flag, bool compute_amax);

  /*! @brief Quantize to NVFP4, skipping local amax computation
   *
   * The input tensor's amax pointer is assumed to already hold the
   * local amax. The amax may still be reduced across the amax
   * reduction group.
   */
  void quantize_with_amax(TensorWrapper& input, TensorWrapper& out);

  std::vector<size_t> get_scale_shape(const std::vector<size_t>& shape, bool columnwise) const;

  /*! @brief Whether a tensor of the given shape is eligible for
   *  the NVFP4 RHT cast-fusion kernel (single-tensor or grouped).
   */
  static bool is_eligible_for_rht_cast_fusion(const std::vector<size_t>& shape,
                                              bool for_grouped_kernel = false);

 private:
  void quantize_with_rht_unfused_helper(const TensorWrapper& input, TensorWrapper& out,
                                        TensorWrapper& rht_output_t_cpp,
                                        QuantizationConfigWrapper& quant_config,
                                        QuantizationConfigWrapper& quant_config_columnwise,
                                        cudaStream_t stream);
};

std::unique_ptr<Quantizer> convert_quantizer(nb::handle quantizer);

std::vector<size_t> getTensorShape(const torch::stable::Tensor& t);

transformer_engine::DType getTransformerEngineFP8Type(bool e4m3_if_hybrid,
                                                      const std::string& fp8_recipe);

/*! @brief Wrap a C++ ``transformer_engine::DType`` as the canonical Python
 *         ``transformer_engine.pytorch.DType`` ``IntEnum`` member.
 *
 * The returned object is cached per enum value (one ``nb::object`` per
 * ``DType``), nb::object corresponds to the python ``DType`` enum member
 * defined in transformer_engine.pytorch.
 */
nb::object MakePythonDType(transformer_engine::DType dtype);

inline size_t typeToNumBits(transformer_engine::DType t) {
  switch (t) {
    case transformer_engine::DType::kInt64:
      return 64;
    case transformer_engine::DType::kInt32:
    case transformer_engine::DType::kFloat32:
      return 32;
    case transformer_engine::DType::kInt16:
    case transformer_engine::DType::kFloat16:
    case transformer_engine::DType::kBFloat16:
      return 16;
    case transformer_engine::DType::kByte:
    case transformer_engine::DType::kFloat8E4M3:
    case transformer_engine::DType::kFloat8E5M2:
    case transformer_engine::DType::kFloat8E8M0:
      return 8;
    case transformer_engine::DType::kFloat4E2M1:
      return 4;
    default:
      NVTE_ERROR("Invalid type (", static_cast<int>(t), ").");
  }
}

inline torch::headeronly::ScalarType GetATenDType(transformer_engine::DType t) {
  using torch::headeronly::ScalarType;
  switch (t) {
    case transformer_engine::DType::kInt16:
      return ScalarType::Short;
    case transformer_engine::DType::kInt32:
      return ScalarType::Int;
    case transformer_engine::DType::kInt64:
      return ScalarType::Long;
    case transformer_engine::DType::kFloat32:
      return ScalarType::Float;
    case transformer_engine::DType::kFloat16:
      return ScalarType::Half;
    case transformer_engine::DType::kBFloat16:
      return ScalarType::BFloat16;
    case transformer_engine::DType::kByte:
      return ScalarType::Byte;
    case transformer_engine::DType::kFloat8E4M3:
      return ScalarType::Float8_e4m3fn;
    case transformer_engine::DType::kFloat8E5M2:
      return ScalarType::Float8_e5m2;
    case transformer_engine::DType::kFloat8E8M0:
      return ScalarType::Byte;  // e8m0 dtype requires PyTorch 2.7.0+
    default:
      NVTE_ERROR("Invalid type (", static_cast<int>(t), ").");
  }
}

inline transformer_engine::DType GetTransformerEngineDType(torch::headeronly::ScalarType t) {
  using torch::headeronly::ScalarType;
  switch (t) {
    case ScalarType::Float8_e4m3fn:
      return transformer_engine::DType::kFloat8E4M3;
    case ScalarType::Float8_e5m2:
      return transformer_engine::DType::kFloat8E5M2;
    case ScalarType::Half:
      return transformer_engine::DType::kFloat16;
    case ScalarType::Float:
      return transformer_engine::DType::kFloat32;
    case ScalarType::BFloat16:
      return transformer_engine::DType::kBFloat16;
    case ScalarType::Bool:
      return transformer_engine::DType::kByte;
    case ScalarType::Byte:
      return transformer_engine::DType::kByte;
    case ScalarType::Short:
      return transformer_engine::DType::kInt16;
    case ScalarType::Int:
      return transformer_engine::DType::kInt32;
    case ScalarType::Long:
      return transformer_engine::DType::kInt64;
    default:
      NVTE_ERROR("Invalid type (", static_cast<int>(t), ").");
  }
}

inline transformer_engine::DType GetTransformerEngineDType(int DType_value) {
  return static_cast<transformer_engine::DType>(DType_value);
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(void* data_ptr,
                                                              const std::vector<size_t>& shape,
                                                              const transformer_engine::DType type);

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, const std::vector<size_t>& shape, const transformer_engine::DType type,
    void* amax_ptr, void* scale_ptr, void* scale_inv_ptr, std::vector<size_t> scale_inv_shape = {1},
    NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING);

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, void* columnwise_data_ptr, const std::vector<size_t>& shape,
    const std::vector<size_t>& columnwise_shape, const transformer_engine::DType type,
    void* amax_ptr, void* scale_ptr, void* scale_inv_ptr, void* columnwise_scale_inv_ptr,
    const std::vector<size_t>& scale_inv_shape = {1},
    const std::vector<size_t>& columnwise_scale_inv_shape = {1},
    NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING);

transformer_engine::TensorWrapper makeTransformerEngineTensor(void* data_ptr,
                                                              const NVTEShape& shape,
                                                              const transformer_engine::DType type);

transformer_engine::TensorWrapper makeTransformerEngineTensor(torch::stable::Tensor tensor);

std::tuple<std::vector<transformer_engine::TensorWrapper>, std::vector<std::vector<NVTETensor>>,
           std::vector<NVTETensor*>, size_t, size_t>
makeTransformerEngineTensorList(std::vector<std::vector<torch::stable::Tensor>> at_tensor_lists);

TensorWrapper makeTransformerEngineTensor(nb::handle tensor, nb::handle quantizer);

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    torch::stable::Tensor tensor, torch::stable::Tensor amax, const torch::stable::Tensor scale,
    torch::stable::Tensor scale_inv, NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING);

template <typename T>
T product(const std::vector<T>& shape);

size_t product(const NVTEShape& shape, size_t begin, size_t end);

std::vector<size_t> nvte_shape_to_vector(const NVTEShape& nvte_shape);

torch::stable::Tensor allocateSpace(const std::vector<size_t>& shape, const transformer_engine::DType type,
                         bool init_to_zeros);

torch::stable::Tensor allocateSpace(const NVTEShape& shape, const transformer_engine::DType type,
                         bool init_to_zeros = false);

torch::stable::Tensor allocateTorchTensor(int M, int N, transformer_engine::DType dtype);

torch::stable::Tensor allocateTorchTensor(int M, transformer_engine::DType dtype);

void* getDataPtr(torch::stable::Tensor tensor, int offset = 0);

std::vector<size_t> convertShape(const NVTEShape& shape);

size_t roundup(size_t value, size_t multiple);

size_t ceildiv(size_t numer, size_t denom);

NVTEShape convertTorchShape(const torch::headeronly::IntHeaderOnlyArrayRef torch_shape);

/*! @brief Current CUDA stream for the active accelerator device.
 *
 * Replaces at::cuda::getCurrentCUDAStream(). Uses the stable accelerator API and
 * its native-handle accessor to recover the raw ``cudaStream_t``.
 */
cudaStream_t getCurrentCUDAStream();

std::vector<size_t> convert_shape_back_from_fp4(const std::vector<size_t>& shape, bool transpose);

// Flatten an N-D shape to 2D: {product(shape[:-1]), shape[-1]}.
// With transpose=true: {shape[0], product(shape[1:])}.
std::array<size_t, 2> get_2d_dims(NVTEShape shape, bool transpose = false);

template <typename T>
inline std::array<size_t, 2> get_2d_dims(const std::vector<T>& shape, bool transpose = false) {
  NVTEShape s{};
  s.ndim = shape.size();
  constexpr size_t max_ndim = sizeof(s.data) / sizeof(size_t);
  NVTE_CHECK(s.ndim <= max_ndim, "Shape has too many dimensions (got ", s.ndim, ", max ", max_ndim,
             ").");
  for (size_t i = 0; i < shape.size(); ++i) s.data[i] = static_cast<size_t>(shape[i]);
  return get_2d_dims(s, transpose);
}

// Resolve the CUDA generator to use: returns `gen` if provided, otherwise the
// default CUDA generator for the current device (torch.cuda.default_generators
// [current_device]), matching at::cuda::detail::getDefaultCUDAGenerator().
// Must be called while holding the GIL.
nb::object get_cuda_generator(const std::optional<nb::object>& gen);

// extract the Philox RNG state from a CUDA torch.Generator (Python object),
// advancing its offset by `elts_per_thread`. Must be called while holding the GIL.
torch::stable::PhiloxCudaState init_philox_state(const nb::object& gen, size_t elts_per_thread);

// unpack the PhiloxCudaState into a size-2 CUDA int64 tensor (seed, offset)
void philox_unpack(const torch::stable::PhiloxCudaState& arg, int64_t* rng_state_ptr);

}  // namespace transformer_engine::pytorch

namespace std {
template <typename T>
string to_string(const vector<T>& vec) {
  string ret = "[";
  for (const auto& val : vec) {
    ret += to_string(val) + ",";
  }
  if (ret.size() > 1) {
    ret[ret.size() - 1] = ']';
  } else {
    ret += "]";
  }
  return ret;
}

// Torch shape -> string
template <typename T>
string to_string(const torch::headeronly::HeaderOnlyArrayRef<T>& vec) {
  string ret = "[";
  for (const auto& val : vec) {
    ret += to_string(val) + ",";
  }
  if (ret.size() > 1) {
    ret[ret.size() - 1] = ']';
  } else {
    ret += "]";
  }
  return ret;
}

inline string to_string(const NVTEShape& s) {
  string ret = "[";
  for (size_t i = 0; i < s.ndim; ++i) {
    ret += to_string(s.data[i]) + ",";
  }
  if (ret.size() > 1) {
    ret[ret.size() - 1] = ']';
  } else {
    ret += "]";
  }
  return ret;
}
}  // namespace std

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_
