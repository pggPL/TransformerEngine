/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/fused_router.h>

#include "../extensions.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

// ============================================================================
// Fused Top-K with Score Function - Forward
// ============================================================================

Error_Type FusedTopkWithScoreFunctionForwardFFI(
    cudaStream_t stream,
    Buffer_Type logits_buf,        // [num_tokens, num_experts]
    Buffer_Type expert_bias_buf,   // [num_experts] or empty
    Result_Type probs_buf,         // [num_tokens, num_experts] (or scores when compute_aux_scores)
    Result_Type routing_map_buf,   // BYTEMAP: [T, E] uint8 / BITMAP_U8: [T, ceil(E/8)] uint8
    Result_Type intermediate_buf,  // [num_tokens, num_experts]
    int64_t topk, int64_t use_pre_softmax, int64_t num_groups, int64_t group_topk,
    double scaling_factor, JAXX_Score_Function score_function, int64_t compute_aux_scores,
    JAXX_Routing_Map_Format routing_map_format) {
  auto dtype = convert_ffi_datatype_to_te_dtype(logits_buf.element_type());
  auto dims = logits_buf.dimensions();
  auto num_tokens = static_cast<int>(product(dims, 0, dims.size() - 1));
  auto num_experts = static_cast<int>(dims[dims.size() - 1]);

  auto *logits = logits_buf.untyped_data();
  auto *expert_bias = expert_bias_buf.untyped_data();
  auto *probs = probs_buf->untyped_data();
  auto *routing_map = routing_map_buf->untyped_data();
  auto *intermediate = intermediate_buf->untyped_data();

  auto flat_shape =
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)};
  auto logits_tensor = TensorWrapper(logits, flat_shape, dtype);
  auto probs_tensor = TensorWrapper(probs, flat_shape, dtype);
  // 2D shape for the kernel: trailing dim = num_experts for BYTEMAP,
  // ceil(num_experts/8) for BITMAP_U8.
  auto routing_map_format_nvte = static_cast<NVTERoutingMapFormat>(routing_map_format);
  size_t routing_map_trailing = (routing_map_format_nvte == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8)
                                    ? static_cast<size_t>((num_experts + 7) / 8)
                                    : static_cast<size_t>(num_experts);
  auto routing_map_shape =
      std::vector<size_t>{static_cast<size_t>(num_tokens), routing_map_trailing};
  auto routing_map_tensor = TensorWrapper(routing_map, routing_map_shape, DType::kByte);
  // intermediate is always float32 (CompType) regardless of logits dtype.
  auto intermediate_dtype = convert_ffi_datatype_to_te_dtype(intermediate_buf->element_type());
  NVTE_CHECK(
      intermediate_dtype == DType::kFloat32,
      "intermediate_output must be float32 (CompType); got dtype ",
      static_cast<int>(intermediate_dtype),
      ". Check FusedTopkWithScoreFunctionFwdPrimitive.abstract in cpp_extensions/router.py.");
  auto intermediate_tensor = TensorWrapper(intermediate, flat_shape, DType::kFloat32);

  if (compute_aux_scores) {
    nvte_fused_score_for_moe_aux_loss_forward_v2(
        logits_tensor.data(), num_tokens, num_experts, static_cast<int>(topk),
        static_cast<int>(score_function), probs_tensor.data(), routing_map_tensor.data(),
        routing_map_format_nvte, intermediate_tensor.data(), stream);
  } else {
    auto bias_dims = expert_bias_buf.dimensions();
    auto expert_bias_tensor =
        (bias_dims.size() > 0 && bias_dims[0] > 0)
            ? TensorWrapper(expert_bias, std::vector<size_t>{static_cast<size_t>(bias_dims[0])},
                            convert_ffi_datatype_to_te_dtype(expert_bias_buf.element_type()))
            : TensorWrapper();

    nvte_fused_topk_with_score_function_forward_v2(
        logits_tensor.data(), num_tokens, num_experts, static_cast<int>(topk),
        static_cast<int>(use_pre_softmax), static_cast<int>(num_groups),
        static_cast<int>(group_topk), static_cast<float>(scaling_factor),
        static_cast<int>(score_function), expert_bias_tensor.data(), probs_tensor.data(),
        routing_map_tensor.data(), routing_map_format_nvte, intermediate_tensor.data(), stream);
  }

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedTopkWithScoreFunctionForwardHandler,
                              FusedTopkWithScoreFunctionForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // logits
                                  .Arg<Buffer_Type>()      // expert_bias
                                  .Ret<Buffer_Type>()      // probs (or scores)
                                  .Ret<Buffer_Type>()      // routing_map
                                  .Ret<Buffer_Type>()      // intermediate_output
                                  .Attr<int64_t>("topk")
                                  .Attr<int64_t>("use_pre_softmax")
                                  .Attr<int64_t>("num_groups")
                                  .Attr<int64_t>("group_topk")
                                  .Attr<double>("scaling_factor")
                                  .Attr<JAXX_Score_Function>("score_function")
                                  .Attr<int64_t>("compute_aux_scores")
                                  .Attr<JAXX_Routing_Map_Format>("routing_map_format"),
                              FFI_CudaGraph_Traits);

Error_Type FusedTopkWithScoreFunctionQBForwardFFI(
    cudaStream_t stream, Buffer_Type logits_buf, Buffer_Type expert_bias_buf,
    Buffer_Type bin_bounds_buf, Buffer_Type histogram_buf, Result_Type probs_buf,
    Result_Type routing_map_buf, Result_Type intermediate_buf, Result_Type cutoff_buf,
    Result_Type histogram_out_buf, int64_t topk, double scaling_factor, int64_t histogram_mode,
    JAXX_Routing_Map_Format routing_map_format) {
  auto dims = logits_buf.dimensions();
  auto bias_dims = expert_bias_buf.dimensions();
  auto bounds_dims = bin_bounds_buf.dimensions();
  auto hist_dims = histogram_buf.dimensions();
  NVTE_CHECK(dims.size() == 2 && dims[0] > 0 && dims[1] > 0,
             "QB logits must have shape [tokens, experts] with positive dimensions.");
  const auto num_tokens = static_cast<int>(dims[0]);
  const auto num_experts = static_cast<int>(dims[1]);
  NVTE_CHECK(topk > 0 && topk < num_experts, "QB requires 0 < topk < num_experts.");
  NVTE_CHECK(bias_dims.size() == 1 && bias_dims[0] == num_experts,
             "QB expert_bias must have shape [experts].");
  NVTE_CHECK(bounds_dims.size() == 1 && bounds_dims[0] == 2, "QB bin_bounds must have shape [2].");
  NVTE_CHECK(hist_dims.size() == 2 && hist_dims[0] == num_experts && hist_dims[1] > 0,
             "QB histogram must have shape [experts, bins].");
  NVTE_CHECK(
      convert_ffi_datatype_to_te_dtype(logits_buf.element_type()) == DType::kFloat32 &&
          convert_ffi_datatype_to_te_dtype(expert_bias_buf.element_type()) == DType::kFloat32 &&
          convert_ffi_datatype_to_te_dtype(bin_bounds_buf.element_type()) == DType::kFloat32,
      "QB logits, expert_bias and bin_bounds must be float32.");
  NVTE_CHECK(convert_ffi_datatype_to_te_dtype(histogram_buf.element_type()) == DType::kInt32,
             "QB histogram must be int32.");
  NVTE_CHECK(histogram_buf.untyped_data() == histogram_out_buf->untyped_data(),
             "QB histogram input and output must alias.");
  NVTE_CHECK(histogram_mode == NVTE_QB_HISTOGRAM_TWO_KERNEL ||
                 histogram_mode == NVTE_QB_HISTOGRAM_FUSED_ATOMIC,
             "Invalid QB histogram mode.");

  const std::vector<size_t> shape{static_cast<size_t>(num_tokens),
                                  static_cast<size_t>(num_experts)};
  const auto format = static_cast<NVTERoutingMapFormat>(routing_map_format);
  const size_t map_width = format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8
                               ? static_cast<size_t>((num_experts + 7) / 8)
                               : static_cast<size_t>(num_experts);
  auto logits = TensorWrapper(logits_buf.untyped_data(), shape, DType::kFloat32);
  auto bias = TensorWrapper(expert_bias_buf.untyped_data(),
                            std::vector<size_t>{static_cast<size_t>(num_experts)}, DType::kFloat32);
  auto bounds =
      TensorWrapper(bin_bounds_buf.untyped_data(), std::vector<size_t>{2}, DType::kFloat32);
  auto probs = TensorWrapper(probs_buf->untyped_data(), shape, DType::kFloat32);
  auto routing_map =
      TensorWrapper(routing_map_buf->untyped_data(),
                    std::vector<size_t>{static_cast<size_t>(num_tokens), map_width}, DType::kByte);
  auto intermediate = TensorWrapper(intermediate_buf->untyped_data(), shape, DType::kFloat32);
  auto cutoff =
      TensorWrapper(cutoff_buf->untyped_data(),
                    std::vector<size_t>{static_cast<size_t>(num_tokens)}, DType::kFloat32);
  auto histogram = TensorWrapper(
      histogram_out_buf->untyped_data(),
      std::vector<size_t>{static_cast<size_t>(num_experts), static_cast<size_t>(hist_dims[1])},
      DType::kInt32);
  const auto mode = static_cast<NVTEQBHistogramMode>(histogram_mode);
  nvte_fused_topk_with_score_function_forward_qb_v2(
      logits.data(), num_tokens, num_experts, static_cast<int>(topk),
      static_cast<float>(scaling_factor), bias.data(), probs.data(), routing_map.data(), format,
      intermediate.data(), cutoff.data(), histogram.data(), bounds.data(), mode, stream);
  if (mode == NVTE_QB_HISTOGRAM_TWO_KERNEL) {
    nvte_qb_histogram_accumulate(intermediate.data(), cutoff.data(), bounds.data(),
                                 histogram.data(), stream);
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedTopkWithScoreFunctionQBForwardHandler,
                              FusedTopkWithScoreFunctionQBForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()
                                  .Arg<Buffer_Type>()  // logits
                                  .Arg<Buffer_Type>()  // expert_bias
                                  .Arg<Buffer_Type>()  // bin_bounds
                                  .Arg<Buffer_Type>()  // histogram
                                  .Ret<Buffer_Type>()  // probs
                                  .Ret<Buffer_Type>()  // routing_map
                                  .Ret<Buffer_Type>()  // intermediate
                                  .Ret<Buffer_Type>()  // cutoff
                                  .Ret<Buffer_Type>()  // updated histogram
                                  .Attr<int64_t>("topk")
                                  .Attr<double>("scaling_factor")
                                  .Attr<int64_t>("histogram_mode")
                                  .Attr<JAXX_Routing_Map_Format>("routing_map_format"),
                              FFI_CudaGraph_Traits);

// ============================================================================
// Fused Top-K with Score Function - Backward
// ============================================================================

Error_Type FusedTopkWithScoreFunctionBackwardFFI(
    cudaStream_t stream,
    Buffer_Type routing_map_buf,   // bytemap [T,E] uint8 or bitmap [T, ceil(E/8)] uint8
    Buffer_Type intermediate_buf,  // [num_tokens, num_experts]
    Buffer_Type grad_probs_buf,   // [num_tokens, num_experts] (grad_scores when compute_aux_scores)
    Result_Type grad_logits_buf,  // [num_tokens, num_experts]
    int64_t topk, int64_t use_pre_softmax, double scaling_factor,
    JAXX_Score_Function score_function, int64_t compute_aux_scores,
    JAXX_Routing_Map_Format routing_map_format) {
  // intermediate is always float32 (CompType) regardless of logits dtype.
  auto intermediate_dtype = convert_ffi_datatype_to_te_dtype(intermediate_buf.element_type());
  NVTE_CHECK(
      intermediate_dtype == DType::kFloat32,
      "intermediate_output must be float32 (CompType); got dtype ",
      static_cast<int>(intermediate_dtype),
      ". Check FusedTopkWithScoreFunctionFwdPrimitive.abstract in cpp_extensions/router.py.");
  auto grad_dtype = convert_ffi_datatype_to_te_dtype(grad_probs_buf.element_type());
  auto dims = intermediate_buf.dimensions();
  auto num_tokens = static_cast<int>(product(dims, 0, dims.size() - 1));
  auto num_experts = static_cast<int>(dims[dims.size() - 1]);

  auto flat_shape =
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)};

  auto intermediate_tensor =
      TensorWrapper(intermediate_buf.untyped_data(), flat_shape, DType::kFloat32);
  auto grad_probs_tensor = TensorWrapper(grad_probs_buf.untyped_data(), flat_shape, grad_dtype);
  auto grad_logits_tensor = TensorWrapper(grad_logits_buf->untyped_data(), flat_shape, grad_dtype);

  if (compute_aux_scores) {
    nvte_fused_score_for_moe_aux_loss_backward(intermediate_tensor.data(), grad_probs_tensor.data(),
                                               num_tokens, num_experts, static_cast<int>(topk),
                                               static_cast<int>(score_function),
                                               grad_logits_tensor.data(), stream);
  } else {
    auto routing_map_format_nvte = static_cast<NVTERoutingMapFormat>(routing_map_format);
    size_t routing_map_trailing = (routing_map_format_nvte == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8)
                                      ? static_cast<size_t>((num_experts + 7) / 8)
                                      : static_cast<size_t>(num_experts);
    auto routing_map_shape =
        std::vector<size_t>{static_cast<size_t>(num_tokens), routing_map_trailing};
    auto routing_map_tensor =
        TensorWrapper(routing_map_buf.untyped_data(), routing_map_shape, DType::kByte);

    nvte_fused_topk_with_score_function_backward_v2(
        routing_map_tensor.data(), routing_map_format_nvte, intermediate_tensor.data(),
        grad_probs_tensor.data(), num_tokens, num_experts, static_cast<int>(topk),
        static_cast<int>(use_pre_softmax), static_cast<float>(scaling_factor),
        static_cast<int>(score_function), grad_logits_tensor.data(), stream);
  }

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedTopkWithScoreFunctionBackwardHandler,
                              FusedTopkWithScoreFunctionBackwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // routing_map
                                  .Arg<Buffer_Type>()      // intermediate_output
                                  .Arg<Buffer_Type>()      // grad_probs
                                  .Ret<Buffer_Type>()      // grad_logits
                                  .Attr<int64_t>("topk")
                                  .Attr<int64_t>("use_pre_softmax")
                                  .Attr<double>("scaling_factor")
                                  .Attr<JAXX_Score_Function>("score_function")
                                  .Attr<int64_t>("compute_aux_scores")
                                  .Attr<JAXX_Routing_Map_Format>("routing_map_format"),
                              FFI_CudaGraph_Traits);

// ============================================================================
// Fused MoE Aux Loss - Forward
// ============================================================================

Error_Type FusedMoEAuxLossForwardFFI(cudaStream_t stream,
                                     Buffer_Type probs_buf,  // [num_tokens, num_experts]
                                     Buffer_Type tokens_per_expert_buf,  // [num_experts]
                                     Result_Type aux_loss_buf,           // scalar
                                     Result_Type const_buf,              // scalar
                                     int64_t topk, double coeff) {
  auto dtype = convert_ffi_datatype_to_te_dtype(probs_buf.element_type());
  auto probs_dims = probs_buf.dimensions();
  auto num_tokens = static_cast<int>(probs_dims[0]);
  auto num_experts = static_cast<int>(probs_dims[1]);

  auto probs_shape =
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)};
  auto tpe_dtype = convert_ffi_datatype_to_te_dtype(tokens_per_expert_buf.element_type());
  auto tpe_shape = std::vector<size_t>{static_cast<size_t>(num_experts)};

  auto probs_tensor = TensorWrapper(probs_buf.untyped_data(), probs_shape, dtype);
  auto tpe_tensor = TensorWrapper(tokens_per_expert_buf.untyped_data(), tpe_shape, tpe_dtype);
  auto aux_loss_tensor = TensorWrapper(aux_loss_buf->untyped_data(), std::vector<size_t>{1}, dtype);
  auto const_buf_tensor =
      TensorWrapper(const_buf->untyped_data(), std::vector<size_t>{2}, DType::kFloat32);

  nvte_fused_moe_aux_loss_forward(probs_tensor.data(), tpe_tensor.data(), num_tokens, num_experts,
                                  num_tokens, num_experts, static_cast<int>(topk),
                                  static_cast<float>(coeff), aux_loss_tensor.data(),
                                  const_buf_tensor.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedMoEAuxLossForwardHandler, FusedMoEAuxLossForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // probs
                                  .Arg<Buffer_Type>()      // tokens_per_expert
                                  .Ret<Buffer_Type>()      // aux_loss
                                  .Ret<Buffer_Type>()      // const_buf
                                  .Attr<int64_t>("topk")
                                  .Attr<double>("coeff"),
                              FFI_CudaGraph_Traits);

// ============================================================================
// Fused MoE Aux Loss - Backward
// ============================================================================

Error_Type FusedMoEAuxLossBackwardFFI(cudaStream_t stream,
                                      Buffer_Type const_buf_in,           // scalar float32
                                      Buffer_Type tokens_per_expert_buf,  // [num_experts]
                                      Buffer_Type grad_aux_loss_buf,      // scalar
                                      Result_Type grad_probs_buf) {  // [num_tokens, num_experts]
  auto grad_dtype = convert_ffi_datatype_to_te_dtype(grad_aux_loss_buf.element_type());
  auto tpe_dtype = convert_ffi_datatype_to_te_dtype(tokens_per_expert_buf.element_type());

  auto grad_probs_dims = grad_probs_buf->dimensions();
  auto num_tokens = static_cast<int>(grad_probs_dims[0]);
  auto num_experts = static_cast<int>(grad_probs_dims[1]);

  auto tpe_dims = tokens_per_expert_buf.dimensions();
  auto tpe_shape = std::vector<size_t>{static_cast<size_t>(tpe_dims[0])};
  auto grad_probs_shape =
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)};

  auto const_buf_tensor =
      TensorWrapper(const_buf_in.untyped_data(), std::vector<size_t>{2}, DType::kFloat32);
  auto tpe_tensor = TensorWrapper(tokens_per_expert_buf.untyped_data(), tpe_shape, tpe_dtype);
  auto grad_aux_loss_tensor =
      TensorWrapper(grad_aux_loss_buf.untyped_data(), std::vector<size_t>{1}, grad_dtype);
  auto grad_probs_tensor =
      TensorWrapper(grad_probs_buf->untyped_data(), grad_probs_shape, grad_dtype);

  nvte_fused_moe_aux_loss_backward(const_buf_tensor.data(), tpe_tensor.data(), num_tokens,
                                   num_experts, grad_aux_loss_tensor.data(),
                                   grad_probs_tensor.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedMoEAuxLossBackwardHandler, FusedMoEAuxLossBackwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // const_buf
                                  .Arg<Buffer_Type>()      // tokens_per_expert
                                  .Arg<Buffer_Type>()      // grad_aux_loss
                                  .Ret<Buffer_Type>(),     // grad_probs
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine
