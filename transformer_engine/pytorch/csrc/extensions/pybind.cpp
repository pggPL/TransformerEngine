/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "pybind.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/stable/c10d.h>
#include <torch/csrc/stable/cuda.h>
#include <torch/csrc/stable/python/interop.h>

#include <memory>
#include <optional>
#include <vector>

#include "../common.h"
#include "../extensions.h"
#include "common.h"

namespace transformer_engine::pytorch {

PyTypeObject *Float8TensorPythonClass = nullptr;  /// TODO Remove
PyTypeObject *Float8TensorStoragePythonClass = nullptr;
PyTypeObject *Float8QuantizerClass = nullptr;
PyTypeObject *Float8CurrentScalingQuantizerClass = nullptr;
PyTypeObject *MXFP8TensorPythonClass = nullptr;  /// TODO Remove
PyTypeObject *MXFP8TensorStoragePythonClass = nullptr;
PyTypeObject *MXFP8QuantizerClass = nullptr;
PyTypeObject *Float8BlockwiseQTensorPythonClass = nullptr;
PyTypeObject *Float8BlockwiseQTensorStoragePythonClass = nullptr;
PyTypeObject *Float8BlockwiseQuantizerClass = nullptr;
PyTypeObject *NVFP4TensorPythonClass = nullptr;
PyTypeObject *NVFP4TensorStoragePythonClass = nullptr;
PyTypeObject *NVFP4QuantizerClass = nullptr;
PyTypeObject *GroupedTensorPythonClass = nullptr;
PyTypeObject *GroupedTensorStoragePythonClass = nullptr;
std::once_flag extension_init_flag;

void init_float8_extension() {
  auto fp8_module = nb::module_::import_("transformer_engine.pytorch.tensor.float8_tensor");
  Float8QuantizerClass =
      reinterpret_cast<PyTypeObject *>(PyObject_GetAttrString(fp8_module.ptr(), "Float8Quantizer"));
  Float8CurrentScalingQuantizerClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(fp8_module.ptr(), "Float8CurrentScalingQuantizer"));
  Float8TensorPythonClass =
      reinterpret_cast<PyTypeObject *>(PyObject_GetAttrString(fp8_module.ptr(), "Float8Tensor"));
  auto fp8_base_module =
      nb::module_::import_("transformer_engine.pytorch.tensor.storage.float8_tensor_storage");
  Float8TensorStoragePythonClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(fp8_base_module.ptr(), "Float8TensorStorage"));
  NVTE_CHECK(Float8TensorPythonClass != nullptr,
             "Internal error: could not initialize pyTorch Float8 extension.");
}

void init_mxfp8_extension() {
  auto fp8_module = nb::module_::import_("transformer_engine.pytorch.tensor.mxfp8_tensor");
  MXFP8QuantizerClass =
      reinterpret_cast<PyTypeObject *>(PyObject_GetAttrString(fp8_module.ptr(), "MXFP8Quantizer"));
  MXFP8TensorPythonClass =
      reinterpret_cast<PyTypeObject *>(PyObject_GetAttrString(fp8_module.ptr(), "MXFP8Tensor"));
  auto fp8_base_module =
      nb::module_::import_("transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage");
  MXFP8TensorStoragePythonClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(fp8_base_module.ptr(), "MXFP8TensorStorage"));
  NVTE_CHECK(MXFP8TensorPythonClass != nullptr,
             "Internal error: could not initialize pyTorch MXFP8 extension.");
}

void init_float8blockwise_extension() {
  auto fp8_module =
      nb::module_::import_("transformer_engine.pytorch.tensor.float8_blockwise_tensor");
  auto fp8_base_module = nb::module_::import_(
      "transformer_engine.pytorch.tensor.storage.float8_blockwise_tensor_storage");
  Float8BlockwiseQuantizerClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(fp8_module.ptr(), "Float8BlockQuantizer"));
  Float8BlockwiseQTensorStoragePythonClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(fp8_base_module.ptr(), "Float8BlockwiseQTensorStorage"));
  Float8BlockwiseQTensorPythonClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(fp8_module.ptr(), "Float8BlockwiseQTensor"));

  NVTE_CHECK(Float8BlockwiseQuantizerClass != nullptr,
             "Internal error: could not initialize pyTorch float8blockwise extension.");
  NVTE_CHECK(Float8BlockwiseQTensorStoragePythonClass != nullptr,
             "Internal error: could not initialize pyTorch float8blockwise extension.");
  NVTE_CHECK(Float8BlockwiseQTensorPythonClass != nullptr,
             "Internal error: could not initialize pyTorch float8blockwise extension.");
}

void init_nvfp4_extensions() {
  auto nvfp4_module = nb::module_::import_("transformer_engine.pytorch.tensor.nvfp4_tensor");
  NVFP4QuantizerClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(nvfp4_module.ptr(), "NVFP4Quantizer"));
  NVFP4TensorPythonClass =
      reinterpret_cast<PyTypeObject *>(PyObject_GetAttrString(nvfp4_module.ptr(), "NVFP4Tensor"));
  auto nvfp4_base_module =
      nb::module_::import_("transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage");
  NVFP4TensorStoragePythonClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(nvfp4_base_module.ptr(), "NVFP4TensorStorage"));
  NVTE_CHECK(NVFP4TensorPythonClass != nullptr,
             "Internal error: could not initialize pyTorch NVFP4 extension.");
}

void init_grouped_tensor_extension() {
  if (GroupedTensorPythonClass && GroupedTensorStoragePythonClass) return;
  auto grouped_tensor_module =
      nb::module_::import_("transformer_engine.pytorch.tensor.grouped_tensor");
  GroupedTensorPythonClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(grouped_tensor_module.ptr(), "GroupedTensor"));
  auto grouped_tensor_storage_module =
      nb::module_::import_("transformer_engine.pytorch.tensor.storage.grouped_tensor_storage");
  GroupedTensorStoragePythonClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(grouped_tensor_storage_module.ptr(), "GroupedTensorStorage"));
  NVTE_CHECK(GroupedTensorPythonClass != nullptr,
             "Internal error: could not initialize pyTorch grouped tensor extension.");
  NVTE_CHECK(GroupedTensorStoragePythonClass != nullptr,
             "Internal error: could not initialize pyTorch grouped tensor extension.");
}

void init_extension() {
  std::call_once(extension_init_flag, []() {
    init_float8_extension();
    init_mxfp8_extension();
    init_float8blockwise_extension();
    init_nvfp4_extensions();
    init_grouped_tensor_extension();
  });
}

// Pybind11 registrations for the fused MoE router kernels. Split out of
// PYBIND11_MODULE() to keep that function under the cpplint readability/fn_size
// limit.
void init_router_bindings(nb::module_ &m) {
  nb::enum_<NVTERoutingMapFormat>(m, "NVTERoutingMapFormat")
      .value("BYTEMAP", NVTE_ROUTING_MAP_FORMAT_BYTEMAP)
      .value("BITMAP_U8", NVTE_ROUTING_MAP_FORMAT_BITMAP_U8);
  m.def("fused_topk_with_score_function_fwd", &fused_topk_with_score_function_fwd,
        nb::arg("logits"), nb::arg("topk"), nb::arg("use_pre_softmax"), nb::arg("num_groups"),
        nb::arg("group_topk"), nb::arg("scaling_factor"), nb::arg("score_function"),
        nb::arg("expert_bias"),
        nb::arg("routing_map_format") = static_cast<int>(NVTE_ROUTING_MAP_FORMAT_BYTEMAP),
        nb::arg("topk_indices") = std::nullopt, "Fused topk with score function fwd");
  m.def("fused_topk_with_score_function_bwd", &fused_topk_with_score_function_bwd,
        nb::arg("routing_map"), nb::arg("intermediate_output"), nb::arg("grad_probs"),
        nb::arg("grad_logits"), nb::arg("topk"), nb::arg("use_pre_softmax"),
        nb::arg("scaling_factor"), nb::arg("score_function"), nb::arg("use_dense_indices") = false,
        nb::arg("routing_map_format") = static_cast<int>(NVTE_ROUTING_MAP_FORMAT_BYTEMAP),
        "Fused topk with score function bwd");
  m.def("fused_score_for_moe_aux_loss_fwd", &fused_score_for_moe_aux_loss_fwd, nb::arg("logits"),
        nb::arg("topk"), nb::arg("score_function"),
        nb::arg("routing_map_format") = static_cast<int>(NVTE_ROUTING_MAP_FORMAT_BYTEMAP),
        "Fused aux loss with score function fwd");
  m.def("fused_score_for_moe_aux_loss_bwd", &fused_score_for_moe_aux_loss_bwd,
        nb::arg("intermediate_output"), nb::arg("grad_scores"), nb::arg("grad_logits"),
        nb::arg("topk"), nb::arg("score_function"), "Fused aux loss with score function bwd");
  m.def("fused_moe_aux_loss_fwd", &fused_moe_aux_loss_fwd, nb::arg("probs"),
        nb::arg("tokens_per_expert"), nb::arg("total_num_tokens"), nb::arg("num_experts"),
        nb::arg("num_rows"), nb::arg("num_cols"), nb::arg("topk"), nb::arg("coeff"),
        "Fused aux loss fwd");
  m.def("fused_moe_aux_loss_bwd", &fused_moe_aux_loss_bwd, nb::arg("Const_buf"),
        nb::arg("tokens_per_expert"), nb::arg("num_rows"), nb::arg("num_cols"),
        nb::arg("grad_aux_loss"), "Fused aux loss bwd");
}

void bind_quantize_with_amax_extensions(nb::module_ &m) {
  m.def("nvfp4_quantize_with_amax", nvfp4_quantize_with_amax, nb::arg("tensor"),
        nb::arg("quantizer"), nb::arg("rowwise_amax"), nb::arg("columnwise_amax"));
  m.def("nvfp4_group_quantize_with_amax", nvfp4_group_quantize_with_amax, nb::arg("tensor"),
        nb::arg("quantizer"), nb::arg("num_tensors"), nb::arg("first_dims"),
        nb::arg("last_dims") = nb::none(), nb::arg("rowwise_amax"), nb::arg("columnwise_amax"),
        nb::arg("tensor_offsets") = nb::none());
}

}  // namespace transformer_engine::pytorch

#include "common/util/pybind_helper.h"

NB_MODULE(TORCH_EXTENSION_NAME, m) {
  // Registers the common TE enums/handles (DType, attention enums, CommOverlap*
  // core classes, etc.) on the module. The nanobind flavor of the shared macro
  // in common/util/pybind_helper.h; the pybind11 flavor is used by JAX.
  NVTE_DECLARE_COMMON_NANOBIND_HANDLES(m)

  // Register __eq__/__ne__ on the pybind ``DType`` enum so it compares by integer.
  nb::object dtype_class = m.attr("DType");
  dtype_class.attr("__eq__") = nb::cpp_function(
      [](transformer_engine::DType self, nb::object other) -> nb::object {
        return nb::cast(static_cast<int>(self) == nb::cast<int>(other));
      },
      nb::is_method());
  dtype_class.attr("__ne__") = nb::cpp_function(
      [](transformer_engine::DType self, nb::object other) -> nb::object {
        return nb::cast(static_cast<int>(self) != nb::cast<int>(other));
      },
      nb::is_method());

  // Override pickling so a ``DType`` value encodes as ``(tex.DType, (int,))``.
  // Only the class itself then needs to be allow-listed for safe unpickling
  // (see transformer_engine/pytorch/__init__.py).
  dtype_class.attr("__reduce__") = nb::cpp_function(
      [](transformer_engine::DType self) {
        return nb::make_tuple(nb::type<transformer_engine::DType>(),
                              nb::make_tuple(static_cast<int>(self)));
      },
      nb::is_method());
  dtype_class.attr("__reduce_ex__") = nb::cpp_function(
      [](transformer_engine::DType self, nb::object /*protocol*/) {
        return nb::make_tuple(nb::type<transformer_engine::DType>(),
                              nb::make_tuple(static_cast<int>(self)));
      },
      nb::is_method());

  m.def("quantize", transformer_engine::pytorch::quantize, nb::arg("tensor"), nb::arg("quantizer"),
        nb::arg("output") = nb::none(), nb::arg("noop") = nb::none());
  m.def("dequantize", &transformer_engine::pytorch::dequantize, "Dequantize", nb::arg("input"),
        nb::arg("otype"));
  m.def("create_empty_quantized_tensor",
        &transformer_engine::pytorch::create_empty_quantized_tensor,
        "Create an empty quantized tensor", nb::arg("quantizer"), nb::arg("shape"),
        nb::arg("dtype"), nb::arg("device"), nb::arg("pin_memory"));
  m.def("group_quantize", transformer_engine::pytorch::group_quantize, nb::arg("tensor"),
        nb::arg("quantizer"), nb::arg("num_tensors"), nb::arg("first_dims"),
        nb::arg("last_dims") = nb::none(), nb::arg("tensor_offsets") = nb::none(),
        nb::arg("noop_flag") = nb::none());
  transformer_engine::pytorch::bind_quantize_with_amax_extensions(m);
  m.def("group_dequantize", transformer_engine::pytorch::group_dequantize,
        "Dequantize group tensor", nb::arg("input"), nb::arg("otype"));
  m.def("bgrad_group_quantize", transformer_engine::pytorch::bgrad_group_quantize,
        nb::arg("tensor"), nb::arg("quantizer"), nb::arg("num_tensors"), nb::arg("first_dims"),
        nb::arg("last_dims") = nb::none(), nb::arg("tensor_offsets") = nb::none());
  m.def("bgrad_quantize", transformer_engine::pytorch::bgrad_quantize,
        "Compute bias gradient and quantize", nb::arg("input"), nb::arg("quantizer"));
  m.def("generic_gemm", transformer_engine::pytorch::gemm, "Compute GEMM (matrix-matrix multiply)",
        nb::arg("A"), nb::arg("transA"), nb::arg("B"), nb::arg("transB"), nb::arg("D"),
        nb::arg("quantizer"), nb::arg("output_dtype"), nb::arg("bias"), nb::arg("bias_type"),
        nb::arg("gelu"), nb::arg("gelu_in"), nb::arg("grad"), nb::arg("workspace"),
        nb::arg("workspace_size"), nb::arg("accumulate"), nb::arg("use_split_accumulator"),
        nb::arg("comm_overlap") = nullptr, nb::arg("comm_type") = std::nullopt,
        nb::arg("extra_output") = std::nullopt, nb::arg("bulk_overlap") = false,
        nb::arg("alpha") = 1.0f, nb::arg("beta") = std::nullopt);
  /* GLU (sigmoid gate) */
  m.def("glu", transformer_engine::pytorch::glu, "GLU activation", nb::arg("input"),
        nb::arg("quantizer"));
  /* GELU and variants*/
  m.def("gelu", transformer_engine::pytorch::gelu, "GeLU activation", nb::arg("input"),
        nb::arg("quantizer"));
  m.def("geglu", transformer_engine::pytorch::geglu, "GeGLU activation", nb::arg("input"),
        nb::arg("quantizer"));
  m.def("qgelu", transformer_engine::pytorch::qgelu, "QuickGELU activation", nb::arg("input"),
        nb::arg("quantizer"));
  m.def("qgeglu", transformer_engine::pytorch::qgeglu, "QuickGeGLU activation", nb::arg("input"),
        nb::arg("quantizer"));
  /* ReLU and variants */
  m.def("relu", transformer_engine::pytorch::relu, "ReLU activation", nb::arg("input"),
        nb::arg("quantizer"));
  m.def("reglu", transformer_engine::pytorch::reglu, "ReGLU activation", nb::arg("input"),
        nb::arg("quantizer"));
  m.def("srelu", transformer_engine::pytorch::srelu, "Squared ReLU activation", nb::arg("input"),
        nb::arg("quantizer"));
  m.def("sreglu", transformer_engine::pytorch::sreglu, "Squared ReGLU activation", nb::arg("input"),
        nb::arg("quantizer"));
  /* SwiGLU and variants */
  m.def("silu", transformer_engine::pytorch::silu, "SiLU activation", nb::arg("input"),
        nb::arg("quantizer"));
  m.def("swiglu", transformer_engine::pytorch::swiglu, "SwiGLU activation", nb::arg("input"),
        nb::arg("quantizer"));
  m.def("clamped_swiglu", transformer_engine::pytorch::clamped_swiglu,
        "SwiGLU activation used in GPT OSS", nb::arg("input"), nb::arg("quantizer"),
        nb::arg("limit") = 7.0f, nb::arg("alpha") = 1.702f, nb::arg("glu_linear_offset") = 1.0f);
  /* Backward of GLU */
  m.def("dglu", transformer_engine::pytorch::dglu, "Backward of GLU", nb::arg("grad"),
        nb::arg("fwd_input"), nb::arg("quantizer"));
  /* Backward of GELU and variants */
  m.def("dgelu", transformer_engine::pytorch::dgelu, "Backward of GeLU", nb::arg("grad"),
        nb::arg("fwd_input"), nb::arg("quantizer"));
  m.def("dgeglu", transformer_engine::pytorch::dgeglu, "Backward of GeGLU", nb::arg("grad"),
        nb::arg("fwd_input"), nb::arg("quantizer"));
  m.def("dqgelu", transformer_engine::pytorch::dqgelu, "Backward of QuickGELU", nb::arg("grad"),
        nb::arg("fwd_input"), nb::arg("quantizer"));
  m.def("dqgeglu", transformer_engine::pytorch::dqgeglu, "Backward of QuickGeGLU", nb::arg("grad"),
        nb::arg("fwd_input"), nb::arg("quantizer"));
  /* Backward of ReLU and variants */
  m.def("drelu", transformer_engine::pytorch::drelu, "Backward of ReLU", nb::arg("grad"),
        nb::arg("fwd_input"), nb::arg("quantizer"));
  m.def("dreglu", transformer_engine::pytorch::dreglu, "Backward of ReGLU", nb::arg("grad"),
        nb::arg("fwd_input"), nb::arg("quantizer"));
  m.def("dsrelu", transformer_engine::pytorch::dsrelu, "Backward of Squared ReLU", nb::arg("grad"),
        nb::arg("fwd_input"), nb::arg("quantizer"));
  m.def("dsreglu", transformer_engine::pytorch::dsreglu, "Backward of Squared ReGLU",
        nb::arg("grad"), nb::arg("fwd_input"), nb::arg("quantizer"));
  /* Backward of SiLU and variants */
  m.def("dsilu", transformer_engine::pytorch::dsilu, "Backward of SiLU", nb::arg("grad"),
        nb::arg("fwd_input"), nb::arg("quantizer"));
  m.def("dswiglu", transformer_engine::pytorch::dswiglu, "Backward of SwiGLU", nb::arg("grad"),
        nb::arg("fwd_input"), nb::arg("quantizer"));
  m.def("clamped_dswiglu", transformer_engine::pytorch::clamped_dswiglu,
        "Backward of SwiGLU used in GPT OSS", nb::arg("grad"), nb::arg("fwd_input"),
        nb::arg("quantizer"), nb::arg("limit") = 7.0f, nb::arg("alpha") = 1.702f,
        nb::arg("glu_linear_offset") = 1.0f);
  /* DBias + DAct fusions*/
  m.def("dbias_dgelu", transformer_engine::pytorch::dbias_dgelu, "DGeLU + DBias + Quantize",
        nb::arg("grad"), nb::arg("fwd_input"), nb::arg("quantizer"));
  m.def("dbias_dsilu", transformer_engine::pytorch::dbias_dsilu, "DSiLU + DBias + Quantize",
        nb::arg("grad"), nb::arg("fwd_input"), nb::arg("quantizer"));
  m.def("dbias_drelu", transformer_engine::pytorch::dbias_drelu, "DReLU + DBias + Quantize",
        nb::arg("grad"), nb::arg("fwd_input"), nb::arg("quantizer"));
  m.def("dbias_dqgelu", transformer_engine::pytorch::dbias_dqgelu, "DQGeLU + DBias + Quantize",
        nb::arg("grad"), nb::arg("fwd_input"), nb::arg("quantizer"));
  m.def("dbias_dsrelu", transformer_engine::pytorch::dbias_dsrelu,
        "DSquaredReLU + DBias + Quantize", nb::arg("grad"), nb::arg("fwd_input"),
        nb::arg("quantizer"));

  // Permutation functions
  m.def("moe_permute_fwd", transformer_engine::pytorch::moe_permute_fwd, "MOE permute FWD",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("moe_permute_bwd", transformer_engine::pytorch::moe_permute_bwd, "MOE permute BWD",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("moe_unpermute_fwd", transformer_engine::pytorch::moe_unpermute_fwd, "MOE unpermute FWD",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("moe_unpermute_bwd", transformer_engine::pytorch::moe_unpermute_bwd, "MOE unpermute BWD",
        nb::call_guard<nb::gil_scoped_release>());

  // Softmax functions
  m.def("scaled_softmax_forward", &transformer_engine::pytorch::scaled_softmax_forward,
        "Scaled Softmax FWD", nb::call_guard<nb::gil_scoped_release>());
  m.def("scaled_softmax_backward", &transformer_engine::pytorch::scaled_softmax_backward,
        "Scaled Softmax BWD", nb::call_guard<nb::gil_scoped_release>());
  m.def("scaled_masked_softmax_forward",
        &transformer_engine::pytorch::scaled_masked_softmax_forward, "Scaled Masked Softmax FWD",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("scaled_masked_softmax_backward",
        &transformer_engine::pytorch::scaled_masked_softmax_backward, "Scaled Masked Softmax BWD",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("scaled_upper_triang_masked_softmax_forward",
        &transformer_engine::pytorch::scaled_upper_triang_masked_softmax_forward,
        "Scaled Upper-Triangular Masked Softmax FWD", nb::call_guard<nb::gil_scoped_release>());
  m.def("scaled_upper_triang_masked_softmax_backward",
        &transformer_engine::pytorch::scaled_upper_triang_masked_softmax_backward,
        "Scaled Upper-Triangular Masked Softmax BWD", nb::call_guard<nb::gil_scoped_release>());
  m.def("scaled_aligned_causal_masked_softmax_forward",
        &transformer_engine::pytorch::scaled_aligned_causal_masked_softmax_forward,
        "Scaled Bottom-Right Corner Aligned Masked Softmax FWD",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("scaled_aligned_causal_masked_softmax_backward",
        &transformer_engine::pytorch::scaled_aligned_causal_masked_softmax_backward,
        "Scaled Bottom-Right Corner Aligned Masked Softmax BWD",
        nb::call_guard<nb::gil_scoped_release>());

  // Other granular functions
  m.def("layernorm_fwd", &transformer_engine::pytorch::layernorm_fwd, "LayerNorm", nb::arg("input"),
        nb::arg("weight"), nb::arg("bias"), nb::arg("eps"), nb::arg("ln_out"), nb::arg("quantizer"),
        nb::arg("otype"), nb::arg("sm_margin"), nb::arg("zero_centered_gamma"));
  m.def("layernorm_bwd", &transformer_engine::pytorch::layernorm_bwd, "Backward of LayerNorm");
  m.def("rmsnorm_fwd", &transformer_engine::pytorch::rmsnorm_fwd, "RMSNorm", nb::arg("input"),
        nb::arg("weight"), nb::arg("eps"), nb::arg("ln_out"), nb::arg("quantizer"),
        nb::arg("otype"), nb::arg("sm_margin"), nb::arg("zero_centered_gamma"));
  m.def("rmsnorm_bwd", &transformer_engine::pytorch::rmsnorm_bwd, "Backward of RMSNorm");
  m.def("rmsnorm_bwd_add", &transformer_engine::pytorch::rmsnorm_bwd_add,
        "Fused backward of RMSNorm + add");
  m.def("multi_tensor_quantize", &transformer_engine::pytorch::multi_tensor_quantize,
        "Multi-tensor quantize", nb::arg("tensor_list"), nb::arg("quantizer_list"));
  m.def("split_quantize", &transformer_engine::pytorch::split_quantize,
        "Split and multi-tensor quantize", nb::arg("tensor"), nb::arg("split_sections"),
        nb::arg("quantizer_list"), nb::arg("disable_bulk_allocation") = false);
  m.def("get_grouped_gemm_setup_workspace_size", &nvte_get_grouped_gemm_setup_workspace_size,
        "Required workspace size for grouped GEMM setup");
  m.def("te_general_grouped_gemm", &transformer_engine::pytorch::te_general_grouped_gemm,
        "Grouped GEMM");
  m.def("te_general_grouped_gemm_for_grouped_tensor",
        &transformer_engine::pytorch::te_general_grouped_gemm_for_grouped_tensor,
        "Grouped GEMM for GroupedTensor");
  m.def("te_general_grouped_gemm_for_discrete_in",
        &transformer_engine::pytorch::te_general_grouped_gemm_for_discrete_in,
        "Grouped GEMM for discrete A input list");
  m.def("te_general_grouped_gemm_for_discrete_out",
        &transformer_engine::pytorch::te_general_grouped_gemm_for_discrete_out,
        "Grouped GEMM for discrete output list");
  m.def("fp8_transpose", &transformer_engine::pytorch::fp8_transpose, "Transpose with FP8 I/O",
        nb::arg("input"), nb::arg("dtype"), nb::kw_only(), nb::arg("out"),
        nb::call_guard<nb::gil_scoped_release>());
  m.def("nvfp4_data_transpose", &transformer_engine::pytorch::nvfp4_data_transpose,
        "Transpose NVFP4 packed data with nibble repacking", nb::arg("input"), nb::kw_only(),
        nb::arg("out"), nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "nvfp4_2d_scale_transpose", &transformer_engine::pytorch::nvfp4_2d_scale_transpose,
      "Transpose NVFP4 tile-level scales (E4M3 stored as uint8) from rowwise to columnwise format",
      nb::arg("input"), nb::arg("output"), nb::arg("M_tiles"), nb::arg("K_tiles"),
      nb::call_guard<nb::gil_scoped_release>());
  m.def("nvfp4_expand_scale_to_fp8", &transformer_engine::pytorch::nvfp4_expand_scale_to_fp8,
        "Expand tile-level scales to row-level scales and convert to FP8 E4M3", nb::arg("input"),
        nb::arg("output"), nb::arg("tile_rows"), nb::arg("tile_cols"), nb::arg("rows_padded"),
        nb::arg("block_len"), nb::call_guard<nb::gil_scoped_release>());
  m.def("nvfp4_compute_per_block_scale",
        &transformer_engine::pytorch::nvfp4_compute_per_block_scale,
        "Compute per-block decode scale from block amax and global amax", nb::arg("block_amax"),
        nb::arg("scale"), nb::arg("global_amax"), nb::call_guard<nb::gil_scoped_release>());
  m.def("nvfp4_compute_global_scale", &transformer_engine::pytorch::nvfp4_compute_global_scale,
        "Compute global encode scale from global amax", nb::arg("global_amax"),
        nb::arg("global_scale"), nb::call_guard<nb::gil_scoped_release>());
  m.def("nvfp4_fused_scale", &transformer_engine::pytorch::nvfp4_fused_scale,
        "Fused kernel: compute per-block decode scale, copy global amax, expand to row-level FP8",
        nb::arg("block_amax"), nb::arg("global_amax"), nb::arg("per_block_scale"),
        nb::arg("target_scale"), nb::arg("target_amax"), nb::arg("tile_rows"), nb::arg("tile_cols"),
        nb::arg("rows_padded"), nb::arg("block_len"), nb::call_guard<nb::gil_scoped_release>());
  m.def("nvfp4_multi_tensor_fused_scale",
        &transformer_engine::pytorch::nvfp4_multi_tensor_fused_scale,
        "Batched fused scale: compute per-block decode scale, copy global amax, expand to FP8 for "
        "multiple tensors",
        nb::arg("block_amax_list"), nb::arg("global_amax_list"), nb::arg("per_block_scale_list"),
        nb::arg("target_scale_list"), nb::arg("target_amax_list"), nb::arg("tile_rows_list"),
        nb::arg("tile_cols_list"), nb::arg("rows_padded_list"), nb::arg("block_len"),
        nb::call_guard<nb::gil_scoped_release>());
  m.def("nvfp4_2d_multi_tensor_transpose",
        &transformer_engine::pytorch::nvfp4_2d_multi_tensor_transpose,
        "Batched NVFP4 columnwise creation: transpose data and scales for multiple tensors",
        nb::arg("rowwise_data_list"), nb::arg("columnwise_data_list"),
        nb::arg("rowwise_scale_inv_list"), nb::arg("columnwise_scale_inv_list"), nb::arg("M_list"),
        nb::arg("K_list"), nb::call_guard<nb::gil_scoped_release>());
  m.def("swap_first_dims", &transformer_engine::pytorch::swap_first_dims,
        "Swap first two tensor dimensions", nb::arg("tensor"), nb::kw_only(), nb::arg("out"),
        nb::call_guard<nb::gil_scoped_release>());
  m.def("get_fused_attn_backend", &transformer_engine::pytorch::get_fused_attn_backend,
        "Get Fused Attention backend", nb::call_guard<nb::gil_scoped_release>());
  m.def("compute_amax", &transformer_engine::pytorch::compute_amax,
        "Compute absolute max value in tensor", nb::arg("input"), nb::arg("amax"),
        nb::call_guard<nb::gil_scoped_release>());
  m.def("fused_amax_and_scale_update_after_reduction",
        &transformer_engine::pytorch::fused_amax_and_scale_update_after_reduction,
        "Update amax history and FP8 scale/scale_inv after reduction",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("fp8_block_scaling_compute_partial_amax",
        &transformer_engine::pytorch::fp8_block_scaling_compute_partial_amax,
        "Compute partial amax from master weights for fp8 block scaling", nb::arg("tensor"),
        nb::arg("amax"), nb::arg("h"), nb::arg("w"), nb::arg("start_offset"), nb::arg("block_len"),
        nb::call_guard<nb::gil_scoped_release>());
  m.def("fp8_block_scaling_partial_cast",
        &transformer_engine::pytorch::fp8_block_scaling_partial_cast,
        "Partial cast from master weights for fp8 block scaling", nb::arg("inp"), nb::arg("out"),
        nb::arg("scale"), nb::arg("h"), nb::arg("w"), nb::arg("start_offset"), nb::arg("block_len"),
        nb::arg("out_dtype"), nb::call_guard<nb::gil_scoped_release>());

  // NVFP4 2D
  m.def("nvfp4_2d_compute_partial_amax",
        &transformer_engine::pytorch::nvfp4_2d_compute_partial_amax,
        "Compute partial amax from master weights for NVFP4 2D", nb::arg("tensor"), nb::arg("amax"),
        nb::arg("h"), nb::arg("w"), nb::arg("start_offset"), nb::arg("block_len") = 16,
        nb::call_guard<nb::gil_scoped_release>());
  m.def("nvfp4_multi_tensor_compute_partial_amax",
        &transformer_engine::pytorch::nvfp4_multi_tensor_compute_partial_amax,
        "Batched compute partial and global amax from master weights for NVFP4 2D",
        nb::arg("master_weight_list"), nb::arg("partial_amax_list"), nb::arg("global_amax_list"),
        nb::arg("h_list"), nb::arg("w_list"), nb::arg("start_offset_list"),
        nb::arg("block_len") = 16, nb::call_guard<nb::gil_scoped_release>());
  m.def("nvfp4_2d_partial_cast", &transformer_engine::pytorch::nvfp4_2d_partial_cast,
        "Partial cast from master weights for NVFP4 2D", nb::arg("inp"), nb::arg("out"),
        nb::arg("scale"), nb::arg("global_scale"), nb::arg("h"), nb::arg("w"),
        nb::arg("start_offset"), nb::arg("block_len") = 16,
        nb::call_guard<nb::gil_scoped_release>());
  m.def("nvfp4_multi_tensor_2d_partial_cast",
        &transformer_engine::pytorch::nvfp4_multi_tensor_2d_partial_cast,
        "Batched partial cast from master weights for NVFP4 2D", nb::arg("inp_list"),
        nb::arg("out_list"), nb::arg("scale_list"), nb::arg("global_scale_list"), nb::arg("h_list"),
        nb::arg("w_list"), nb::arg("start_offset_list"), nb::arg("block_len") = 16,
        nb::call_guard<nb::gil_scoped_release>());
  m.def("mxfp8_scaling_compute_partial_amax",
        &transformer_engine::pytorch::mxfp8_scaling_compute_partial_amax,
        "Compute partial amax from master weights for fp8 mxfp8 scaling", nb::arg("input"),
        nb::arg("amax_rowwise"), nb::arg("amax_colwise"), nb::arg("rows"), nb::arg("cols"),
        nb::arg("start_offset"), nb::call_guard<nb::gil_scoped_release>());
  m.def("mxfp8_scaling_partial_cast", &transformer_engine::pytorch::mxfp8_scaling_partial_cast,
        "Partial cast from master weights for fp8 mxfp8 scaling", nb::arg("input"),
        nb::arg("output_rowwise"), nb::arg("output_colwise"), nb::arg("scale_inv_rowwise"),
        nb::arg("scale_inv_colwise"), nb::arg("rows"), nb::arg("cols"), nb::arg("start_offset"),
        nb::call_guard<nb::gil_scoped_release>());
  m.def("fused_multi_row_padding", &transformer_engine::pytorch::fused_multi_row_padding,
        "Fused Multi-tensor padding", nb::call_guard<nb::gil_scoped_release>());
  m.def("fused_multi_row_unpadding", &transformer_engine::pytorch::fused_multi_row_unpadding,
        "Fused Multi-tensor unpadding", nb::call_guard<nb::gil_scoped_release>());
  m.def("swizzle_scales_for_gemm_", &transformer_engine::pytorch::inplace_swizzle_scale_for_gemm,
        "Convert tensor block scales into GEMM swizzled format");
  m.def("multi_tensor_swizzle_scales_for_gemm_",
        &transformer_engine::pytorch::inplace_multi_tensor_swizzle_scales_for_gemm,
        "Convert multiple tensors' block scales into GEMM swizzled format", nb::arg("tensors"),
        nb::arg("rowwise_usage"), nb::arg("columnwise_usage"));
  m.def(
      "multi_tensor_swizzle_scales_for_gemm_unchecked_",
      &transformer_engine::pytorch::inplace_multi_tensor_swizzle_scales_for_gemm_unchecked,
      "Convert multiple tensors' block scales into GEMM swizzled format (skip scale shape checks)",
      nb::arg("tensors"), nb::arg("rowwise_usage"), nb::arg("columnwise_usage"));
  m.def("grouped_swizzle_for_gemm", &transformer_engine::pytorch::grouped_swizzle_for_gemm,
        "In-place swizzle of grouped tensor scales for GEMM", nb::arg("tensor"), nb::arg("rowwise"),
        nb::arg("columnwise"));

  // Tensor allocation
  m.def("bulk_allocate", &transformer_engine::pytorch::bulk_allocate,
        "Allocate tensors backed by a single contiguous buffer", nb::arg("shapes"),
        nb::arg("dtypes"), nb::arg("device") = nb::none(), nb::arg("alignments") = nb::none(),
        nb::call_guard<nb::gil_scoped_release>());

  // attention kernels
  m.def("fa_prepare_fwd", &transformer_engine::pytorch::fa_prepare_fwd,
        "Prepare QKV for Flash Attention", nb::call_guard<nb::gil_scoped_release>());
  m.def("fa_prepare_bwd", &transformer_engine::pytorch::fa_prepare_bwd,
        "Backward of QKV preparation for Flash Attention",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("multi_tensor_transpose_to_bhsd",
        &transformer_engine::pytorch::multi_tensor_transpose_to_bhsd,
        "Permute multiple tensors from BSHD/SBHD to BHSD.", nb::arg("inputs"),
        nb::arg("original_format"), nb::arg("outputs") = std::vector<std::optional<torch::stable::Tensor>>{},
        nb::call_guard<nb::gil_scoped_release>());
  m.def("multi_tensor_pad_last_dim", &transformer_engine::pytorch::multi_tensor_pad_last_dim,
        "Pad multiple tensors' last dimension to a common alignment.", nb::arg("inputs"),
        nb::arg("alignment"), nb::call_guard<nb::gil_scoped_release>());
  m.def("fused_attn_fwd", &transformer_engine::pytorch::fused_attn_fwd,
        "Fused Attention FP8/BF16/FP16 FWD with separate Q, K and V");
  m.def("fused_attn_bwd", &transformer_engine::pytorch::fused_attn_bwd,
        "Fused Attention FP8/BF16/FP16 BWD with separate Q, K and V");
  m.def("copy_to_kv_cache", &transformer_engine::pytorch::copy_to_kv_cache,
        "Copy new KV tokens to KV cache", nb::call_guard<nb::gil_scoped_release>());
  m.def("convert_thd_to_bshd", &transformer_engine::pytorch::convert_thd_to_bshd,
        "Convert a tensor from THD to BSHD", nb::call_guard<nb::gil_scoped_release>());
  m.def("convert_bshd_to_thd", &transformer_engine::pytorch::convert_bshd_to_thd,
        "Convert a tesnor from BSHD to THD", nb::call_guard<nb::gil_scoped_release>());

  // fused apply rope
  m.def("fused_rope_forward", &transformer_engine::pytorch::fused_rope_forward,
        "Fused Apply RoPE FWD", nb::call_guard<nb::gil_scoped_release>());
  m.def("fused_rope_backward", &transformer_engine::pytorch::fused_rope_backward,
        "Fused Apply RoPE BWD", nb::call_guard<nb::gil_scoped_release>());
  m.def("fused_qkv_rope_forward", &transformer_engine::pytorch::fused_qkv_rope_forward,
        "Fused Apply QKV RoPE FWD", nb::call_guard<nb::gil_scoped_release>());
  m.def("fused_qkv_rope_backward", &transformer_engine::pytorch::fused_qkv_rope_backward,
        "Fused Apply QKV RoPE BWD", nb::call_guard<nb::gil_scoped_release>());

  // fused router
  transformer_engine::pytorch::init_router_bindings(m);

  // Dropout
  m.def("dropout_fwd", transformer_engine::pytorch::dropout_fwd, "Dropout forward with 8-bit RNG",
        nb::arg("input"), nb::arg("dropout_probability"), nb::arg("out") = std::nullopt);
  m.def("dropout_bwd", transformer_engine::pytorch::dropout_bwd, "Dropout backward with 8-bit RNG",
        nb::arg("grad_output"), nb::arg("mask"), nb::arg("dropout_probability"),
        nb::arg("grad_input") = std::nullopt);

  // Misc
  m.def("get_cublasLt_version", &transformer_engine::pytorch::get_cublasLt_version,
        "Get cublasLt version", nb::call_guard<nb::gil_scoped_release>());
  m.def("get_cudnn_version", &transformer_engine::pytorch::get_cudnn_version, "Get cuDNN version",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("copy_data_ptrs_to_device", &transformer_engine::pytorch::copy_data_ptrs_to_device,
        nb::arg("tensors"), nb::arg("device"), nb::call_guard<nb::gil_scoped_release>());
  m.def("splits_to_offsets", &transformer_engine::pytorch::splits_to_offsets,
        "Compute grouped tensor offsets from split sizes", nb::arg("first_dims"),
        nb::arg("logical_last_dim"), nb::call_guard<nb::gil_scoped_release>());
  m.def("splits_to_offsets_multi", &transformer_engine::pytorch::splits_to_offsets_multi,
        "Compute multiple scaled inclusive-scan offsets from a split-sizes vector",
        nb::arg("split_sizes"), nb::arg("device"), nb::kw_only(), nb::arg("strides"),
        nb::arg("include_leading_zero"), nb::arg("dtypes"), nb::arg("bulk_allocate") = false);
  m.def("get_num_cublas_streams", &nvte_get_num_compute_streams, "Get number of compute streams",
        nb::call_guard<nb::gil_scoped_release>());

  // Support THD format for Context Parallel
  m.def("thd_read_half_tensor", &transformer_engine::pytorch::thd_read_half_tensor,
        "Read the first half(half_idx=0) or the second half(half_idx=1) of each sequence in a THD "
        "tensor",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("thd_second_half_lse_correction",
        &transformer_engine::pytorch::thd_second_half_lse_correction,
        "Correct the second half of the softmax_lse", nb::call_guard<nb::gil_scoped_release>());
  m.def("thd_read_second_half_lse", &transformer_engine::pytorch::thd_read_second_half_lse,
        "Read the second half of the softmax_lse", nb::call_guard<nb::gil_scoped_release>());
  m.def("thd_out_correction", &transformer_engine::pytorch::thd_out_correction,
        "Correct the THD format output of context parallelism in forward pass",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("thd_grad_correction", &transformer_engine::pytorch::thd_grad_correction,
        "Correct the THD format gradients of context parallelism in backward pass",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("thd_get_partitioned_indices", &transformer_engine::pytorch::thd_get_partitioned_indices,
        "Generate partitioned indices for inputs in THD format",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("thd_sequence_order_to_cp_rank_order",
        &transformer_engine::pytorch::thd_sequence_order_to_cp_rank_order,
        "Reorder a THD tensor from sequence order to dual-chunk CP rank order",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("thd_cp_rank_order_to_sequence_order",
        &transformer_engine::pytorch::thd_cp_rank_order_to_sequence_order,
        "Reorder a THD tensor from dual-chunk CP rank order to sequence order",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("thd_copy_valid_tokens_from_per_split_to_rank_local",
        &transformer_engine::pytorch::thd_copy_valid_tokens_from_per_split_to_rank_local,
        "Copy valid THD token entries from a per-split tensor into a rank-local accumulator",
        nb::call_guard<nb::gil_scoped_release>());

  // nvshmem functions
  m.def("init_nvshmem_backend", &transformer_engine::pytorch::init_nvshmem_backend,
        "Initialize nvshmem backend with Pytorch distributed process groups",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("create_nvshmem_tensor", &transformer_engine::pytorch::create_nvshmem_tensor,
        "Create a tensor in NVSHMEM shared memory", nb::call_guard<nb::gil_scoped_release>());
  m.def("nvshmem_send_on_current_stream",
        &transformer_engine::pytorch::nvshmem_send_on_current_stream,
        "Asynchronously send tensor data to a remote PE using NVSHMEM on the current CUDA stream",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("nvshmem_wait_on_current_stream",
        &transformer_engine::pytorch::nvshmem_wait_on_current_stream,
        "Wait for a signal value to be updated by a remote PE using NVSHMEM on the current CUDA "
        "stream",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("nvshmem_finalize", &transformer_engine::pytorch::nvshmem_finalize,
        "Clean up and finalize the NVSHMEM communication backend and free associated resources",
        nb::call_guard<nb::gil_scoped_release>());

  // multi-tensor functions
  m.def("multi_tensor_scale", &transformer_engine::pytorch::multi_tensor_scale_cuda,
        "Fused overflow check + scale for a list of contiguous tensors",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("multi_tensor_scale_tensor", &transformer_engine::pytorch::multi_tensor_scale_tensor_cuda,
        "Fused overflow check + scale for a list of contiguous tensors with scale passed as tensor",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("multi_tensor_l2norm", &transformer_engine::pytorch::multi_tensor_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("multi_tensor_unscale_l2norm",
        &transformer_engine::pytorch::multi_tensor_unscale_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors after unscaling (unscaling is only "
        "performed for L2 norm computation, and tensors are not updated)",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("multi_tensor_adam", &transformer_engine::pytorch::multi_tensor_adam_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("multi_tensor_adam_param_remainder",
        &transformer_engine::pytorch::multi_tensor_adam_param_remainder_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer"
        "where the master parameters only store the remainder bits",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("multi_tensor_adam_fp8", &transformer_engine::pytorch::multi_tensor_adam_fp8_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("multi_tensor_adam_capturable",
        &transformer_engine::pytorch::multi_tensor_adam_capturable_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer with CUDA graph "
        "support and LR scheduling",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("multi_tensor_adam_capturable_master",
        &transformer_engine::pytorch::multi_tensor_adam_capturable_master_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer with CUDA graph "
        "support, LR scheduling and FP32 master weights",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("multi_tensor_sgd", &transformer_engine::pytorch::multi_tensor_sgd_cuda,
        "Fused SGD optimizer for list of contiguous tensors",
        nb::call_guard<nb::gil_scoped_release>());
  m.def("multi_tensor_compute_scale_and_scale_inv",
        &transformer_engine::pytorch::multi_tensor_compute_scale_and_scale_inv_cuda,
        "Fused compute scale and scale_inv from amax", nb::call_guard<nb::gil_scoped_release>());
  m.def("multi_tensor_compute_scale_inv_e8m0",
        &transformer_engine::pytorch::multi_tensor_compute_scale_inv_e8m0_cuda,
        "Fused compute E8M0 scale_inv from amax", nb::call_guard<nb::gil_scoped_release>());

  // Newton-Schulz (cuSolverMp)
  m.def("cusolvermp_ctx_create", &transformer_engine::pytorch::cusolvermp_ctx_create,
        "Create cuSolverMp context for Newton-Schulz", nb::arg("nccl_comm_ptr"), nb::arg("nranks"),
        nb::arg("rank"), nb::call_guard<nb::gil_scoped_release>());
  m.def("cusolvermp_ctx_destroy", &transformer_engine::pytorch::cusolvermp_ctx_destroy,
        "Destroy cuSolverMp context", nb::arg("ctx_ptr"), nb::call_guard<nb::gil_scoped_release>());
  m.def("newton_schulz", &transformer_engine::pytorch::newton_schulz,
        "Newton-Schulz matrix orthogonalization", nb::arg("ctx_ptr"), nb::arg("m"), nb::arg("n"),
        nb::arg("x"), nb::arg("num_iterations"), nb::arg("coefficients"),
        nb::call_guard<nb::gil_scoped_release>());

  // Comm+GEMM Overlap
  // Accepts torch.cuda.Stream objects; their raw cudaStream_t (``.cuda_stream``)
  // is adopted through the stable getStreamFromExternal wrapper.
  m.def(
      "bulk_overlap_ag_with_external_gemm",
      [](CommOverlap &allgather_communicator, nb::object send_stream, nb::object recv_stream) {
        const auto device = torch::stable::accelerator::getCurrentDeviceIndex();
        auto to_stream = [&](nb::object &stream) {
          auto ptr = reinterpret_cast<void *>(nb::cast<uintptr_t>(stream.attr("cuda_stream")));
          return torch::stable::cuda::getStreamFromExternal(ptr, device);
        };
        auto send = to_stream(send_stream);
        auto recv = to_stream(recv_stream);
        transformer_engine::pytorch::bulk_overlap_ag_with_external_gemm(
            allgather_communicator, std::move(send), std::move(recv));
      },
      "Bulk overlap All-Gather with a GEMM operation launched by another communicator",
      nb::arg("allgather_communicator"), nb::arg("send_stream"), nb::arg("recv_stream"));

  // Experimental fused grouped MLP
  auto grouped_mlp_experimental = m.def_submodule(
      "grouped_mlp_experimental",
      "Experimental helpers for the fused grouped MLP (unstable, may change or disappear).");
  grouped_mlp_experimental.def("swizzle_scales_and_pack_ptrs_for_discrete_weights",
                               &transformer_engine::pytorch::grouped_mlp_experimental::
                                   swizzle_scales_and_pack_ptrs_for_discrete_weights,
                               nb::arg("data_tensors"), nb::arg("scale_tensors"),
                               nb::arg("swizzle_type"), nb::arg("device"),
                               nb::call_guard<nb::gil_scoped_release>());

  // Data structures
  nb::class_<transformer_engine::pytorch::FP8TensorMeta>(m, "FP8TensorMeta")
      .def(nb::init<>())
      .def_rw("scale", &transformer_engine::pytorch::FP8TensorMeta::scale)
      .def_rw("scale_inv", &transformer_engine::pytorch::FP8TensorMeta::scale_inv)
      .def_rw("amax_history", &transformer_engine::pytorch::FP8TensorMeta::amax_history);

  nb::enum_<transformer_engine::pytorch::FP8FwdTensors>(m, "FP8FwdTensors")
      .value("GEMM1_INPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM1_INPUT)
      .value("GEMM1_WEIGHT", transformer_engine::pytorch::FP8FwdTensors::GEMM1_WEIGHT)
      .value("GEMM1_OUTPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM1_OUTPUT)
      .value("GEMM2_INPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM2_INPUT)
      .value("GEMM2_WEIGHT", transformer_engine::pytorch::FP8FwdTensors::GEMM2_WEIGHT)
      .value("GEMM2_OUTPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM2_OUTPUT)
      .value("GEMM3_INPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM3_INPUT)
      .value("GEMM3_WEIGHT", transformer_engine::pytorch::FP8FwdTensors::GEMM3_WEIGHT)
      .value("GEMM3_OUTPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM3_OUTPUT);

  nb::enum_<transformer_engine::pytorch::FP8BwdTensors>(m, "FP8BwdTensors")
      .value("GRAD_OUTPUT1", transformer_engine::pytorch::FP8BwdTensors::GRAD_OUTPUT1)
      .value("GRAD_INPUT1", transformer_engine::pytorch::FP8BwdTensors::GRAD_INPUT1)
      .value("GRAD_OUTPUT2", transformer_engine::pytorch::FP8BwdTensors::GRAD_OUTPUT2)
      .value("GRAD_INPUT2", transformer_engine::pytorch::FP8BwdTensors::GRAD_INPUT2)
      .value("GRAD_OUTPUT3", transformer_engine::pytorch::FP8BwdTensors::GRAD_OUTPUT3)
      .value("GRAD_INPUT3", transformer_engine::pytorch::FP8BwdTensors::GRAD_INPUT3);

  // The process groups arrive as Python torch.distributed.ProcessGroup objects
  // and are adopted through the stable-ABI processgroup_from_pyobject bridge.
  // The GIL must be held for that conversion, so this factory does not release it.
  nb::class_<CommOverlapHelper>(m, "CommOverlapHelper")
      .def(nb::init<>(), nb::call_guard<nb::gil_scoped_release>())
      .def(
          "__init__",
          [](CommOverlapHelper *self, nb::object world_group,
             std::optional<nb::object> intra_node_group) {
            auto world = torch::stable::processgroup_from_pyobject(world_group.ptr());
            std::optional<torch::stable::ProcessGroup> intra;
            if (intra_node_group.has_value() && !intra_node_group->is_none()) {
              intra = torch::stable::processgroup_from_pyobject(intra_node_group->ptr());
            }
            new (self) CommOverlapHelper(std::move(world), std::move(intra));
          },
          nb::arg("world_group"), nb::arg("intra_node_group") = nb::none());

  // TODO(stable-abi): nanobind manages shared_ptr automatically (no holder template
  // arg) and supports only a single bound base class, so the CommOverlapCore base is
  // dropped here relative to the pybind11 registration.
  nb::class_<CommOverlap, transformer_engine::CommOverlapBase>(m, "CommOverlap")
      .def(nb::new_([](const std::vector<size_t> &buffer_shape,
                       torch::headeronly::ScalarType buffer_dtype,
                       CommOverlapHelper *helper, int tp_size, bool use_cublasmp,
                       transformer_engine::CommOverlapType comm_type, int num_splits,
                       int num_max_streams, int comm_cga_size, int gemm_priority, int comm_priority,
                       int num_comm_sm, bool set_sm_margin, bool atomic_gemm,
                       bool rs_overlap_first_gemm) {
             if (use_cublasmp) {
               return std::make_shared<CommOverlap>(helper, helper->mylocal, tp_size, comm_type,
                                                    buffer_shape, buffer_dtype, num_comm_sm,
                                                    atomic_gemm);
             }
             return std::make_shared<CommOverlap>(
                 buffer_shape, buffer_dtype, helper, tp_size, num_splits, num_max_streams,
                 comm_cga_size, gemm_priority, comm_priority, num_comm_sm, set_sm_margin,
                 atomic_gemm, rs_overlap_first_gemm);
           }),
           nb::call_guard<nb::gil_scoped_release>(), nb::arg("buffer_shape"),
           nb::arg("buffer_dtype"), nb::arg("helper"), nb::arg("tp_size"),
           nb::arg("use_cublasmp") = false,
           nb::arg("comm_type") = transformer_engine::CommOverlapType::RS,
           nb::arg("num_splits") = 4, nb::arg("num_max_streams") = NVTE_COMM_OVERLAP_MAX_STREAMS,
           nb::arg("comm_cga_size") = 2, nb::arg("gemm_priority") = 0, nb::arg("comm_priority") = 0,
           nb::arg("num_comm_sm") = 16, nb::arg("set_sm_margin") = true,
           nb::arg("atomic_gemm") = false, nb::arg("rs_overlap_first_gemm") = false)
      .def("copy_into_buffer",
           static_cast<void (CommOverlap::*)(const torch::stable::Tensor &, bool)>(
               &CommOverlap::copy_into_buffer),
           nb::arg("input"), nb::arg("local_chunk") = false)
      .def("get_buffer", &CommOverlap::get_buffer, nb::arg("local_chunk") = false,
           nb::arg("shape") = std::nullopt)
      .def("get_communication_stream", [](CommOverlap &self) {
        auto streams = self.get_communication_stream();
        auto external_stream = nb::module_::import_("torch.cuda").attr("ExternalStream");
        return nb::make_tuple(
            external_stream(reinterpret_cast<uintptr_t>(streams.first.nativeHandle())),
            external_stream(reinterpret_cast<uintptr_t>(streams.second.nativeHandle())));
      });

  // TODO(stable-abi): see CommOverlap note above (shared_ptr holder + single base).
  nb::class_<CommOverlapP2P, transformer_engine::CommOverlapP2PBase>(m, "CommOverlapP2P")
      .def(nb::new_([](const std::vector<size_t> &buffer_shape,
                       torch::headeronly::ScalarType buffer_dtype,
                       CommOverlapHelper *helper, int tp_size,
                       transformer_engine::CommOverlapType comm_type, int num_max_streams,
                       int comm_cga_size, int gemm_priority, int comm_priority, int num_comm_sm,
                       bool set_sm_margin, bool atomic_gemm, bool use_ce, bool aggregate,
                       bool use_cublasmp) {
             if (use_cublasmp) {
               return std::make_shared<CommOverlapP2P>(helper, helper->mylocal, tp_size, comm_type,
                                                       buffer_shape, buffer_dtype, num_comm_sm,
                                                       atomic_gemm);
             }
             return std::make_shared<CommOverlapP2P>(buffer_shape, buffer_dtype, helper, tp_size,
                                                     comm_type, num_max_streams, comm_cga_size,
                                                     gemm_priority, comm_priority, num_comm_sm,
                                                     set_sm_margin, atomic_gemm, use_ce, aggregate);
           }),
           nb::call_guard<nb::gil_scoped_release>(), nb::arg("buffer_shape"),
           nb::arg("buffer_dtype"), nb::arg("helper"), nb::arg("tp_size"), nb::arg("comm_type"),
           nb::arg("num_max_streams") = NVTE_COMM_OVERLAP_MAX_STREAMS, nb::arg("comm_cga_size") = 1,
           nb::arg("gemm_priority") = 0, nb::arg("comm_priority") = 0, nb::arg("num_comm_sm") = 1,
           nb::arg("set_sm_margin") = false, nb::arg("atomic_gemm") = false,
           nb::arg("use_ce") = true, nb::arg("aggregate") = false, nb::arg("use_cublasmp") = false)
      .def("copy_into_buffer",
           static_cast<void (CommOverlapP2P::*)(const torch::stable::Tensor &, bool)>(
               &CommOverlapP2P::copy_into_buffer),
           nb::arg("input"), nb::arg("local_chunk") = false)
      .def("get_buffer", &CommOverlapP2P::get_buffer, nb::arg("local_chunk") = false,
           nb::arg("shape") = std::nullopt)
      .def("get_communication_stream", [](CommOverlapP2P &self) {
        auto streams = self.get_communication_stream();
        auto external_stream = nb::module_::import_("torch.cuda").attr("ExternalStream");
        return nb::make_tuple(
            external_stream(reinterpret_cast<uintptr_t>(streams.first.nativeHandle())),
            external_stream(reinterpret_cast<uintptr_t>(streams.second.nativeHandle())));
      });
}  // NOLINT(readability/fn_size)
