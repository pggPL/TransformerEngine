/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_

#include <nccl.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "common.h"

class CommOverlapHelper;
class CommOverlap;
class CommOverlapP2P;

namespace transformer_engine::pytorch {

/***************************************************************************************************
 * Router fusion
 **************************************************************************************************/

std::tuple<Tensor, Tensor, Tensor> fused_topk_with_score_function_fwd(
    Tensor logits, int topk, bool use_pre_softmax, std::optional<int> num_groups,
    std::optional<int> group_topk, std::optional<float> scaling_factor, std::string score_function,
    std::optional<Tensor> expert_bias,
    int routing_map_format = static_cast<int>(NVTE_ROUTING_MAP_FORMAT_BYTEMAP));

void fused_topk_with_score_function_bwd(
    Tensor routing_map, Tensor intermediate_output, Tensor grad_probs,
    Tensor grad_logits, int topk, bool use_pre_softmax, std::optional<float> scaling_factor,
    std::string score_function,
    int routing_map_format = static_cast<int>(NVTE_ROUTING_MAP_FORMAT_BYTEMAP));

std::tuple<Tensor, Tensor, Tensor> fused_score_for_moe_aux_loss_fwd(
    Tensor logits, int topk, std::string score_function,
    int routing_map_format = static_cast<int>(NVTE_ROUTING_MAP_FORMAT_BYTEMAP));

void fused_score_for_moe_aux_loss_bwd(Tensor intermediate_output, Tensor grad_scores,
                                      Tensor grad_logits, int topk, std::string score_function);

std::tuple<Tensor, Tensor> fused_moe_aux_loss_fwd(Tensor probs,
                                                          Tensor tokens_per_expert,
                                                          int total_num_tokens, int num_experts,
                                                          int num_rows, int num_cols, int topk,
                                                          float coeff);

Tensor fused_moe_aux_loss_bwd(Tensor Const_buf, Tensor tokens_per_expert, int num_rows,
                                  int num_cols, Tensor grad_aux_loss);

/***************************************************************************************************
 * Permutation
 **************************************************************************************************/

std::tuple<Tensor, Tensor, std::vector<Tensor>> moe_permute_fwd(
    Tensor input, const DType dtype, Tensor indices, int64_t num_out_tokens,
    std::vector<Tensor> workspace, int64_t max_expanded_token_num);

Tensor moe_permute_bwd(Tensor input, const DType dtype, Tensor row_id_map,
                           Tensor prob, int64_t num_tokens, int64_t topK);

Tensor moe_unpermute_fwd(Tensor input, const DType dtype, Tensor row_id_map,
                             Tensor prob, int64_t num_tokens, int64_t topK);

std::tuple<Tensor, Tensor> moe_unpermute_bwd(Tensor input_bwd, Tensor input_fwd,
                                                     const DType dtype, Tensor row_id_map,
                                                     Tensor prob);

/***************************************************************************************************
 * Attention
 **************************************************************************************************/

NVTE_Fused_Attn_Backend get_fused_attn_backend(
    bool is_training, const DType q_dtype, const DType kv_dtype, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
    float p_dropout, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, int64_t window_size_left,
    int64_t window_size_right, bool return_max_logit, bool cuda_graph, bool deterministic);

std::vector<py::object> fused_attn_fwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training, float attn_scale, float p_dropout,
    bool set_zero, NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format,
    NVTE_QKV_Format qkv_scale_inv_format, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    NVTE_Softmax_Type softmax_type, const std::vector<int64_t> window_size,
    bool bottom_right_diagonal, const Tensor cu_seqlens_q, const Tensor cu_seqlens_kv,
    const py::handle Q, const py::handle K, const py::handle V, const ScalarType fake_dtype,
    const std::optional<Tensor> cu_seqlens_q_padded,
    const std::optional<Tensor> cu_seqlens_kv_padded,
    const std::optional<Tensor> page_table_k, const std::optional<Tensor> page_table_v,
    py::handle s_quantizer, py::handle o_quantizer, const std::optional<Tensor> Bias,
    const std::optional<Tensor> SoftmaxOffset, const std::optional<Generator> rng_gen,
    size_t rng_elts_per_thread, bool return_max_logit, bool cuda_graph);

std::vector<py::object> fused_attn_bwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format, NVTE_QKV_Format do_format,
    NVTE_QKV_Layout dqkv_layout, NVTE_QKV_Format qkv_scale_inv_format,
    NVTE_QKV_Format do_scale_inv_format, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    NVTE_Softmax_Type softmax_type, const std::vector<int64_t> window_size,
    bool bottom_right_diagonal, bool deterministic, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_kv, const py::handle Q, const py::handle K, const py::handle V,
    const py::handle O, const py::handle dO, const ScalarType fake_dtype,
    const std::vector<Tensor> Aux_CTX_Tensors,
    const std::optional<Tensor> cu_seqlens_q_padded,
    const std::optional<Tensor> cu_seqlens_kv_padded, py::handle s_quantizer,
    py::handle dp_quantizer, py::handle dqkv_quantizer, bool cuda_graph);

Tensor fa_prepare_fwd(Tensor qkvi);
Tensor fa_prepare_bwd(Tensor q, Tensor k, Tensor v);

std::vector<std::optional<Tensor>> multi_tensor_transpose_to_bhsd(
    std::vector<std::optional<Tensor>> inputs, const std::string &original_format,
    std::vector<std::optional<Tensor>> outputs = {});

std::vector<Tensor> multi_tensor_pad_last_dim(std::vector<Tensor> inputs,
                                                  int64_t alignment);

Tensor convert_thd_to_bshd(Tensor tensor, Tensor cu_seqlens, int b, int max_seq_len);
Tensor convert_bshd_to_thd(Tensor tensor, Tensor cu_seqlens, int t);
void copy_to_kv_cache(Tensor new_k, Tensor new_v, Tensor k_cache, Tensor v_cache,
                      Tensor page_table, Tensor cu_new_lens, Tensor cu_cached_lens,
                      NVTE_QKV_Format kv_format, int b, int max_ctx_len, int max_seq_len,
                      int max_pages_per_seq, bool is_non_paged);

/***************************************************************************************************
 * GEMM
 **************************************************************************************************/

using MaybeTensor = std::optional<Tensor>;

std::vector<py::object> gemm(py::handle A, bool transa, py::handle B, bool transb, py::object D,
                             py::handle quantizer, std::optional<DType> out_dtype, MaybeTensor bias,
                             DType bias_type, bool gelu, MaybeTensor gelu_in, bool grad,
                             Tensor workspace, size_t workspaceSize, bool accumulate,
                             bool use_split_accumulator, CommOverlapCore *comm_overlap = nullptr,
                             std::optional<CommOverlapType> comm_type = std::nullopt,
                             MaybeTensor extra_output = std::nullopt, bool bulk_overlap = false,
                             float alpha = 1.0f, std::optional<float> beta = std::nullopt);

void te_atomic_gemm(Tensor A, Tensor A_scale_inverse, DType A_type,
                    std::vector<int64_t> A_scaling_mode, bool transa, Tensor B,
                    Tensor B_scale_inverse, DType B_type, std::vector<int64_t> B_scaling_mode,
                    bool transb, Tensor D, Tensor D_scale, DType D_type, Tensor D_amax,
                    Tensor bias, DType bias_type, Tensor pre_gelu_out, bool grad,
                    Tensor workspace, size_t workspaceSize, bool accumulate,
                    bool use_split_accumulator, int math_sm_count, int m_split, int n_split,
                    bool gemm_producer, Tensor counter);

std::optional<std::vector<Tensor>> te_general_grouped_gemm(
    std::vector<py::handle> A, bool transa, std::vector<py::handle> B, bool transb,
    std::optional<std::vector<Tensor>> D, DType D_type, std::vector<int64_t> m_splits,
    std::vector<Tensor> bias, DType bias_type, bool single_output,
    std::vector<Tensor> pre_gelu_out, bool grad, std::vector<Tensor> workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, int math_sm_count);

py::object te_general_grouped_gemm_for_grouped_tensor(
    py::handle A, bool transa, py::handle B, bool transb, py::handle D, py::object bias,
    std::optional<Tensor> bias_scale, Tensor alpha, Tensor beta,
    Tensor workspace_setup, Tensor workspace_cublas, bool use_split_accumulator,
    int math_sm_count);

py::object te_general_grouped_gemm_for_discrete_in(py::handle A, bool transa, py::handle B,
                                                   bool transb, py::handle D, py::object bias,
                                                   std::optional<Tensor> bias_scale,
                                                   Tensor alpha, Tensor beta,
                                                   Tensor workspace_setup,
                                                   Tensor workspace_cublas,
                                                   bool use_split_accumulator, int math_sm_count);

py::object te_general_grouped_gemm_for_discrete_out(py::handle A, bool transa, py::handle B,
                                                    bool transb, py::handle D, py::object bias,
                                                    std::optional<Tensor> bias_scale,
                                                    Tensor alpha, Tensor beta,
                                                    Tensor workspace_setup,
                                                    Tensor workspace_cublas,
                                                    bool use_split_accumulator, int math_sm_count);

/***************************************************************************************************
 * Transpose
 **************************************************************************************************/

Tensor fp8_transpose(Tensor input, DType otype,
                         std::optional<Tensor> output = std::nullopt);

Tensor nvfp4_data_transpose(Tensor input, std::optional<Tensor> output = std::nullopt);

void nvfp4_2d_scale_transpose(Tensor input, Tensor output, int64_t M_tiles,
                              int64_t K_tiles);

void nvfp4_2d_multi_tensor_transpose(std::vector<Tensor> rowwise_data_list,
                                     std::vector<Tensor> columnwise_data_list,
                                     std::vector<Tensor> rowwise_scale_inv_list,
                                     std::vector<Tensor> columnwise_scale_inv_list,
                                     std::vector<int64_t> M_list, std::vector<int64_t> K_list);

void nvfp4_multi_tensor_compute_partial_amax(
    std::vector<Tensor> master_weight_list, std::vector<Tensor> partial_amax_list,
    std::vector<Tensor> global_amax_list, std::vector<int64_t> h_list,
    std::vector<int64_t> w_list, std::vector<int64_t> start_offset_list, int64_t block_len);

void nvfp4_expand_scale_to_fp8(Tensor input, Tensor output, int64_t tile_rows,
                               int64_t tile_cols, int64_t rows_padded, int64_t block_len);

void nvfp4_compute_per_block_scale(Tensor block_amax, Tensor scale, Tensor global_amax);

void nvfp4_fused_scale(Tensor block_amax, Tensor global_amax, Tensor per_block_scale,
                       Tensor target_scale, Tensor target_amax, int64_t tile_rows,
                       int64_t tile_cols, int64_t rows_padded, int64_t block_len);

void nvfp4_multi_tensor_fused_scale(
    std::vector<Tensor> block_amax_list, std::vector<Tensor> global_amax_list,
    std::vector<Tensor> per_block_scale_list, std::vector<Tensor> target_scale_list,
    std::vector<Tensor> target_amax_list, std::vector<int64_t> tile_rows_list,
    std::vector<int64_t> tile_cols_list, std::vector<int64_t> rows_padded_list, int64_t block_len);

void nvfp4_compute_global_scale(Tensor global_amax, Tensor global_scale);

Tensor swap_first_dims(Tensor tensor, std::optional<Tensor> out = std::nullopt);

/***************************************************************************************************
 * Activations
 **************************************************************************************************/

/* GLU (sigmoid gate) */
py::object glu(const Tensor &input, py::handle quantizer);

py::object dglu(const Tensor &grad, const Tensor &input, py::handle quantizer);

/* GELU and variants*/
py::object gelu(const Tensor &input, py::handle quantizer);

py::object dgelu(const Tensor &grad, const Tensor &input, py::handle quantizer);

py::object geglu(const Tensor &input, py::handle quantizer);

py::object dgeglu(const Tensor &grad, const Tensor &input, py::handle quantizer);

py::object qgelu(const Tensor &input, py::handle quantizer);

py::object dqgelu(const Tensor &grad, const Tensor &input, py::handle quantizer);

py::object qgeglu(const Tensor &input, py::handle quantizer);

py::object dqgeglu(const Tensor &grad, const Tensor &input, py::handle quantizer);

/* ReLU and variants*/
py::object relu(const Tensor &input, py::handle quantizer);

py::object drelu(const Tensor &grad, const Tensor &input, py::handle quantizer);

py::object reglu(const Tensor &input, py::handle quantizer);

py::object dreglu(const Tensor &grad, const Tensor &input, py::handle quantizer);

py::object srelu(const Tensor &input, py::handle quantizer);

py::object dsrelu(const Tensor &grad, const Tensor &input, py::handle quantizer);

py::object sreglu(const Tensor &input, py::handle quantizer);

py::object dsreglu(const Tensor &grad, const Tensor &input, py::handle quantizer);

/* Silu and variants*/
py::object silu(const Tensor &input, py::handle quantizer);

py::object dsilu(const Tensor &grad, const Tensor &input, py::handle quantizer);

py::object swiglu(const Tensor &input, py::handle quantizer);

py::object dswiglu(const Tensor &grad, const Tensor &input, py::handle quantizer);

py::object clamped_swiglu(const Tensor &input, py::handle quantizer, float limit, float alpha,
                          float glu_linear_offset);

py::object clamped_dswiglu(const Tensor &grad, const Tensor &input, py::handle quantizer,
                           float limit, float alpha, float glu_linear_offset);
/***************************************************************************************************
 * LayerNorm
 **************************************************************************************************/

std::vector<py::object> layernorm_bwd(const Tensor &dz, const Tensor &x,
                                      const Tensor &mu, const Tensor &rsigma,
                                      const Tensor &gamma, const int sm_margin,
                                      const bool zero_centered_gamma);

std::vector<py::object> layernorm_fwd(py::handle input, py::handle weight, MaybeTensor bias,
                                      float eps, py::object ln_out, py::handle quantizer,
                                      DType out_dtype, const int sm_margin,
                                      const bool zero_centered_gamma);

/***************************************************************************************************
 * RMSNorm
 **************************************************************************************************/

std::vector<py::object> rmsnorm_bwd(const Tensor &dz, const Tensor &x,
                                    const Tensor &rsigma, const Tensor &gamma,
                                    const int sm_margin, const bool zero_centered_gamma);

std::vector<py::object> rmsnorm_bwd_add(const Tensor &dz, const Tensor &x,
                                        const Tensor &add, const Tensor &rsigma,
                                        const Tensor &gamma, const int sm_margin,
                                        const bool zero_centered_gamma);

std::vector<py::object> rmsnorm_fwd(const py::handle &input, const py::handle &weight, float eps,
                                    py::object ln_out, py::handle quantizer, DType otype,
                                    const int sm_margin, const bool zero_centered_gamma);

/***************************************************************************************************
 * Memory allocation
 **************************************************************************************************/

// Allocates tensors all backed by a single contiguous buffer.
std::vector<Tensor> bulk_allocate(const std::vector<std::vector<size_t>> &shapes,
                                      const std::vector<ScalarType> &dtypes,
                                      std::optional<Device> device = std::nullopt,
                                      std::optional<std::vector<size_t>> alignments = std::nullopt);

/***************************************************************************************************
 * Quantize
 **************************************************************************************************/

py::object create_empty_quantized_tensor(py::handle quantizer, const std::vector<size_t> &shape,
                                         ScalarType dtype, Device device, bool pin_memory);

py::object quantize(const Tensor &tensor, py::handle quantizer, const py::object &output,
                    std::optional<Tensor> noop_flag);

py::object nvfp4_quantize_with_amax(const Tensor &tensor, py::handle quantizer,
                                    const Tensor &rowwise_amax,
                                    const Tensor &columnwise_amax);

py::object dequantize(const py::handle &input, DType otype);

py::object group_quantize(const Tensor &tensor, py::handle quantizer, const size_t num_tensors,
                          std::optional<Tensor> first_dims,
                          std::optional<Tensor> tensor_offsets);

py::object nvfp4_group_quantize_with_amax(const Tensor &tensor, py::handle quantizer,
                                          const size_t num_tensors,
                                          std::optional<Tensor> first_dims,
                                          const Tensor &rowwise_amax,
                                          const Tensor &columnwise_amax,
                                          std::optional<Tensor> tensor_offsets);

py::object group_dequantize(const py::handle &input, DType otype);

py::object bgrad_group_quantize(const Tensor &tensor, py::handle quantizer,
                                const size_t num_tensors, std::optional<Tensor> first_dims,
                                std::optional<Tensor> tensor_offsets);

std::vector<py::object> multi_tensor_quantize(const std::vector<Tensor> &tensor_list,
                                              std::vector<py::handle> quantizer_list);

std::vector<py::object> split_quantize(const Tensor &tensor,
                                       const std::vector<size_t> &split_sections,
                                       std::vector<py::handle> quantizer_list,
                                       bool disable_bulk_allocation = false);

/***************************************************************************************************
 * Bias gradient fusions
 **************************************************************************************************/

std::vector<py::object> bgrad_quantize(const Tensor &input, py::handle py_quantizer);

std::vector<py::object> dbias_dgelu(const Tensor &grad_output, const Tensor &act_input,
                                    py::handle quantizer);

std::vector<py::object> dbias_dsilu(const Tensor &grad_output, const Tensor &act_input,
                                    py::handle quantizer);

std::vector<py::object> dbias_drelu(const Tensor &grad_output, const Tensor &act_input,
                                    py::handle quantizer);

std::vector<py::object> dbias_dqgelu(const Tensor &grad_output, const Tensor &act_input,
                                     py::handle quantizer);

std::vector<py::object> dbias_dsrelu(const Tensor &grad_output, const Tensor &act_input,
                                     py::handle quantizer);

/***************************************************************************************************
 * Dropout
 **************************************************************************************************/

std::vector<py::object> dropout_fwd(const py::handle &input, const float dropout_probability,
                                    std::optional<Tensor> out = std::nullopt);

py::object dropout_bwd(const Tensor &grad_output, const Tensor &mask,
                       const float dropout_probability,
                       std::optional<Tensor> grad_input = std::nullopt);

/***************************************************************************************************
 * Softmax
 **************************************************************************************************/

Tensor scaled_softmax_forward(Tensor input, float scale_factor);

Tensor scaled_softmax_backward(Tensor output_grad_, Tensor softmax_results_,
                                   float scale_factor);

Tensor scaled_masked_softmax_forward(Tensor input, Tensor mask, float scale_factor);

Tensor scaled_masked_softmax_backward(Tensor output_grad_, Tensor softmax_results_,
                                          float scale_factor);

Tensor scaled_upper_triang_masked_softmax_forward(Tensor input, float scale_factor);

Tensor scaled_upper_triang_masked_softmax_backward(Tensor output_grads_,
                                                       Tensor softmax_results_,
                                                       float scale_factor);

Tensor scaled_aligned_causal_masked_softmax_forward(Tensor input, float scale_factor);

Tensor scaled_aligned_causal_masked_softmax_backward(Tensor output_grads_,
                                                         Tensor softmax_results_,
                                                         float scale_factor);

/***************************************************************************************************
 * FP8 recipe
 **************************************************************************************************/

void compute_amax(const Tensor &tensor, Tensor &amax);

void fused_amax_and_scale_update_after_reduction(const Tensor &amax_reduction_buffer,
                                                 std::vector<Tensor> amax_histories,
                                                 std::vector<Tensor> scales,
                                                 const std::string &amax_compute_algo,
                                                 DType fp8_dtype, float margin);

// Note that the start_offset is the logical offset along the tensor dimension.
// The offset in bytes is start_offset * sizeof(tensor.dtype)
void fp8_block_scaling_compute_partial_amax(const Tensor &tensor, Tensor amax, size_t h,
                                            size_t w, size_t start_offset, size_t block_len);

void fp8_block_scaling_partial_cast(const Tensor &inp, Tensor out, const Tensor &scale,
                                    size_t h, size_t w, size_t start_offset, size_t block_len,
                                    const DType out_dtype);

void nvfp4_2d_compute_partial_amax(const Tensor &tensor, Tensor amax, size_t h, size_t w,
                                   size_t start_offset, size_t block_len);

void nvfp4_2d_partial_cast(const Tensor &inp, py::handle out, const Tensor &scale,
                           const Tensor &global_scale, size_t h, size_t w, size_t start_offset,
                           size_t block_len);

void nvfp4_multi_tensor_2d_partial_cast(std::vector<Tensor> inp_list,
                                        std::vector<Tensor> out_list,
                                        std::vector<Tensor> scale_list,
                                        std::vector<Tensor> global_scale_list,
                                        std::vector<int64_t> h_list, std::vector<int64_t> w_list,
                                        std::vector<int64_t> start_offset_list, int64_t block_len);
void mxfp8_scaling_compute_partial_amax(const Tensor &input, Tensor amax_rowwise,
                                        Tensor amax_colwise, int rows, int cols,
                                        size_t start_offset);

void mxfp8_scaling_partial_cast(const Tensor &input, Tensor output_rowwise,
                                Tensor output_colwise, const Tensor &scale_inv_rowwise,
                                const Tensor &scale_inv_colwise, int rows, int cols,
                                size_t start_offset);

/***************************************************************************************************
 * Rotary positional embedding
 **************************************************************************************************/

Tensor fused_rope_forward(const Tensor &input, const Tensor &freqs,
                              const std::optional<Tensor> start_positions,
                              const NVTE_QKV_Format qkv_format, const bool interleaved,
                              const std::optional<Tensor> cu_seqlens, const int cp_size,
                              const int cp_rank);

Tensor fused_rope_backward(const Tensor &output_grads, const Tensor &freqs,
                               const std::optional<Tensor> start_positions,
                               const NVTE_QKV_Format qkv_format, const bool interleaved,
                               const std::optional<Tensor> cu_seqlens, const int cp_size,
                               const int cp_rank);

std::tuple<Tensor, Tensor, Tensor> fused_qkv_rope_forward(
    const Tensor &qkv_input, const Tensor &q_freqs, const Tensor &k_freqs,
    const std::optional<Tensor> start_positions, const std::vector<int> &qkv_split_arg_list,
    const NVTE_QKV_Format qkv_format, const bool interleaved, const int cp_size, const int cp_rank);

Tensor fused_qkv_rope_backward(const Tensor &q_grad_out, const Tensor &k_grad_out,
                                   const Tensor &v_grad_out, const Tensor &q_freqs,
                                   const Tensor &k_freqs,
                                   const std::vector<int> &qkv_split_arg_list,
                                   const NVTE_QKV_Format qkv_format, const bool interleaved,
                                   const int cp_size, const int cp_rank);

/***************************************************************************************************
 * Miscellaneous
 **************************************************************************************************/

size_t get_cublasLt_version();

size_t get_cudnn_version();

Tensor splits_to_offsets(const Tensor &first_dims, int64_t logical_last_dim);
std::tuple<Tensor, std::vector<Tensor>> splits_to_offsets_multi(
    const Tensor &split_sizes, const Device &device, const std::vector<int64_t> &strides,
    const std::vector<bool> &include_leading_zero, const std::vector<ScalarType> &dtypes,
    bool bulk_allocate_outputs);

Tensor copy_data_ptrs_to_device(const std::vector<Tensor> &tensors,
                                    const Device &device);

/***************************************************************************************************
 * Experimental helpers for the fused grouped MLP
 *
 * These primarily exist to support cuDNN CuTe DSL grouped GEMM
 * kernels. Since those are unstable and under active development,
 * these helpers should also be considered unstable.
 **************************************************************************************************/

namespace grouped_mlp_experimental {

// Prepare discrete weight tensors for the cuDNN CuTe DSL grouped GEMM
// kernel by swizzling scales and copying data and scale pointers to
// device. All tensors must share a uniform shape and `swizzle_type`
// must be one of "mxfp8_rowwise", "mxfp8_columnwise", or "nvfp4".
// Returns {data_ptrs_device, scale_ptrs_device, swizzled_scales_buffer}.
std::tuple<Tensor, Tensor, Tensor> swizzle_scales_and_pack_ptrs_for_discrete_weights(
    const std::vector<Tensor> &data_tensors, const std::vector<Tensor> &scale_tensors,
    const std::string &swizzle_type, const Device &device);

}  // namespace grouped_mlp_experimental

/***************************************************************************************************
 * Support THD format for Context Parallel
 **************************************************************************************************/

Tensor thd_read_half_tensor(const Tensor &tensor, const Tensor &cu_seqlens,
                                int half_idx);

void thd_second_half_lse_correction(Tensor lse, const Tensor &lse_per_step,
                                    const Tensor &cu_seqlens, bool lse_packed);

Tensor thd_read_second_half_lse(const Tensor &lse, const Tensor &cu_seqlens,
                                    bool lse_packed, int second_half_lse_seqlen);

void thd_out_correction(Tensor out, const Tensor &out_per_step, const Tensor &lse,
                        const Tensor &lse_per_step, const Tensor &cu_seqlens,
                        bool only_second_half, bool lse_packed);

void thd_grad_correction(Tensor grad, const Tensor &grad_per_step,
                         const Tensor &cu_seqlens, const std::string &first_half,
                         const std::string &second_half);

Tensor thd_get_partitioned_indices(const Tensor &cu_seqlens, int total_tokens,
                                       int world_size, int rank);

/***************************************************************************************************
 * multi_tensor_* kernels
 **************************************************************************************************/

void multi_tensor_scale_cuda(int chunk_size, Tensor noop_flag,
                             std::vector<std::vector<Tensor>> tensor_lists, float scale);

void multi_tensor_scale_tensor_cuda(int chunk_size, Tensor is_infinite,
                                    std::vector<std::vector<Tensor>> tensor_lists,
                                    Tensor scale);

std::tuple<Tensor, Tensor> multi_tensor_l2norm_cuda(
    int chunk_size, Tensor noop_flag, std::vector<std::vector<Tensor>> tensor_lists,
    std::optional<bool> per_tensor_python);

std::tuple<Tensor, Tensor> multi_tensor_unscale_l2norm_cuda(
    int chunk_size, Tensor noop_flag, std::vector<std::vector<Tensor>> tensor_lists,
    Tensor inv_scale, std::optional<bool> per_tensor_python);

void multi_tensor_adam_cuda(int chunk_size, Tensor noop_flag,
                            std::vector<std::vector<Tensor>> tensor_lists, const float lr,
                            const float beta1, const float beta2, const float epsilon,
                            const int step, const int mode, const int bias_correction,
                            const float weight_decay);

void multi_tensor_adam_param_remainder_cuda(int chunk_size, Tensor noop_flag,
                                            std::vector<std::vector<Tensor>> tensor_lists,
                                            const float lr, const float beta1, const float beta2,
                                            const float epsilon, const int step, const int mode,
                                            const int bias_correction, const float weight_decay);

void multi_tensor_adam_fp8_cuda(int chunk_size, Tensor noop_flag,
                                std::vector<std::vector<Tensor>> tensor_lists, const float lr,
                                const float beta1, const float beta2, const float epsilon,
                                const int step, const int mode, const int bias_correction,
                                const float weight_decay, DType fp8_dtype);

void multi_tensor_adam_capturable_cuda(int chunk_size, Tensor noop_flag,
                                       std::vector<std::vector<Tensor>> tensor_lists,
                                       Tensor lr, const float beta1, const float beta2,
                                       const float epsilon, Tensor step, const int mode,
                                       const int bias_correction, const float weight_decay,
                                       Tensor inv_scale);

void multi_tensor_adam_capturable_master_cuda(int chunk_size, Tensor noop_flag,
                                              std::vector<std::vector<Tensor>> tensor_lists,
                                              Tensor lr, const float beta1, const float beta2,
                                              const float epsilon, Tensor step, const int mode,
                                              const int bias_correction, const float weight_decay,
                                              Tensor inv_scale);

void multi_tensor_sgd_cuda(int chunk_size, Tensor noop_flag,
                           std::vector<std::vector<Tensor>> tensor_lists, float wd,
                           float momentum, float dampening, float lr, bool nesterov, bool first_run,
                           bool wd_after_momentum, float scale);

void multi_tensor_compute_scale_and_scale_inv_cuda(
    int chunk_size, Tensor noop_flag, std::vector<std::vector<Tensor>> tensor_lists,
    float max_fp8, bool force_pow_2_scales, float epsilon);

void multi_tensor_compute_scale_inv_e8m0_cuda(int chunk_size, const py::object &dummy,
                                              std::vector<std::vector<Tensor>> tensor_lists);

/***************************************************************************************************
 * padding
 **************************************************************************************************/

void fused_multi_row_padding(Tensor input, Tensor output,
                             std::vector<size_t> input_row_list,
                             std::vector<size_t> padded_input_row_list);

void fused_multi_row_unpadding(Tensor input, Tensor output,
                               std::vector<size_t> input_row_list,
                               std::vector<size_t> unpadded_input_row_list);

/***************************************************************************************************
 * Scale swizzling for GEMM
 **************************************************************************************************/

void inplace_swizzle_scale_for_gemm(py::handle &tensor);

void inplace_multi_tensor_swizzle_scales_for_gemm(std::vector<py::object> &tensors,
                                                  bool rowwise_usage, bool columnwise_usage);

void inplace_multi_tensor_swizzle_scales_for_gemm_unchecked(std::vector<py::object> &tensors,
                                                            bool rowwise_usage,
                                                            bool columnwise_usage);

void grouped_swizzle_for_gemm(py::handle &tensor, bool rowwise, bool columnwise);

/***************************************************************************************************
 * NVSHMEM APIs
 **************************************************************************************************/

void init_nvshmem_backend(ProcessGroup *process_group);

Tensor create_nvshmem_tensor(const std::vector<int64_t> &shape, ScalarType dtype);

void nvshmem_send_on_current_stream(Tensor src, Tensor dst, int peer, Tensor signal);

void nvshmem_wait_on_current_stream(Tensor signal, const std::string &wait_kind);

void nvshmem_finalize();

/***************************************************************************************************
 * Comm+GEMM Overlap Wrappers
 **************************************************************************************************/

void bulk_overlap_ag_with_external_gemm(CommOverlap &allgather_communicator, Stream send_stream,
                                        Stream recv_stream);

/***************************************************************************************************
 * Newton-Schulz (cuSolverMp)
 **************************************************************************************************/

int64_t cusolvermp_ctx_create(int64_t nccl_comm_ptr, int nranks, int rank);

void cusolvermp_ctx_destroy(int64_t ctx_ptr);

void newton_schulz(int64_t ctx_ptr, int64_t m, int64_t n, Tensor x, int64_t num_iterations,
                   std::vector<float> coefficients);

}  // namespace transformer_engine::pytorch

/***************************************************************************************************
 * Comm+GEMM Overlap Wrappers
 **************************************************************************************************/

class CommOverlapHelper : CustomClassHolder {
 public:
  // Shared ownership of an ncclComm_t. The deleter calls ncclCommDestroy when
  // the last reference (held by the helper and/or any CommOverlap consumers)
  // is released, so the communicator outlives whichever owner is destroyed
  // first.
  using NcclCommSharedPtr = std::shared_ptr<std::remove_pointer<ncclComm_t>::type>;

 private:
  bool initialized{false};
  bool backend_is_nccl{false};
  std::map<std::string, ProcessGroup *> torch_pgs;
  std::map<std::string, NcclCommSharedPtr> nccl_comms;

 public:
  int myrank = -1;
  int numranks = -1;
  int mylocal = -1;
  int numlocal = -1;
  int mynode = -1;
  int numnodes = -1;

  CommOverlapHelper();

  CommOverlapHelper(ProcessGroup *world_group,
                    std::optional<ProcessGroup *> intra_node_group);

  ~CommOverlapHelper();

  void ub_allgather(void *globaldata, size_t globalbytes, void *localdata, size_t localbytes,
                    ExtComm comm);

  void ub_barrier(ExtComm comm);

  NcclCommSharedPtr get_nccl_comm(std::string comm_name);
};

class CommOverlap : CustomClassHolder, public transformer_engine::CommOverlapBase {
 private:
  // Keeps the cuBLASMp NCCL communicator alive for the lifetime of this
  // instance, independent of the CommOverlapHelper that created it.
  CommOverlapHelper::NcclCommSharedPtr _nccl_comm;

 public:
  CommOverlap(const std::vector<size_t> &buffer_shape, ScalarType buffer_dtype,
              CommOverlapHelper *helper, int tp_size, int num_splits = 4,
              int num_max_streams = NVTE_COMM_OVERLAP_MAX_STREAMS, int comm_cga_size = 2,
              int gemm_priority = 0, int comm_priority = 0, int num_comm_sm = 16,
              bool set_sm_margin = true, bool atomic_gemm = false,
              bool rs_overlap_first_gemm = false);

  // cuBLASMp variant. `comm_type`, `buffer_shape`, and `buffer_dtype` size
  // the construction-time warmup matmul that primes cuBLASMp's lazy NCCL
  // window registrations and workspace allocation so subsequent matmuls
  // (including those captured in CUDA graphs) avoid the unsafe lazy paths.
  CommOverlap(CommOverlapHelper *helper, int tp_rank, int tp_size,
              transformer_engine::CommOverlapType comm_type,
              const std::vector<size_t> &buffer_shape, ScalarType buffer_dtype,
              int num_comm_sm = 16, bool atomic_gemm = false);

  ~CommOverlap() {}

  using transformer_engine::CommOverlapCore::copy_into_buffer;
  void copy_into_buffer(const Tensor &input, bool local_chunk = false);

  Tensor get_buffer(bool local_chunk = false,
                        std::optional<std::vector<int64_t>> shape = std::nullopt);

  std::pair<Stream, Stream> get_communication_stream();

};  // CommOverlap

class CommOverlapP2P : CustomClassHolder, public transformer_engine::CommOverlapP2PBase {
 private:
  // Keeps the cuBLASMp NCCL communicator alive for the lifetime of this
  // instance, independent of the CommOverlapHelper that created it.
  CommOverlapHelper::NcclCommSharedPtr _nccl_comm;

 public:
  CommOverlapP2P(const std::vector<size_t> &buffer_shape, ScalarType buffer_dtype,
                 CommOverlapHelper *helper, int tp_size,
                 transformer_engine::CommOverlapType comm_type,
                 int num_max_streams = NVTE_COMM_OVERLAP_MAX_STREAMS, int comm_cga_size = 1,
                 int gemm_priority = 0, int comm_priority = 0, int num_comm_sm = 1,
                 bool set_sm_margin = false, bool atomic_gemm = false, bool use_ce = true,
                 bool aggregate = false);

  // cuBLASMp variant. See CommOverlap for the `comm_type`/buffer args.
  CommOverlapP2P(CommOverlapHelper *helper, int tp_rank, int tp_size,
                 transformer_engine::CommOverlapType comm_type,
                 const std::vector<size_t> &buffer_shape, ScalarType buffer_dtype,
                 int num_comm_sm = 1, bool atomic_gemm = false);

  ~CommOverlapP2P() {}

  using transformer_engine::CommOverlapP2PBase::copy_into_buffer;
  void copy_into_buffer(const Tensor &input, bool local_chunk = false);

  Tensor get_buffer(bool local_chunk = false,
                        std::optional<std::vector<int64_t>> shape = std::nullopt);

  std::pair<Stream, Stream> get_communication_stream();

};  // CommOverlapP2P

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_
