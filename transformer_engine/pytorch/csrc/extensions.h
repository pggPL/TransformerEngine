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

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor> fused_topk_with_score_function_fwd(
    torch::stable::Tensor logits, int topk, bool use_pre_softmax, std::optional<int> num_groups,
    std::optional<int> group_topk, std::optional<float> scaling_factor, std::string score_function,
    std::optional<torch::stable::Tensor> expert_bias,
    int routing_map_format = static_cast<int>(NVTE_ROUTING_MAP_FORMAT_BYTEMAP),
    std::optional<torch::stable::Tensor> topk_indices = std::nullopt);

void fused_topk_with_score_function_bwd(
    torch::stable::Tensor routing_map, torch::stable::Tensor intermediate_output, torch::stable::Tensor grad_probs,
    torch::stable::Tensor grad_logits, int topk, bool use_pre_softmax, std::optional<float> scaling_factor,
    std::string score_function, bool use_dense_indices = false,
    int routing_map_format = static_cast<int>(NVTE_ROUTING_MAP_FORMAT_BYTEMAP));

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor> fused_score_for_moe_aux_loss_fwd(
    torch::stable::Tensor logits, int topk, std::string score_function,
    int routing_map_format = static_cast<int>(NVTE_ROUTING_MAP_FORMAT_BYTEMAP));

void fused_score_for_moe_aux_loss_bwd(torch::stable::Tensor intermediate_output, torch::stable::Tensor grad_scores,
                                      torch::stable::Tensor grad_logits, int topk, std::string score_function);

std::tuple<torch::stable::Tensor, torch::stable::Tensor> fused_moe_aux_loss_fwd(torch::stable::Tensor probs,
                                                          torch::stable::Tensor tokens_per_expert,
                                                          int total_num_tokens, int num_experts,
                                                          int num_rows, int num_cols, int topk,
                                                          float coeff);

torch::stable::Tensor fused_moe_aux_loss_bwd(torch::stable::Tensor Const_buf, torch::stable::Tensor tokens_per_expert, int num_rows,
                                  int num_cols, torch::stable::Tensor grad_aux_loss);

/***************************************************************************************************
 * Permutation
 **************************************************************************************************/

std::tuple<torch::stable::Tensor, torch::stable::Tensor, std::vector<torch::stable::Tensor>> moe_permute_fwd(
    torch::stable::Tensor input, const DType dtype, torch::stable::Tensor indices, int64_t num_out_tokens,
    std::vector<torch::stable::Tensor> workspace, int64_t max_expanded_token_num);

torch::stable::Tensor moe_permute_bwd(torch::stable::Tensor input, const DType dtype, torch::stable::Tensor row_id_map,
                           torch::stable::Tensor prob, int64_t num_tokens, int64_t topK);

torch::stable::Tensor moe_unpermute_fwd(torch::stable::Tensor input, const DType dtype, torch::stable::Tensor row_id_map,
                             torch::stable::Tensor prob, int64_t num_tokens, int64_t topK);

std::tuple<torch::stable::Tensor, torch::stable::Tensor> moe_unpermute_bwd(torch::stable::Tensor input_bwd, torch::stable::Tensor input_fwd,
                                                     const DType dtype, torch::stable::Tensor row_id_map,
                                                     torch::stable::Tensor prob);

/***************************************************************************************************
 * Attention
 **************************************************************************************************/

NVTE_Fused_Attn_Backend get_fused_attn_backend(
    bool is_training, const DType q_dtype, const DType kv_dtype, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
    float p_dropout, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, int64_t window_size_left,
    int64_t window_size_right, bool return_max_logit, bool cuda_graph, bool deterministic);

std::vector<nb::object> fused_attn_fwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training, float attn_scale, float p_dropout,
    bool set_zero, NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format,
    NVTE_QKV_Format qkv_scale_inv_format, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    NVTE_Softmax_Type softmax_type, const std::vector<int64_t> window_size,
    bool bottom_right_diagonal, const torch::stable::Tensor cu_seqlens_q, const torch::stable::Tensor cu_seqlens_kv,
    const nb::handle Q, const nb::handle K, const nb::handle V, const torch::headeronly::ScalarType fake_dtype,
    const std::optional<torch::stable::Tensor> cu_seqlens_q_padded,
    const std::optional<torch::stable::Tensor> cu_seqlens_kv_padded,
    const std::optional<torch::stable::Tensor> page_table_k, const std::optional<torch::stable::Tensor> page_table_v,
    nb::handle s_quantizer, nb::handle o_quantizer, const std::optional<torch::stable::Tensor> Bias,
    const std::optional<torch::stable::Tensor> SoftmaxOffset,
    // TODO(stable-abi): at::Generator (RNG) has no stable-ABI equivalent.
    const std::optional<at::Generator> rng_gen, size_t rng_elts_per_thread, bool return_max_logit,
    bool cuda_graph);

std::vector<nb::object> fused_attn_bwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format, NVTE_QKV_Format do_format,
    NVTE_QKV_Layout dqkv_layout, NVTE_QKV_Format qkv_scale_inv_format,
    NVTE_QKV_Format do_scale_inv_format, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    NVTE_Softmax_Type softmax_type, const std::vector<int64_t> window_size,
    bool bottom_right_diagonal, bool deterministic, const torch::stable::Tensor cu_seqlens_q,
    const torch::stable::Tensor cu_seqlens_kv, const nb::handle Q, const nb::handle K, const nb::handle V,
    const nb::handle O, const nb::handle dO, const torch::headeronly::ScalarType fake_dtype,
    const std::vector<torch::stable::Tensor> Aux_CTX_Tensors,
    const std::optional<torch::stable::Tensor> cu_seqlens_q_padded,
    const std::optional<torch::stable::Tensor> cu_seqlens_kv_padded, nb::handle s_quantizer,
    nb::handle dp_quantizer, nb::handle dqkv_quantizer, bool cuda_graph);

torch::stable::Tensor fa_prepare_fwd(torch::stable::Tensor qkvi);
torch::stable::Tensor fa_prepare_bwd(torch::stable::Tensor q, torch::stable::Tensor k, torch::stable::Tensor v);

std::vector<std::optional<torch::stable::Tensor>> multi_tensor_transpose_to_bhsd(
    std::vector<std::optional<torch::stable::Tensor>> inputs, const std::string &original_format,
    std::vector<std::optional<torch::stable::Tensor>> outputs = {});

std::vector<torch::stable::Tensor> multi_tensor_pad_last_dim(std::vector<torch::stable::Tensor> inputs,
                                                  int64_t alignment);

torch::stable::Tensor convert_thd_to_bshd(torch::stable::Tensor tensor, torch::stable::Tensor cu_seqlens, int b, int max_seq_len);
torch::stable::Tensor convert_bshd_to_thd(torch::stable::Tensor tensor, torch::stable::Tensor cu_seqlens, int t);
void copy_to_kv_cache(torch::stable::Tensor new_k, torch::stable::Tensor new_v, torch::stable::Tensor k_cache, torch::stable::Tensor v_cache,
                      torch::stable::Tensor page_table, torch::stable::Tensor cu_new_lens, torch::stable::Tensor cu_cached_lens,
                      NVTE_QKV_Format kv_format, int b, int max_ctx_len, int max_seq_len,
                      int max_pages_per_seq, bool is_non_paged);

/***************************************************************************************************
 * GEMM
 **************************************************************************************************/

using MaybeTensor = std::optional<torch::stable::Tensor>;

std::vector<nb::object> gemm(nb::handle A, bool transa, nb::handle B, bool transb, nb::object D,
                             nb::handle quantizer, std::optional<DType> out_dtype, MaybeTensor bias,
                             DType bias_type, bool gelu, MaybeTensor gelu_in, bool grad,
                             torch::stable::Tensor workspace, size_t workspaceSize, bool accumulate,
                             bool use_split_accumulator, CommOverlapCore *comm_overlap = nullptr,
                             std::optional<CommOverlapType> comm_type = std::nullopt,
                             MaybeTensor extra_output = std::nullopt, bool bulk_overlap = false,
                             float alpha = 1.0f, std::optional<float> beta = std::nullopt);

void te_atomic_gemm(torch::stable::Tensor A, torch::stable::Tensor A_scale_inverse, DType A_type,
                    std::vector<int64_t> A_scaling_mode, bool transa, torch::stable::Tensor B,
                    torch::stable::Tensor B_scale_inverse, DType B_type, std::vector<int64_t> B_scaling_mode,
                    bool transb, torch::stable::Tensor D, torch::stable::Tensor D_scale, DType D_type, torch::stable::Tensor D_amax,
                    torch::stable::Tensor bias, DType bias_type, torch::stable::Tensor pre_gelu_out, bool grad,
                    torch::stable::Tensor workspace, size_t workspaceSize, bool accumulate,
                    bool use_split_accumulator, int math_sm_count, int m_split, int n_split,
                    bool gemm_producer, torch::stable::Tensor counter);

std::optional<std::vector<torch::stable::Tensor>> te_general_grouped_gemm(
    std::vector<nb::handle> A, bool transa, std::vector<nb::handle> B, bool transb,
    std::optional<std::vector<torch::stable::Tensor>> D, DType D_type, std::vector<int64_t> m_splits,
    std::vector<torch::stable::Tensor> bias, DType bias_type, bool single_output,
    std::vector<torch::stable::Tensor> pre_gelu_out, bool grad, std::vector<torch::stable::Tensor> workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, int math_sm_count);

nb::object te_general_grouped_gemm_for_grouped_tensor(
    nb::handle A, bool transa, nb::handle B, bool transb, nb::handle D, nb::object bias,
    std::optional<torch::stable::Tensor> bias_scale, torch::stable::Tensor alpha, torch::stable::Tensor beta,
    torch::stable::Tensor workspace_setup, torch::stable::Tensor workspace_cublas, bool use_split_accumulator,
    int math_sm_count);

nb::object te_general_grouped_gemm_for_discrete_in(nb::handle A, bool transa, nb::handle B,
                                                   bool transb, nb::handle D, nb::object bias,
                                                   std::optional<torch::stable::Tensor> bias_scale,
                                                   torch::stable::Tensor alpha, torch::stable::Tensor beta,
                                                   torch::stable::Tensor workspace_setup,
                                                   torch::stable::Tensor workspace_cublas,
                                                   bool use_split_accumulator, int math_sm_count);

nb::object te_general_grouped_gemm_for_discrete_out(nb::handle A, bool transa, nb::handle B,
                                                    bool transb, nb::handle D, nb::object bias,
                                                    std::optional<torch::stable::Tensor> bias_scale,
                                                    torch::stable::Tensor alpha, torch::stable::Tensor beta,
                                                    torch::stable::Tensor workspace_setup,
                                                    torch::stable::Tensor workspace_cublas,
                                                    bool use_split_accumulator, int math_sm_count);

/***************************************************************************************************
 * Transpose
 **************************************************************************************************/

torch::stable::Tensor fp8_transpose(torch::stable::Tensor input, DType otype,
                         std::optional<torch::stable::Tensor> output = std::nullopt);

torch::stable::Tensor nvfp4_data_transpose(torch::stable::Tensor input, std::optional<torch::stable::Tensor> output = std::nullopt);

void nvfp4_2d_scale_transpose(torch::stable::Tensor input, torch::stable::Tensor output, int64_t M_tiles,
                              int64_t K_tiles);

void nvfp4_2d_multi_tensor_transpose(std::vector<torch::stable::Tensor> rowwise_data_list,
                                     std::vector<torch::stable::Tensor> columnwise_data_list,
                                     std::vector<torch::stable::Tensor> rowwise_scale_inv_list,
                                     std::vector<torch::stable::Tensor> columnwise_scale_inv_list,
                                     std::vector<int64_t> M_list, std::vector<int64_t> K_list);

void nvfp4_multi_tensor_compute_partial_amax(
    std::vector<torch::stable::Tensor> master_weight_list, std::vector<torch::stable::Tensor> partial_amax_list,
    std::vector<torch::stable::Tensor> global_amax_list, std::vector<int64_t> h_list,
    std::vector<int64_t> w_list, std::vector<int64_t> start_offset_list, int64_t block_len);

void nvfp4_expand_scale_to_fp8(torch::stable::Tensor input, torch::stable::Tensor output, int64_t tile_rows,
                               int64_t tile_cols, int64_t rows_padded, int64_t block_len);

void nvfp4_compute_per_block_scale(torch::stable::Tensor block_amax, torch::stable::Tensor scale, torch::stable::Tensor global_amax);

void nvfp4_fused_scale(torch::stable::Tensor block_amax, torch::stable::Tensor global_amax, torch::stable::Tensor per_block_scale,
                       torch::stable::Tensor target_scale, torch::stable::Tensor target_amax, int64_t tile_rows,
                       int64_t tile_cols, int64_t rows_padded, int64_t block_len);

void nvfp4_multi_tensor_fused_scale(
    std::vector<torch::stable::Tensor> block_amax_list, std::vector<torch::stable::Tensor> global_amax_list,
    std::vector<torch::stable::Tensor> per_block_scale_list, std::vector<torch::stable::Tensor> target_scale_list,
    std::vector<torch::stable::Tensor> target_amax_list, std::vector<int64_t> tile_rows_list,
    std::vector<int64_t> tile_cols_list, std::vector<int64_t> rows_padded_list, int64_t block_len);

void nvfp4_compute_global_scale(torch::stable::Tensor global_amax, torch::stable::Tensor global_scale);

torch::stable::Tensor swap_first_dims(torch::stable::Tensor tensor, std::optional<torch::stable::Tensor> out = std::nullopt);

/***************************************************************************************************
 * Activations
 **************************************************************************************************/

/* GLU (sigmoid gate) */
nb::object glu(const torch::stable::Tensor &input, nb::handle quantizer);

nb::object dglu(const torch::stable::Tensor &grad, const torch::stable::Tensor &input, nb::handle quantizer);

/* GELU and variants*/
nb::object gelu(const torch::stable::Tensor &input, nb::handle quantizer);

nb::object dgelu(const torch::stable::Tensor &grad, const torch::stable::Tensor &input, nb::handle quantizer);

nb::object geglu(const torch::stable::Tensor &input, nb::handle quantizer);

nb::object dgeglu(const torch::stable::Tensor &grad, const torch::stable::Tensor &input, nb::handle quantizer);

nb::object qgelu(const torch::stable::Tensor &input, nb::handle quantizer);

nb::object dqgelu(const torch::stable::Tensor &grad, const torch::stable::Tensor &input, nb::handle quantizer);

nb::object qgeglu(const torch::stable::Tensor &input, nb::handle quantizer);

nb::object dqgeglu(const torch::stable::Tensor &grad, const torch::stable::Tensor &input, nb::handle quantizer);

/* ReLU and variants*/
nb::object relu(const torch::stable::Tensor &input, nb::handle quantizer);

nb::object drelu(const torch::stable::Tensor &grad, const torch::stable::Tensor &input, nb::handle quantizer);

nb::object reglu(const torch::stable::Tensor &input, nb::handle quantizer);

nb::object dreglu(const torch::stable::Tensor &grad, const torch::stable::Tensor &input, nb::handle quantizer);

nb::object srelu(const torch::stable::Tensor &input, nb::handle quantizer);

nb::object dsrelu(const torch::stable::Tensor &grad, const torch::stable::Tensor &input, nb::handle quantizer);

nb::object sreglu(const torch::stable::Tensor &input, nb::handle quantizer);

nb::object dsreglu(const torch::stable::Tensor &grad, const torch::stable::Tensor &input, nb::handle quantizer);

/* Silu and variants*/
nb::object silu(const torch::stable::Tensor &input, nb::handle quantizer);

nb::object dsilu(const torch::stable::Tensor &grad, const torch::stable::Tensor &input, nb::handle quantizer);

nb::object swiglu(const torch::stable::Tensor &input, nb::handle quantizer);

nb::object dswiglu(const torch::stable::Tensor &grad, const torch::stable::Tensor &input, nb::handle quantizer);

nb::object clamped_swiglu(const torch::stable::Tensor &input, nb::handle quantizer, float limit, float alpha,
                          float glu_linear_offset);

nb::object clamped_dswiglu(const torch::stable::Tensor &grad, const torch::stable::Tensor &input, nb::handle quantizer,
                           float limit, float alpha, float glu_linear_offset);
/***************************************************************************************************
 * LayerNorm
 **************************************************************************************************/

std::vector<nb::object> layernorm_bwd(const torch::stable::Tensor &dz, const torch::stable::Tensor &x,
                                      const torch::stable::Tensor &mu, const torch::stable::Tensor &rsigma,
                                      const torch::stable::Tensor &gamma, const int sm_margin,
                                      const bool zero_centered_gamma);

std::vector<nb::object> layernorm_fwd(nb::handle input, nb::handle weight, MaybeTensor bias,
                                      float eps, nb::object ln_out, nb::handle quantizer,
                                      DType out_dtype, const int sm_margin,
                                      const bool zero_centered_gamma);

/***************************************************************************************************
 * RMSNorm
 **************************************************************************************************/

std::vector<nb::object> rmsnorm_bwd(const torch::stable::Tensor &dz, const torch::stable::Tensor &x,
                                    const torch::stable::Tensor &rsigma, const torch::stable::Tensor &gamma,
                                    const int sm_margin, const bool zero_centered_gamma);

std::vector<nb::object> rmsnorm_bwd_add(const torch::stable::Tensor &dz, const torch::stable::Tensor &x,
                                        const torch::stable::Tensor &add, const torch::stable::Tensor &rsigma,
                                        const torch::stable::Tensor &gamma, const int sm_margin,
                                        const bool zero_centered_gamma);

std::vector<nb::object> rmsnorm_fwd(const nb::handle &input, const nb::handle &weight, float eps,
                                    nb::object ln_out, nb::handle quantizer, DType otype,
                                    const int sm_margin, const bool zero_centered_gamma);

/***************************************************************************************************
 * Memory allocation
 **************************************************************************************************/

// Allocates tensors all backed by a single contiguous buffer.
std::vector<torch::stable::Tensor> bulk_allocate(const std::vector<std::vector<size_t>> &shapes,
                                      const std::vector<torch::headeronly::ScalarType> &dtypes,
                                      std::optional<torch::stable::Device> device = std::nullopt,
                                      std::optional<std::vector<size_t>> alignments = std::nullopt);

/***************************************************************************************************
 * Quantize
 **************************************************************************************************/

nb::object create_empty_quantized_tensor(nb::handle quantizer, const std::vector<size_t> &shape,
                                         torch::headeronly::ScalarType dtype, torch::stable::Device device, bool pin_memory);

nb::object quantize(const torch::stable::Tensor &tensor, nb::handle quantizer, const nb::object &output,
                    std::optional<torch::stable::Tensor> noop_flag);

nb::object nvfp4_quantize_with_amax(const torch::stable::Tensor &tensor, nb::handle quantizer,
                                    const torch::stable::Tensor &rowwise_amax,
                                    const torch::stable::Tensor &columnwise_amax);

nb::object dequantize(const nb::handle &input, DType otype);

nb::object group_quantize(const torch::stable::Tensor &tensor, nb::handle quantizer, const size_t num_tensors,
                          std::optional<torch::stable::Tensor> first_dims, std::optional<torch::stable::Tensor> last_dims,
                          std::optional<torch::stable::Tensor> tensor_offsets,
                          std::optional<torch::stable::Tensor> noop_flag);

nb::object nvfp4_group_quantize_with_amax(const torch::stable::Tensor &tensor, nb::handle quantizer,
                                          const size_t num_tensors,
                                          std::optional<torch::stable::Tensor> first_dims,
                                          std::optional<torch::stable::Tensor> last_dims,
                                          const torch::stable::Tensor &rowwise_amax,
                                          const torch::stable::Tensor &columnwise_amax,
                                          std::optional<torch::stable::Tensor> tensor_offsets);

nb::object group_dequantize(const nb::handle &input, DType otype);

nb::object bgrad_group_quantize(const torch::stable::Tensor &tensor, nb::handle quantizer,
                                const size_t num_tensors, std::optional<torch::stable::Tensor> first_dims,
                                std::optional<torch::stable::Tensor> last_dims,
                                std::optional<torch::stable::Tensor> tensor_offsets);

std::vector<nb::object> multi_tensor_quantize(const std::vector<torch::stable::Tensor> &tensor_list,
                                              std::vector<nb::handle> quantizer_list);

std::vector<nb::object> split_quantize(const torch::stable::Tensor &tensor,
                                       const std::vector<size_t> &split_sections,
                                       std::vector<nb::handle> quantizer_list,
                                       bool disable_bulk_allocation = false);

/***************************************************************************************************
 * Bias gradient fusions
 **************************************************************************************************/

std::vector<nb::object> bgrad_quantize(const torch::stable::Tensor &input, nb::handle py_quantizer);

std::vector<nb::object> dbias_dgelu(const torch::stable::Tensor &grad_output, const torch::stable::Tensor &act_input,
                                    nb::handle quantizer);

std::vector<nb::object> dbias_dsilu(const torch::stable::Tensor &grad_output, const torch::stable::Tensor &act_input,
                                    nb::handle quantizer);

std::vector<nb::object> dbias_drelu(const torch::stable::Tensor &grad_output, const torch::stable::Tensor &act_input,
                                    nb::handle quantizer);

std::vector<nb::object> dbias_dqgelu(const torch::stable::Tensor &grad_output, const torch::stable::Tensor &act_input,
                                     nb::handle quantizer);

std::vector<nb::object> dbias_dsrelu(const torch::stable::Tensor &grad_output, const torch::stable::Tensor &act_input,
                                     nb::handle quantizer);

/***************************************************************************************************
 * Dropout
 **************************************************************************************************/

std::vector<nb::object> dropout_fwd(const nb::handle &input, const float dropout_probability,
                                    std::optional<torch::stable::Tensor> out = std::nullopt);

nb::object dropout_bwd(const torch::stable::Tensor &grad_output, const torch::stable::Tensor &mask,
                       const float dropout_probability,
                       std::optional<torch::stable::Tensor> grad_input = std::nullopt);

/***************************************************************************************************
 * Softmax
 **************************************************************************************************/

torch::stable::Tensor scaled_softmax_forward(torch::stable::Tensor input, float scale_factor);

torch::stable::Tensor scaled_softmax_backward(torch::stable::Tensor output_grad_, torch::stable::Tensor softmax_results_,
                                   float scale_factor);

torch::stable::Tensor scaled_masked_softmax_forward(torch::stable::Tensor input, torch::stable::Tensor mask, float scale_factor);

torch::stable::Tensor scaled_masked_softmax_backward(torch::stable::Tensor output_grad_, torch::stable::Tensor softmax_results_,
                                          float scale_factor);

torch::stable::Tensor scaled_upper_triang_masked_softmax_forward(torch::stable::Tensor input, float scale_factor);

torch::stable::Tensor scaled_upper_triang_masked_softmax_backward(torch::stable::Tensor output_grads_,
                                                       torch::stable::Tensor softmax_results_,
                                                       float scale_factor);

torch::stable::Tensor scaled_aligned_causal_masked_softmax_forward(torch::stable::Tensor input, float scale_factor);

torch::stable::Tensor scaled_aligned_causal_masked_softmax_backward(torch::stable::Tensor output_grads_,
                                                         torch::stable::Tensor softmax_results_,
                                                         float scale_factor);

/***************************************************************************************************
 * FP8 recipe
 **************************************************************************************************/

void compute_amax(const torch::stable::Tensor &tensor, torch::stable::Tensor &amax);

void fused_amax_and_scale_update_after_reduction(const torch::stable::Tensor &amax_reduction_buffer,
                                                 std::vector<torch::stable::Tensor> amax_histories,
                                                 std::vector<torch::stable::Tensor> scales,
                                                 const std::string &amax_compute_algo,
                                                 DType fp8_dtype, float margin);

// Note that the start_offset is the logical offset along the tensor dimension.
// The offset in bytes is start_offset * sizeof(tensor.dtype)
void fp8_block_scaling_compute_partial_amax(const torch::stable::Tensor &tensor, torch::stable::Tensor amax, size_t h,
                                            size_t w, size_t start_offset, size_t block_len);

void fp8_block_scaling_partial_cast(const torch::stable::Tensor &inp, torch::stable::Tensor out, const torch::stable::Tensor &scale,
                                    size_t h, size_t w, size_t start_offset, size_t block_len,
                                    const DType out_dtype);

void nvfp4_2d_compute_partial_amax(const torch::stable::Tensor &tensor, torch::stable::Tensor amax, size_t h, size_t w,
                                   size_t start_offset, size_t block_len);

void nvfp4_2d_partial_cast(const torch::stable::Tensor &inp, nb::handle out, const torch::stable::Tensor &scale,
                           const torch::stable::Tensor &global_scale, size_t h, size_t w, size_t start_offset,
                           size_t block_len);

void nvfp4_multi_tensor_2d_partial_cast(std::vector<torch::stable::Tensor> inp_list,
                                        std::vector<torch::stable::Tensor> out_list,
                                        std::vector<torch::stable::Tensor> scale_list,
                                        std::vector<torch::stable::Tensor> global_scale_list,
                                        std::vector<int64_t> h_list, std::vector<int64_t> w_list,
                                        std::vector<int64_t> start_offset_list, int64_t block_len);
void mxfp8_scaling_compute_partial_amax(const torch::stable::Tensor &input, torch::stable::Tensor amax_rowwise,
                                        torch::stable::Tensor amax_colwise, int rows, int cols,
                                        size_t start_offset);

void mxfp8_scaling_partial_cast(const torch::stable::Tensor &input, torch::stable::Tensor output_rowwise,
                                torch::stable::Tensor output_colwise, const torch::stable::Tensor &scale_inv_rowwise,
                                const torch::stable::Tensor &scale_inv_colwise, int rows, int cols,
                                size_t start_offset);

/***************************************************************************************************
 * Rotary positional embedding
 **************************************************************************************************/

torch::stable::Tensor fused_rope_forward(const torch::stable::Tensor &input, const torch::stable::Tensor &freqs,
                              const std::optional<torch::stable::Tensor> start_positions,
                              const NVTE_QKV_Format qkv_format, const bool interleaved,
                              const std::optional<torch::stable::Tensor> cu_seqlens, const int cp_size,
                              const int cp_rank);

torch::stable::Tensor fused_rope_backward(const torch::stable::Tensor &output_grads, const torch::stable::Tensor &freqs,
                               const std::optional<torch::stable::Tensor> start_positions,
                               const NVTE_QKV_Format qkv_format, const bool interleaved,
                               const std::optional<torch::stable::Tensor> cu_seqlens, const int cp_size,
                               const int cp_rank);

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor> fused_qkv_rope_forward(
    const torch::stable::Tensor &qkv_input, const torch::stable::Tensor &q_freqs, const torch::stable::Tensor &k_freqs,
    const std::optional<torch::stable::Tensor> start_positions, const std::vector<int> &qkv_split_arg_list,
    const NVTE_QKV_Format qkv_format, const bool interleaved, const int cp_size, const int cp_rank);

torch::stable::Tensor fused_qkv_rope_backward(const torch::stable::Tensor &q_grad_out, const torch::stable::Tensor &k_grad_out,
                                   const torch::stable::Tensor &v_grad_out, const torch::stable::Tensor &q_freqs,
                                   const torch::stable::Tensor &k_freqs,
                                   const std::vector<int> &qkv_split_arg_list,
                                   const NVTE_QKV_Format qkv_format, const bool interleaved,
                                   const int cp_size, const int cp_rank);

/***************************************************************************************************
 * Miscellaneous
 **************************************************************************************************/

size_t get_cublasLt_version();

size_t get_cudnn_version();

torch::stable::Tensor splits_to_offsets(const torch::stable::Tensor &first_dims, int64_t logical_last_dim);
std::tuple<torch::stable::Tensor, std::vector<torch::stable::Tensor>> splits_to_offsets_multi(
    const torch::stable::Tensor &split_sizes, const torch::stable::Device &device, const std::vector<int64_t> &strides,
    const std::vector<bool> &include_leading_zero, const std::vector<torch::headeronly::ScalarType> &dtypes,
    bool bulk_allocate_outputs);

torch::stable::Tensor copy_data_ptrs_to_device(const std::vector<torch::stable::Tensor> &tensors,
                                    const torch::stable::Device &device);

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
std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor> swizzle_scales_and_pack_ptrs_for_discrete_weights(
    const std::vector<torch::stable::Tensor> &data_tensors, const std::vector<torch::stable::Tensor> &scale_tensors,
    const std::string &swizzle_type, const torch::stable::Device &device);

}  // namespace grouped_mlp_experimental

/***************************************************************************************************
 * Support THD format for Context Parallel
 **************************************************************************************************/

torch::stable::Tensor thd_read_half_tensor(const torch::stable::Tensor &tensor, const torch::stable::Tensor &cu_seqlens,
                                int half_idx);

void thd_second_half_lse_correction(torch::stable::Tensor lse, const torch::stable::Tensor &lse_per_step,
                                    const torch::stable::Tensor &cu_seqlens, bool lse_packed);

torch::stable::Tensor thd_read_second_half_lse(const torch::stable::Tensor &lse, const torch::stable::Tensor &cu_seqlens,
                                    bool lse_packed, int second_half_lse_seqlen);

void thd_out_correction(torch::stable::Tensor out, const torch::stable::Tensor &out_per_step, const torch::stable::Tensor &lse,
                        const torch::stable::Tensor &lse_per_step, const torch::stable::Tensor &cu_seqlens,
                        bool only_second_half, bool lse_packed);

void thd_grad_correction(torch::stable::Tensor grad, const torch::stable::Tensor &grad_per_step,
                         const torch::stable::Tensor &cu_seqlens, const std::string &first_half,
                         const std::string &second_half);

torch::stable::Tensor thd_get_partitioned_indices(const torch::stable::Tensor &cu_seqlens, int total_tokens,
                                       int world_size, int rank);

torch::stable::Tensor thd_sequence_order_to_cp_rank_order(const torch::stable::Tensor &inp, const torch::stable::Tensor &cu_seqlens,
                                               int cp_size, int total_tokens);

torch::stable::Tensor thd_cp_rank_order_to_sequence_order(const torch::stable::Tensor &inp, const torch::stable::Tensor &cu_seqlens,
                                               int cp_size, int total_tokens);

void thd_copy_valid_tokens_from_per_split_to_rank_local(torch::stable::Tensor out, const torch::stable::Tensor &inp,
                                                        const torch::stable::Tensor &cu_seqlens_padded,
                                                        const torch::stable::Tensor &cu_seqlens);

/***************************************************************************************************
 * multi_tensor_* kernels
 **************************************************************************************************/

void multi_tensor_scale_cuda(int chunk_size, torch::stable::Tensor noop_flag,
                             std::vector<std::vector<torch::stable::Tensor>> tensor_lists, float scale);

void multi_tensor_scale_tensor_cuda(int chunk_size, torch::stable::Tensor is_infinite,
                                    std::vector<std::vector<torch::stable::Tensor>> tensor_lists,
                                    torch::stable::Tensor scale);

std::tuple<torch::stable::Tensor, torch::stable::Tensor> multi_tensor_l2norm_cuda(
    int chunk_size, torch::stable::Tensor noop_flag, std::vector<std::vector<torch::stable::Tensor>> tensor_lists,
    std::optional<bool> per_tensor_python);

std::tuple<torch::stable::Tensor, torch::stable::Tensor> multi_tensor_unscale_l2norm_cuda(
    int chunk_size, torch::stable::Tensor noop_flag, std::vector<std::vector<torch::stable::Tensor>> tensor_lists,
    torch::stable::Tensor inv_scale, std::optional<bool> per_tensor_python);

void multi_tensor_adam_cuda(int chunk_size, torch::stable::Tensor noop_flag,
                            std::vector<std::vector<torch::stable::Tensor>> tensor_lists, const float lr,
                            const float beta1, const float beta2, const float epsilon,
                            const int step, const int mode, const int bias_correction,
                            const float weight_decay);

void multi_tensor_adam_param_remainder_cuda(int chunk_size, torch::stable::Tensor noop_flag,
                                            std::vector<std::vector<torch::stable::Tensor>> tensor_lists,
                                            const float lr, const float beta1, const float beta2,
                                            const float epsilon, const int step, const int mode,
                                            const int bias_correction, const float weight_decay);

void multi_tensor_adam_fp8_cuda(int chunk_size, torch::stable::Tensor noop_flag,
                                std::vector<std::vector<torch::stable::Tensor>> tensor_lists, const float lr,
                                const float beta1, const float beta2, const float epsilon,
                                const int step, const int mode, const int bias_correction,
                                const float weight_decay, DType fp8_dtype);

void multi_tensor_adam_capturable_cuda(int chunk_size, torch::stable::Tensor noop_flag,
                                       std::vector<std::vector<torch::stable::Tensor>> tensor_lists,
                                       torch::stable::Tensor lr, const float beta1, const float beta2,
                                       const float epsilon, torch::stable::Tensor step, const int mode,
                                       const int bias_correction, const float weight_decay,
                                       torch::stable::Tensor inv_scale);

void multi_tensor_adam_capturable_master_cuda(int chunk_size, torch::stable::Tensor noop_flag,
                                              std::vector<std::vector<torch::stable::Tensor>> tensor_lists,
                                              torch::stable::Tensor lr, const float beta1, const float beta2,
                                              const float epsilon, torch::stable::Tensor step, const int mode,
                                              const int bias_correction, const float weight_decay,
                                              torch::stable::Tensor inv_scale);

void multi_tensor_sgd_cuda(int chunk_size, torch::stable::Tensor noop_flag,
                           std::vector<std::vector<torch::stable::Tensor>> tensor_lists, float wd,
                           float momentum, float dampening, float lr, bool nesterov, bool first_run,
                           bool wd_after_momentum, float scale);

void multi_tensor_compute_scale_and_scale_inv_cuda(
    int chunk_size, torch::stable::Tensor noop_flag, std::vector<std::vector<torch::stable::Tensor>> tensor_lists,
    float max_fp8, bool force_pow_2_scales, float epsilon);

void multi_tensor_compute_scale_inv_e8m0_cuda(int chunk_size, const nb::object &dummy,
                                              std::vector<std::vector<torch::stable::Tensor>> tensor_lists);

/***************************************************************************************************
 * padding
 **************************************************************************************************/

void fused_multi_row_padding(torch::stable::Tensor input, torch::stable::Tensor output,
                             std::vector<size_t> input_row_list,
                             std::vector<size_t> padded_input_row_list);

void fused_multi_row_unpadding(torch::stable::Tensor input, torch::stable::Tensor output,
                               std::vector<size_t> input_row_list,
                               std::vector<size_t> unpadded_input_row_list);

/***************************************************************************************************
 * Scale swizzling for GEMM
 **************************************************************************************************/

void inplace_swizzle_scale_for_gemm(nb::handle &tensor);

void inplace_multi_tensor_swizzle_scales_for_gemm(std::vector<nb::object> &tensors,
                                                  bool rowwise_usage, bool columnwise_usage);

void inplace_multi_tensor_swizzle_scales_for_gemm_unchecked(std::vector<nb::object> &tensors,
                                                            bool rowwise_usage,
                                                            bool columnwise_usage);

void grouped_swizzle_for_gemm(nb::handle &tensor, bool rowwise, bool columnwise);

/***************************************************************************************************
 * NVSHMEM APIs
 **************************************************************************************************/

// TODO(stable-abi): c10d::ProcessGroup is not part of the PyTorch stable ABI.
void init_nvshmem_backend(c10d::ProcessGroup *process_group);

torch::stable::Tensor create_nvshmem_tensor(const std::vector<int64_t> &shape, torch::headeronly::ScalarType dtype);

void nvshmem_send_on_current_stream(torch::stable::Tensor src, torch::stable::Tensor dst, int peer, torch::stable::Tensor signal);

void nvshmem_wait_on_current_stream(torch::stable::Tensor signal, const std::string &wait_kind);

void nvshmem_finalize();

/***************************************************************************************************
 * Comm+GEMM Overlap Wrappers
 **************************************************************************************************/

void bulk_overlap_ag_with_external_gemm(CommOverlap &allgather_communicator, torch::stable::accelerator::Stream send_stream,
                                        torch::stable::accelerator::Stream recv_stream);

/***************************************************************************************************
 * Newton-Schulz (cuSolverMp)
 **************************************************************************************************/

int64_t cusolvermp_ctx_create(int64_t nccl_comm_ptr, int nranks, int rank);

void cusolvermp_ctx_destroy(int64_t ctx_ptr);

void newton_schulz(int64_t ctx_ptr, int64_t m, int64_t n, torch::stable::Tensor x, int64_t num_iterations,
                   std::vector<float> coefficients);

}  // namespace transformer_engine::pytorch

/***************************************************************************************************
 * Comm+GEMM Overlap Wrappers
 *
 * TODO(stable-abi): these wrappers depend on torch::CustomClassHolder and
 * c10d::ProcessGroup, neither of which is part of the PyTorch stable ABI. Tensor
 * arguments/returns are migrated to torch::stable::Tensor and streams to
 * torch::stable::accelerator::Stream, but the holder base and process-group
 * plumbing remain non-stable.
 **************************************************************************************************/

class CommOverlapHelper : torch::CustomClassHolder {
 public:
  // Shared ownership of an ncclComm_t. The deleter calls ncclCommDestroy when
  // the last reference (held by the helper and/or any CommOverlap consumers)
  // is released, so the communicator outlives whichever owner is destroyed
  // first.
  using NcclCommSharedPtr = std::shared_ptr<std::remove_pointer<ncclComm_t>::type>;

 private:
  bool initialized{false};
  bool backend_is_nccl{false};
  std::map<std::string, c10d::ProcessGroup *> torch_pgs;
  std::map<std::string, NcclCommSharedPtr> nccl_comms;

 public:
  int myrank = -1;
  int numranks = -1;
  int mylocal = -1;
  int numlocal = -1;
  int mynode = -1;
  int numnodes = -1;

  CommOverlapHelper();

  CommOverlapHelper(c10d::ProcessGroup *world_group,
                    std::optional<c10d::ProcessGroup *> intra_node_group);

  ~CommOverlapHelper();

  void ub_allgather(void *globaldata, size_t globalbytes, void *localdata, size_t localbytes,
                    ExtComm comm);

  void ub_barrier(ExtComm comm);

  NcclCommSharedPtr get_nccl_comm(std::string comm_name);
};

class CommOverlap : torch::CustomClassHolder, public transformer_engine::CommOverlapBase {
 private:
  // Keeps the cuBLASMp NCCL communicator alive for the lifetime of this
  // instance, independent of the CommOverlapHelper that created it.
  CommOverlapHelper::NcclCommSharedPtr _nccl_comm;

 public:
  CommOverlap(const std::vector<size_t> &buffer_shape, torch::headeronly::ScalarType buffer_dtype,
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
              const std::vector<size_t> &buffer_shape, torch::headeronly::ScalarType buffer_dtype,
              int num_comm_sm = 16, bool atomic_gemm = false);

  ~CommOverlap() {}

  using transformer_engine::CommOverlapCore::copy_into_buffer;
  void copy_into_buffer(const torch::stable::Tensor &input, bool local_chunk = false);

  torch::stable::Tensor get_buffer(bool local_chunk = false,
                        std::optional<std::vector<int64_t>> shape = std::nullopt);

  std::pair<torch::stable::accelerator::Stream, torch::stable::accelerator::Stream> get_communication_stream();

};  // CommOverlap

class CommOverlapP2P : torch::CustomClassHolder, public transformer_engine::CommOverlapP2PBase {
 private:
  // Keeps the cuBLASMp NCCL communicator alive for the lifetime of this
  // instance, independent of the CommOverlapHelper that created it.
  CommOverlapHelper::NcclCommSharedPtr _nccl_comm;

 public:
  CommOverlapP2P(const std::vector<size_t> &buffer_shape, torch::headeronly::ScalarType buffer_dtype,
                 CommOverlapHelper *helper, int tp_size,
                 transformer_engine::CommOverlapType comm_type,
                 int num_max_streams = NVTE_COMM_OVERLAP_MAX_STREAMS, int comm_cga_size = 1,
                 int gemm_priority = 0, int comm_priority = 0, int num_comm_sm = 1,
                 bool set_sm_margin = false, bool atomic_gemm = false, bool use_ce = true,
                 bool aggregate = false);

  // cuBLASMp variant. See CommOverlap for the `comm_type`/buffer args.
  CommOverlapP2P(CommOverlapHelper *helper, int tp_rank, int tp_size,
                 transformer_engine::CommOverlapType comm_type,
                 const std::vector<size_t> &buffer_shape, torch::headeronly::ScalarType buffer_dtype,
                 int num_comm_sm = 1, bool atomic_gemm = false);

  ~CommOverlapP2P() {}

  using transformer_engine::CommOverlapP2PBase::copy_into_buffer;
  void copy_into_buffer(const torch::stable::Tensor &input, bool local_chunk = false);

  torch::stable::Tensor get_buffer(bool local_chunk = false,
                        std::optional<std::vector<int64_t>> shape = std::nullopt);

  std::pair<torch::stable::accelerator::Stream, torch::stable::accelerator::Stream> get_communication_stream();

};  // CommOverlapP2P

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_H_
