# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Register the shared CP implementations for torch.compile."""

from functools import partial
import math

import torch

from transformer_engine.pytorch.constants import CPLoadBalancingStrategy
from transformer_engine.pytorch.dynamo.custom_op import register_custom_op_with_autograd
from transformer_engine.pytorch.dynamo.tensor_spec import TensorSpec
from transformer_engine.pytorch.attention.dot_product_attention import context_parallel as cp


def _split_shape(shape, seq_dim):
    return (*shape[:seq_dim], 2, shape[seq_dim] // 2, *shape[seq_dim + 1 :])


def _a2a_shape(shape, seq_dim, size):
    shape = list(shape)
    shape[seq_dim] *= size
    shape[-2] //= size
    return tuple(shape)


def _forward_fake(args, bwd_type):
    def spec(shape, dtype=None):
        return TensorSpec(tuple(shape), dtype or args.q.dtype, device=args.q.device)

    state = cp._cp_backward_args(args, bwd_type)
    state.cp_group = args.cp_group
    state.attn_mask_type = args.attn_mask_type
    state.softmax_scale = (
        args.q.shape[-1] ** -0.5 if args.softmax_scale is None else args.softmax_scale
    )
    state.fp8 = state.is_input_fp8 = False
    state.fwd_nominal_dtype = args.q.dtype
    state.qkv_layout = "_".join([args.qkv_format] * 3)
    state.max_seqlen_q, state.max_seqlen_kv = args.max_seqlen_q, args.max_seqlen_kv

    q_shape, k_shape, v_shape = args.q.shape, args.k.shape, args.v.shape
    out_shape = (*q_shape[:-1], v_shape[-1])
    out = spec(out_shape)
    max_logit = spec((q_shape[-2],)) if args.return_max_logit else None
    seq_dim = args.qkv_format.index("s")
    batch = q_shape[1 - seq_dim]
    size = torch.distributed.get_world_size(args.cp_group)
    rank = torch.distributed.get_rank(args.cp_group)

    if isinstance(state, cp.CPP2PBwdArgs):
        state.cp_group_a2a = args.cp_group_a2a
        state.cp_size_a2a = (
            torch.distributed.get_world_size(args.cp_group_a2a) if args.cp_group_a2a else 1
        )
        state.rank_a2a = torch.distributed.get_rank(args.cp_group_a2a) if args.cp_group_a2a else 0
        state.cp_global_ranks = args.cp_global_ranks
        state.layer_number = args.layer_number
        state.is_output_fp8 = False
        state.max_seqlen_q //= size
        state.max_seqlen_kv //= size
        state.qkv_format = args.qkv_format
        state.orig_q_shape, state.orig_k_shape, state.orig_v_shape = q_shape, k_shape, v_shape
        state.orig_o_shape = out_shape
        q_shape, k_shape, v_shape = [
            _a2a_shape(shape, seq_dim, state.cp_size_a2a) for shape in (q_shape, k_shape, v_shape)
        ]
        state.post_a2a_o_shape = (*q_shape[:-1], v_shape[-1])
        state.softmax_lse_in_packed_format = False
        state.second_half_lse_seqlen = (
            q_shape[seq_dim] // 2 if "causal" in args.attn_mask_type and rank < size - 1 else None
        )
        state.softmax_lse = spec((batch, q_shape[-2], q_shape[seq_dim]), torch.float32)
        state.out = spec(state.post_a2a_o_shape)
        if "causal" in args.attn_mask_type:
            q_shape, k_shape, v_shape = [
                _split_shape(shape, seq_dim) for shape in (q_shape, k_shape, v_shape)
            ]
        state.k_shape, state.v_shape = k_shape, v_shape
        state.k_numel = math.prod(k_shape)
        state.o_shape = (*q_shape[:-1], v_shape[-1])
        state.q = spec(q_shape)
        state.kv = spec((math.prod(k_shape) + math.prod(v_shape),))
        state.cu_seqlens_q_per_step = [spec((batch + 1,), torch.int32) for _ in range(size)]
        state.cu_seqlens_kv_per_step = [spec((batch + 1,), torch.int32) for _ in range(size)]
        state.rng_states = [spec((2,), torch.int64) for _ in range(size)]
        state.attn_biases = [None] * size
    elif isinstance(state, cp.CPAllGatherBwdArgs):
        state.qkv_format = state.dqkv_format = state.o_format = args.qkv_format
        state.dqkv_layout = state.qkv_layout
        state.window_size = args.window_size
        state.load_balancing_strategy = args.load_balancing_strategy
        state.qkv_reshaped = True
        state.max_seqlen_q //= 2 * size
        state.max_seqlen_kv //= 2 * size
        if "causal" in state.attn_mask_type and "bottom_right" not in state.attn_mask_type:
            state.attn_mask_type += "_bottom_right"
        state.q_shape = _split_shape(q_shape, seq_dim)
        state.k_shape = (k_shape[seq_dim], k_shape[1 - seq_dim], *k_shape[2:])
        state.v_shape = (v_shape[seq_dim], v_shape[1 - seq_dim], *v_shape[2:])
        state.o_shape = (*state.q_shape[:-1], v_shape[-1])
        ranges = [
            cp.get_kv_seq_info_after_all_gather(
                chunk,
                size,
                state.max_seqlen_q,
                state.max_seqlen_kv,
                args.window_size,
                "causal" in args.attn_mask_type,
            )
            for chunk in (rank, 2 * size - rank - 1)
        ]
        state.kv_seq_range_per_step = [entry[0] for entry in ranges]
        state.window_size_per_step = [entry[1] for entry in ranges]
        state.cu_seqlens_q = spec((batch + 1,), torch.int32)
        state.cu_seqlens_kv_per_step = [spec((batch + 1,), torch.int32) for _ in range(2)]
        state.softmax_lse_per_step = [
            spec((batch, q_shape[-2], q_shape[seq_dim] // 2, 1), torch.float32) for _ in range(2)
        ]
        state.rng_states = [spec((2,), torch.int64) for _ in range(2)]
        state.thd_cu_seqlens_q_per_step = [None, None]
        state.thd_cu_seqlens_q_padded_per_step = [None, None]
    else:
        state.is_output_fp8 = False
        state.dqkv_format = state.o_format = args.qkv_format
        state.dqkv_layout = state.qkv_layout
        state.softmax_type = args.softmax_type
        state.window_size = args.window_size
        state.orig_q_shape, state.orig_k_shape, state.orig_v_shape = q_shape, k_shape, v_shape
        state.orig_o_shape = out_shape
        q_shape, k_shape, v_shape = [
            _a2a_shape(shape, seq_dim, size) for shape in (q_shape, k_shape, v_shape)
        ]
        if args.is_training:
            state.q, state.k, state.v = spec(q_shape), spec(k_shape), spec(v_shape)
            state.out = spec((*q_shape[:-1], v_shape[-1]))
        state.aux_ctx_tensors = (
            [
                spec((batch, q_shape[-2], q_shape[seq_dim], 1), torch.float32),
                spec((2,), torch.int64),
            ]
            if args.is_training
            else []
        )

    return cp._cp_forward_result(out, max_logit, state, args)


def _backward_fake(args):
    if isinstance(args, cp.CPAllGatherBwdArgs):
        seq_dim = args.qkv_format.index("s")
        shape = args.q_shape
        q_shape = (*shape[:seq_dim], shape[seq_dim] * shape[seq_dim + 1], *shape[seq_dim + 2 :])
        k_shape, v_shape = [
            (shape[seq_dim], shape[1 - seq_dim], *shape[2:])
            for shape in (args.k_shape, args.v_shape)
        ]
    else:
        q_shape, k_shape, v_shape = args.orig_q_shape, args.orig_k_shape, args.orig_v_shape
    return (
        *(
            TensorSpec(tuple(shape), args.grad_output.dtype, device=args.grad_output.device)
            for shape in (q_shape, k_shape, v_shape)
        ),
        None,
        None,
    )


_cp_attention_ops = {
    name: register_custom_op_with_autograd(
        op_name=f"context_parallel_{name}",
        input_tensors_for_grad=["q", "k", "v", "attn_bias", "softmax_offset"],
        fwd_arg_type=cp.CPAttentionFwdArgs,
        fwd_impl=forward,
        fwd_fake_impl=partial(_forward_fake, bwd_type=bwd_type),
        setup_context=cp._cp_setup_context,
        bwd_arg_type=bwd_type,
        bwd_impl=backward,
        bwd_fake_impl=_backward_fake,
        fwd_tags=(torch.Tag.nondeterministic_seeded,),
    )
    for name, forward, backward, bwd_type in (
        ("p2p", cp._cp_p2p_forward, cp._cp_p2p_backward, cp.CPP2PBwdArgs),
        (
            "all_gather",
            cp._cp_all_gather_forward,
            cp._cp_all_gather_backward,
            cp.CPAllGatherBwdArgs,
        ),
        ("a2a", cp._cp_a2a_forward, cp._cp_a2a_backward, cp.CPA2ABwdArgs),
    )
}


def cp_compile_reason(qkv_format, bias_type, softmax_type, load_balancing_strategy):
    """Return the eager fallback reason, if this configuration needs one."""
    if any(op is None for op in _cp_attention_ops.values()):
        return "context parallelism without custom-op support"
    if qkv_format not in ("bshd", "sbhd"):
        return "context parallelism with THD inputs"
    if bias_type != "no_bias" or softmax_type != "vanilla":
        return "context parallelism with attention bias or non-vanilla softmax"
    if load_balancing_strategy != CPLoadBalancingStrategy.DUAL_CHUNK_SWAP:
        return "context parallelism without dual-chunk load balancing"
    return None


def context_parallel_attention(args, comm_type):
    """Call the registered shared forward implementation."""
    op = _cp_attention_ops["p2p" if comm_type == "a2a+p2p" else comm_type]
    result = op(args)
    return result if args.return_max_logit else result[0]
