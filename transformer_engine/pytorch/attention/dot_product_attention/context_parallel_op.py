# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Custom-op boundary for dense, high-precision context-parallel attention."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch

from transformer_engine.pytorch.constants import CPLoadBalancingStrategy, dist_group_type
from transformer_engine.pytorch.dynamo.custom_op import register_custom_op_with_autograd
from transformer_engine.pytorch.dynamo.tensor_spec import TensorSpec
from transformer_engine.pytorch.quantized_tensor import restore_from_func_ctx
from transformer_engine.pytorch.attention.dot_product_attention import context_parallel as cp


@dataclass
class CPAttentionFwdArgs:
    """Inputs and static communication configuration."""

    q: Optional[torch.Tensor] = None
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_kv: Optional[torch.Tensor] = None
    cp_group: Optional[dist_group_type] = None
    cp_group_a2a: Optional[dist_group_type] = None
    cp_global_ranks: Tuple[int, ...] = ()
    stream: Tuple[int, int] = (0, 0)
    cp_comm_type: str = "p2p"
    is_training: bool = True
    max_seqlen_q: int = 0
    max_seqlen_kv: int = 0
    dropout_p: float = 0.0
    softmax_scale: Optional[float] = None
    qkv_format: str = "bshd"
    attn_mask_type: str = "causal"
    deterministic: bool = False
    window_size: Tuple[int, int] = (-1, 0)
    return_max_logit: bool = False


_SAVED_FIELDS = ("saved_q", "saved_k", "saved_v", "saved_out", "lse", "cu", "rng")


@dataclass
class CPAttentionBwdArgs(CPAttentionFwdArgs):
    """Saved tensors cross the boundary independently of Python context objects."""

    grad_output: Optional[torch.Tensor] = None
    out: Optional[torch.Tensor] = None
    saved_q: Optional[torch.Tensor] = None
    saved_k: Optional[torch.Tensor] = None
    saved_v: Optional[torch.Tensor] = None
    saved_out: Optional[torch.Tensor] = None
    lse: Optional[torch.Tensor] = None
    cu: Optional[torch.Tensor] = None
    rng: Optional[torch.Tensor] = None

    def setup_saved_tensors(self, ctx):
        """Restore original inputs and CP intermediates."""
        for name, tensor in zip(
            ("q", "k", "v", "cu_seqlens_q", "cu_seqlens_kv", "out", *_SAVED_FIELDS),
            restore_from_func_ctx(ctx),
        ):
            setattr(self, name, tensor)


class _CPContext:
    """Per-call context for the existing eager implementation."""

    def save_for_backward(self, *tensors):
        """Collect intermediates to return through the custom op."""
        self.saved_tensors = tensors


def _split_shape(shape, seq_dim):
    return (*shape[:seq_dim], 2, shape[seq_dim] // 2, *shape[seq_dim + 1 :])


def _a2a_shape(shape, seq_dim, size):
    shape = list(shape)
    shape[seq_dim] *= size
    shape[-2] //= size
    return tuple(shape)


def _cp_function(comm_type):
    if comm_type in ("p2p", "a2a+p2p"):
        return cp.AttnFuncWithCPAndKVP2P
    if comm_type == "all_gather":
        return cp.AttnFuncWithCPAndKVAllGather
    return cp.AttnFuncWithCPAndQKVOA2A


def _forward_impl(args):
    ctx = _CPContext()
    stream = torch.cuda.ExternalStream(args.stream[0], device=args.stream[1])
    common = (
        args.is_training,
        args.q,
        args.k,
        args.v,
        args.cu_seqlens_q,
        args.cu_seqlens_kv,
        args.max_seqlen_q,
        args.max_seqlen_kv,
        None,
        None,
        args.dropout_p,
        args.softmax_scale,
        args.qkv_format,
        args.attn_mask_type,
        "no_bias",
        None,
        args.deterministic,
        True,
        args.return_max_logit,
        0.0,
    )
    if args.cp_comm_type in ("p2p", "a2a+p2p"):
        group = args.cp_group
        if args.cp_group_a2a is not None:
            group = [args.cp_group_a2a, group]
        extra = (
            False,
            None,
            group,
            args.cp_global_ranks,
            stream,
            None,
            False,
            False,
            False,
            False,
            1,
        )
    elif args.cp_comm_type == "all_gather":
        extra = (
            args.window_size,
            args.cp_group,
            stream,
            False,
            False,
            False,
            False,
            None,
            None,
            False,
            CPLoadBalancingStrategy.DUAL_CHUNK_SWAP,
        )
    else:
        extra = (
            args.window_size,
            False,
            None,
            args.cp_group,
            stream,
            None,
            False,
            False,
            False,
            "vanilla",
            None,
            False,
        )
    result = _cp_function(args.cp_comm_type).forward(ctx, *common, *extra)
    out, max_logit = result if args.return_max_logit else (result, None)
    if not args.is_training:
        return out, max_logit, (None,) * len(_SAVED_FIELDS), {}
    saved = ctx.saved_tensors
    if args.cp_comm_type in ("p2p", "a2a+p2p"):
        size = cp.get_distributed_world_size(args.cp_group)
        q = saved[3] if args.cp_group_a2a is not None else None
        saved_out = saved[5] if args.cp_group_a2a is not None else None
        cu = torch.stack(saved[9 : 9 + 2 * size])
        rng = torch.stack(saved[9 + 2 * size : 9 + 3 * size])
        tensors = (q, saved[4], None, saved_out, saved[6], cu, rng)
    elif args.cp_comm_type == "all_gather":
        tensors = (
            None,
            None,
            None,
            None,
            torch.stack(saved[12:14]),
            torch.stack((saved[8], saved[10], saved[11])),
            torch.stack(saved[14:16]),
        )
    else:
        tensors = (*saved[4:8], saved[12], None, saved[13])
    return out, max_logit, tensors, {}


def _forward_fake(args):
    def spec(shape, dtype=None):
        return TensorSpec(tuple(shape), dtype or args.q.dtype, device=args.q.device)

    out_shape = (*args.q.shape[:-1], args.v.shape[-1])
    out = spec(out_shape)
    max_logit = spec((args.q.shape[-2],)) if args.return_max_logit else None
    if not args.is_training:
        return out, max_logit, (None,) * len(_SAVED_FIELDS), {}
    seq_dim = args.qkv_format.index("s")
    batch = args.q.shape[1 - seq_dim]
    size = torch.distributed.get_world_size(args.cp_group)
    q_shape, k_shape, v_shape = args.q.shape, args.k.shape, args.v.shape
    if args.cp_comm_type in ("p2p", "a2a+p2p"):
        a2a_size = torch.distributed.get_world_size(args.cp_group_a2a) if args.cp_group_a2a else 1
        q_shape, k_shape, v_shape = [
            _a2a_shape(shape, seq_dim, a2a_size) for shape in (q_shape, k_shape, v_shape)
        ]
        saved_q = q_shape
        if "causal" in args.attn_mask_type:
            saved_q = _split_shape(q_shape, seq_dim)
        tensors = (
            spec(saved_q) if args.cp_group_a2a else None,
            spec((math.prod(k_shape) + math.prod(v_shape),)),
            None,
            spec((*q_shape[:-1], v_shape[-1])) if args.cp_group_a2a else None,
            spec((batch, q_shape[-2], q_shape[seq_dim]), torch.float32),
            spec((2 * size, batch + 1), torch.int32),
            spec((size, 2), torch.int64),
        )
    elif args.cp_comm_type == "all_gather":
        tensors = (
            None,
            None,
            None,
            None,
            spec((2, batch, q_shape[-2], q_shape[seq_dim] // 2, 1), torch.float32),
            spec((3, batch + 1), torch.int32),
            spec((2, 2), torch.int64),
        )
    else:
        q_shape, k_shape, v_shape = [
            _a2a_shape(shape, seq_dim, size) for shape in (q_shape, k_shape, v_shape)
        ]
        tensors = (
            spec(q_shape),
            spec(k_shape),
            spec(v_shape),
            spec((*q_shape[:-1], v_shape[-1])),
            spec((batch, q_shape[-2], q_shape[seq_dim], 1), torch.float32),
            None,
            spec((2,), torch.int64),
        )
    return out, max_logit, tensors, {}


def _setup_context(bwd, fwd, outputs, _attrs, saved):
    for name in CPAttentionFwdArgs.__dataclass_fields__:
        if name not in ("q", "k", "v", "cu_seqlens_q", "cu_seqlens_kv"):
            setattr(bwd, name, getattr(fwd, name))
    return (fwd.q, fwd.k, fwd.v, fwd.cu_seqlens_q, fwd.cu_seqlens_kv, outputs[0], *saved)


def _backward_impl(args):
    ctx = _CPContext()
    ctx.cp_group = args.cp_group
    ctx.cp_stream = torch.cuda.ExternalStream(args.stream[0], device=args.stream[1])
    ctx.dropout_p = args.dropout_p
    ctx.softmax_scale = (
        args.q.shape[-1] ** -0.5 if args.softmax_scale is None else args.softmax_scale
    )
    ctx.attn_mask_type = args.attn_mask_type
    ctx.attn_bias_type = "no_bias"
    ctx.attn_bias_shape = None
    ctx.deterministic = args.deterministic
    ctx.softcap = 0.0
    ctx.use_fused_attention = True
    ctx.use_flash_attn_3 = ctx.use_flash_attn_4 = False
    ctx.pad_between_seqs = False
    ctx.fp8 = ctx.is_input_fp8 = ctx.is_output_fp8 = False
    ctx.fp8_meta = ctx.fp8_recipe = None
    ctx.QKV_quantizer = ctx.O_quantizer = ctx.S_quantizer = None
    ctx.dQKV_quantizer = ctx.dO_quantizer = ctx.dP_quantizer = None
    ctx.fwd_nominal_dtype = args.q.dtype
    ctx.qkv_format = ctx.dqkv_format = ctx.o_format = args.qkv_format
    ctx.qkv_layout = ctx.dqkv_layout = "_".join((args.qkv_format,) * 3)
    ctx.qkv_scale_inv_format = None
    ctx.softmax_type = "vanilla"
    ctx.window_size = args.window_size
    ctx.orig_q_shape, ctx.orig_k_shape, ctx.orig_v_shape = args.q.shape, args.k.shape, args.v.shape
    ctx.orig_o_shape = (*args.q.shape[:-1], args.v.shape[-1])
    ctx.max_seqlen_q, ctx.max_seqlen_kv = args.max_seqlen_q, args.max_seqlen_kv
    seq_dim = args.qkv_format.index("s")
    size = cp.get_distributed_world_size(args.cp_group)
    rank = cp.get_distributed_rank(args.cp_group)
    q, k, v = args.q, args.k, args.v
    if args.cp_comm_type in ("p2p", "a2a+p2p"):
        ctx.cp_group_a2a = args.cp_group_a2a
        ctx.cp_size_a2a = (
            cp.get_distributed_world_size(args.cp_group_a2a) if args.cp_group_a2a else 1
        )
        ctx.rank_a2a = cp.get_distributed_rank(args.cp_group_a2a) if args.cp_group_a2a else 0
        ctx.cp_global_ranks = args.cp_global_ranks
        ctx.layer_number = 1
        ctx.max_seqlen_q //= size
        ctx.max_seqlen_kv //= size
        shapes = [_a2a_shape(x.shape, seq_dim, ctx.cp_size_a2a) for x in (q, k, v)]
        ctx.post_a2a_o_shape = (*shapes[0][:-1], shapes[2][-1])
        ctx.softmax_lse_in_packed_format = False
        ctx.second_half_lse_seqlen = (
            shapes[0][seq_dim] // 2 if "causal" in args.attn_mask_type and rank < size - 1 else None
        )
        if "causal" in args.attn_mask_type:
            shapes = [_split_shape(shape, seq_dim) for shape in shapes]
        ctx.k_shape, ctx.v_shape = shapes[1:]
        ctx.k_numel = math.prod(ctx.k_shape)
        ctx.o_shape = (*shapes[0][:-1], shapes[2][-1])
        q = args.saved_q if args.cp_group_a2a else q.view(shapes[0])
        out = args.saved_out if args.cp_group_a2a else args.out
        # Backward adds a dimension to LSE; give it a private tensor view.
        tensors = (
            None,
            None,
            None,
            q,
            args.saved_k,
            out,
            args.lse.view_as(args.lse),
            None,
            None,
            *args.cu.unbind(),
            *args.rng.unbind(),
            *((None,) * size),
        )
    elif args.cp_comm_type == "all_gather":
        ctx.load_balancing_strategy = CPLoadBalancingStrategy.DUAL_CHUNK_SWAP
        ctx.qkv_reshaped = True
        ctx.max_seqlen_q //= 2 * size
        ctx.max_seqlen_kv //= 2 * size
        if "causal" in ctx.attn_mask_type and "bottom_right" not in ctx.attn_mask_type:
            ctx.attn_mask_type += "_bottom_right"
        q = q.view(_split_shape(q.shape, seq_dim))
        k, v = [x.movedim(seq_dim, 0).contiguous() for x in (k, v)]
        ctx.q_shape, ctx.k_shape, ctx.v_shape = q.shape, k.shape, v.shape
        ctx.o_shape = (*q.shape[:-1], v.shape[-1])
        ranges = [
            cp.get_kv_seq_info_after_all_gather(
                chunk,
                size,
                ctx.max_seqlen_q,
                ctx.max_seqlen_kv,
                args.window_size,
                "causal" in args.attn_mask_type,
            )
            for chunk in (rank, 2 * size - rank - 1)
        ]
        ctx.kv_seq_range_per_step = [x[0] for x in ranges]
        ctx.window_size_per_step = [x[1] for x in ranges]
        tensors = (
            None,
            None,
            None,
            None,
            q,
            k,
            v,
            args.out,
            args.cu[0],
            None,
            args.cu[1],
            args.cu[2],
            *args.lse.unbind(),
            *args.rng.unbind(),
        )
    else:
        tensors = (
            None,
            None,
            None,
            None,
            args.saved_q,
            args.saved_k,
            args.saved_v,
            args.saved_out,
            args.cu_seqlens_q,
            args.cu_seqlens_kv,
            None,
            None,
            args.lse,
            args.rng,
        )
    ctx.saved_tensors = tensors
    ctx.tensor_objects = [None] * len(tensors)
    grads = _cp_function(args.cp_comm_type).backward(ctx, args.grad_output)
    dq, dk, dv = grads[1:4]
    # Ring backward returns dK and dV as views of one allocation.
    if args.cp_comm_type in ("p2p", "a2a+p2p"):
        dk = dk.clone()
    return dq.contiguous(), dk.contiguous(), dv.contiguous()


def _backward_fake(args):
    return tuple(
        TensorSpec(tuple(x.shape), x.dtype, device=x.device) for x in (args.q, args.k, args.v)
    )


_cp_attention_op = register_custom_op_with_autograd(
    op_name="context_parallel_attention",
    input_tensors_for_grad=["q", "k", "v"],
    fwd_arg_type=CPAttentionFwdArgs,
    fwd_impl=_forward_impl,
    fwd_fake_impl=_forward_fake,
    setup_context=_setup_context,
    bwd_arg_type=CPAttentionBwdArgs,
    bwd_impl=_backward_impl,
    bwd_fake_impl=_backward_fake,
    fwd_tags=(torch.Tag.nondeterministic_seeded,),
)


def cp_compile_reason(qkv_format, bias_type, softmax_type, load_balancing_strategy):
    """Return a fallback reason for configurations outside the custom-op contract."""
    if _cp_attention_op is None:
        return "context parallelism without custom-op support"
    if qkv_format not in ("bshd", "sbhd"):
        return "context parallelism with THD inputs"
    if bias_type != "no_bias" or softmax_type != "vanilla":
        return "context parallelism with attention bias or non-vanilla softmax"
    if load_balancing_strategy != CPLoadBalancingStrategy.DUAL_CHUNK_SWAP:
        return "context parallelism without dual-chunk load balancing"
    return None


def context_parallel_attention(args):
    """Dispatch the supported CP call from the traceable Python frontend."""
    group = args["cp_group"]
    group_a2a = None
    if isinstance(group, list):
        group_a2a, group = group
    inputs = CPAttentionFwdArgs(
        **{
            name: args[name]
            for name in CPAttentionFwdArgs.__dataclass_fields__
            if name in args and name not in ("cp_group", "cp_global_ranks")
        },
        cp_group=group,
        cp_group_a2a=group_a2a,
        cp_global_ranks=tuple(args["cp_global_ranks"] or ()),
        stream=args["cp_stream"],
    )
    result = _cp_attention_op(inputs)
    return result if inputs.return_max_logit else result[0]
