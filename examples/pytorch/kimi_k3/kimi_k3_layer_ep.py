# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""KimiK3Layer with expert parallelism over all ranks: forward + backward timing.

One process per GPU, launched via run_kimi_k3_layer_ep.sh (torchrun). Every rank holds
``--num-local-experts`` routed experts of the latent MoE; tokens are exchanged with NCCL EP.
``--attention kda`` times a Kimi Delta Attention layer (cuDNN KDA kernel, Blackwell),
``--attention mla`` a NoPE multi-latent attention layer. Timed iterations run inside a
``torch.cuda.profiler`` window, so ``nsys profile -c cudaProfilerApi --capture-range-end=stop
torchrun ...`` records only them.
"""

import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe
from transformer_engine.pytorch.ep import ep_bootstrap, ep_finalize, release_symm_mem_pool
from transformer_engine.pytorch.models import DeepSeekV3MoE, KimiK3Layer


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="KimiK3Layer EP example (fwd + bwd)")
    p.add_argument("--tokens-per-rank", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64, help="KDA head size (64 or 128)")
    p.add_argument("--latent", type=int, default=None, help="latent MoE width (default hidden/2)")
    p.add_argument("--moe-ffn", type=int, default=1024)
    p.add_argument(
        "--shared-ffn", type=int, default=None, help="shared expert ffn (default moe-ffn)"
    )
    p.add_argument("--num-local-experts", type=int, default=16)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--q-lora-rank", type=int, default=512)
    p.add_argument("--kv-lora-rank", type=int, default=256)
    p.add_argument("--qk-nope-head-dim", type=int, default=64)
    p.add_argument("--qk-rope-head-dim", type=int, default=32)
    p.add_argument("--v-head-dim", type=int, default=64)
    p.add_argument(
        "--kimi",
        action="store_true",
        help=(
            "Kimi-K3 layer dims (hidden 7168, 96 heads x 128, MLA 1536/512/128/64/128, latent"
            " 3584, expert ffn 3072, top-k 16, shared expert ffn 6144). The model has 896"
            " experts; --num-local-experts sets how many each rank holds."
        ),
    )
    p.add_argument("--attention", choices=["kda", "mla"], default="kda")
    p.add_argument("--recipe", choices=["none", "mxfp8"], default="none")
    p.add_argument(
        "--ep-buffer-per-call",
        action="store_true",
        help="Let the layer create a new EpBuffer per forward instead of reusing one.",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args(argv)
    if args.warmup < 0:
        p.error("--warmup must be non-negative")
    if args.iters <= 0:
        p.error("--iters must be positive")
    if args.tokens_per_rank <= 0 or args.tokens_per_rank % 4:
        p.error("--tokens-per-rank must be a positive multiple of 4")
    if args.kimi:
        args.hidden, args.num_heads, args.head_dim = 7168, 96, 128
        args.q_lora_rank, args.kv_lora_rank = 1536, 512
        args.qk_nope_head_dim, args.qk_rope_head_dim, args.v_head_dim = 128, 64, 128
        args.latent, args.moe_ffn, args.topk, args.shared_ffn = 3584, 3072, 16, 6144
    if args.latent is None:
        args.latent = args.hidden // 2
    if args.shared_ffn is None:
        args.shared_ffn = args.moe_ffn
    return args


def _autocast(name):
    if name == "none":
        return nullcontext()
    return te.autocast(enabled=True, recipe=te_recipe.MXFP8BlockScaling())


def main():
    """Build the layer, run warmup + timed fwd/bwd iterations, print throughput on rank 0."""
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world_size = dist.get_rank(), dist.get_world_size()

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 90:
        if rank == 0:
            print(f"SKIPPED: NCCL EP requires SM>=90 (got SM{major}{minor})")
        dist.destroy_process_group()
        return 0

    ep_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    dist.all_reduce(torch.zeros(1, device="cuda"), group=ep_group)
    num_experts = args.num_local_experts * world_size
    # The routed experts run on the latent, so the EP buffers are sized by it.
    ep_bootstrap(
        ep_group,
        num_experts=num_experts,
        max_tokens_per_rank=args.tokens_per_rank,
        hidden_dim=args.latent,
        num_topk=args.topk,
        recv_capacity_per_rank=DeepSeekV3MoE.ep_recv_capacity(
            world_size, args.tokens_per_rank, args.topk, args.num_local_experts
        ),
    )

    torch.manual_seed(0)
    if args.attention == "kda":
        attention_kwargs = dict(head_dim=args.head_dim)
    else:
        attention_kwargs = dict(
            q_lora_rank=args.q_lora_rank,
            kv_lora_rank=args.kv_lora_rank,
            qk_nope_head_dim=args.qk_nope_head_dim,
            qk_rope_head_dim=args.qk_rope_head_dim,
            v_head_dim=args.v_head_dim,
        )
    # One MoE layer (layer 2 of the model); with no block residuals the AttnRes pooling is
    # the identity, so the trace holds attention + latent MoE only.
    layer = KimiK3Layer(
        args.hidden,
        args.num_heads,
        layer_number=2,
        attention_type=args.attention,
        num_experts=num_experts,
        moe_ffn_hidden_size=args.moe_ffn,
        topk=args.topk,
        latent_size=args.latent,
        shared_expert_ffn_hidden_size=args.shared_ffn,
        params_dtype=torch.bfloat16,
        ep_group=ep_group,
        ep_max_tokens_per_rank=args.tokens_per_rank,
        **attention_kwargs,
    )
    seq = args.tokens_per_rank // 4
    x = torch.randn(seq, 4, args.hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    ep_buffer = None if args.ep_buffer_per_call else layer.mlp.make_ep_buffer()

    def step():
        layer.zero_grad(set_to_none=True)
        x.grad = None
        with torch.cuda.nvtx.range("forward"), _autocast(args.recipe):
            out, _ = layer(x, block_residuals=None, ep_buffer=ep_buffer)
        with torch.cuda.nvtx.range("backward"):
            out.backward(torch.ones_like(out))

    for _ in range(args.warmup):
        step()
    # Quantile balancing: pool the histograms of the warmup steps and replace the bias once.
    dist.all_reduce(layer.mlp.routed_experts.qb_histogram, group=ep_group)
    layer.mlp.update_expert_bias()
    torch.cuda.synchronize()
    dist.barrier()

    torch.cuda.profiler.start()
    start = time.perf_counter()
    for i in range(args.iters):
        with torch.cuda.nvtx.range(f"iter{i}"):
            step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / args.iters * 1e3
    torch.cuda.profiler.stop()
    dist.barrier()

    if rank == 0:
        tok_s = args.tokens_per_rank * world_size / (ms / 1e3)
        print(
            f"KimiK3Layer attention={args.attention}:"
            f" ranks={world_size} experts={num_experts} topk={args.topk}"
            f" tokens/rank={args.tokens_per_rank} hidden={args.hidden} latent={args.latent}"
            f" recipe={args.recipe}"
            f" fused_mlp={os.environ.get('NVTE_CUTEDSL_FUSED_GROUPED_MLP', '0')}"
            f" ep_buffer={'per_call' if ep_buffer is None else 'reused'} fwd+bwd"
            f" {ms:.3f} ms/iter ({tok_s / 1e6:.2f} Mtok/s)",
            flush=True,
        )
    ep_finalize()
    release_symm_mem_pool()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
