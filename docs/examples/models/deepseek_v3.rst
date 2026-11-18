..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

DeepSeek-V3
===========

:class:`~transformer_engine.pytorch.models.DeepSeekV3Layer` is a drop-in
analogue of :class:`~transformer_engine.pytorch.TransformerLayer` for the
DeepSeek-V3 architecture: pre-RMSNorm, multi-latent attention (MLA),
residual, pre-RMSNorm, dense MLP or DeepSeek MoE, residual. The first layers
of DeepSeek-V3 are dense (``num_experts=None``), the rest use the MoE.

Features used
-------------

Attention (:class:`~transformer_engine.pytorch.models.MultiLatentAttention`)

* :class:`~transformer_engine.pytorch.DotProductAttention` with asymmetric
  head sizes (``kv_channels=(192, 128)``), which selects the cuDNN fused
  attention backend for the MLA shapes.
* Fused MLA RoPE kernels (:func:`~transformer_engine.pytorch.attention.mla_rope.apply_mla_rope_q`,
  :func:`~transformer_engine.pytorch.attention.mla_rope.apply_mla_rope_kv`)
  that rotate the rope slice and assemble the key and value tensors in one
  pass, with optional YaRN context extension
  (:func:`~transformer_engine.pytorch.attention.mla_rope.build_rope_tables`).
* :class:`~transformer_engine.pytorch.LayerNormLinear` with RMSNorm, so the
  latent normalizations are fused into the up-projections;
  :class:`~transformer_engine.pytorch.Linear` with column and row tensor
  parallelism for the projections.

MoE (:class:`~transformer_engine.pytorch.models.DeepSeekV3MoE`)

* :ref:`Fused router <moe-router>`
  (:func:`~transformer_engine.pytorch.router.fused_topk_with_score_function`):
  sigmoid scores, aux-loss-free expert bias and node-limited grouped top-k in
  one kernel; :ref:`load balancing <moe-load-balancing>` through
  ``update_expert_bias()``.
* :ref:`Token permutation <moe-token-permutation>`
  (:func:`~transformer_engine.pytorch.moe_permute_with_probs`,
  :func:`~transformer_engine.pytorch.moe_permute_and_pad_with_probs`,
  :func:`~transformer_engine.pytorch.moe_unpermute`), padding each expert's
  rows to the alignment the active quantization recipe needs.
* Experts as an :doc:`operation fuser </examples/op_fuser/op_fuser>` sequence
  :class:`~transformer_engine.pytorch.ops.GroupedLinear` ->
  :class:`~transformer_engine.pytorch.ops.ScaledSwiGLU` ->
  :class:`~transformer_engine.pytorch.ops.GroupedLinear`, with the routing
  probabilities applied inside the activation. On Blackwell GPUs with MXFP8 or
  NVFP4 the fuser replaces the three operations by the
  :ref:`fused grouped MLP <moe-grouped-mlp>` kernel
  (``NVTE_CUTEDSL_FUSED_GROUPED_MLP=1``).
* :ref:`Expert parallelism <moe-expert-parallelism>` over NCCL
  (:class:`~transformer_engine.pytorch.ops.MoeDispatch`,
  :class:`~transformer_engine.pytorch.ops.MoeCombine`,
  :func:`~transformer_engine.pytorch.ep.ep_bootstrap`), fused with the expert
  sequence so dispatch and combine feed the grouped GEMM directly.
* Shared expert as a dense ``Linear -> SwiGLU -> Linear`` operation sequence:
  the GEMMs stay separate kernels, while :class:`~transformer_engine.pytorch.ops.SwiGLU`
  computes the gated product in one kernel and writes it directly in the
  quantized format the next GEMM consumes.

Precision

* Every GEMM runs under :class:`~transformer_engine.pytorch.autocast` with any
  :doc:`low-precision recipe </features/low_precision_training/index>`
  (FP8 delayed and current scaling, FP8 block scaling, MXFP8, NVFP4).

Profile on 1 GB200 node
-----------------------

``nsys`` traces of rank 0 on one GB200 node: 4 GPUs, expert parallelism over
all four, 8 local experts per GPU (32 routed experts), top-k 8, 4096 tokens
per GPU, DeepSeek-V3 dimensions, bf16 unless noted, this revision of
``examples/pytorch/deepseek_v3`` with 5 warmup and 10 timed iterations. Each
tab shows one iteration with the forward and the backward pass on separate
rows and a common time scale; every rectangle is one kernel on the compute
stream at its measured time, attributed to forward or backward by the host
call that launched it, and gaps are GPU idle time. Hover a kernel for its
group, name and duration; the images are wider than the page, scroll them
sideways. The ms/iter numbers are from runs without the profiler.

.. tabs::

   .. tab:: Baseline: plain PyTorch

      ``--impl naive``: **27.6 ms/iter**, 26.5 ms of kernels (forward 8.5 ms,
      backward 18.0 ms). Sigmoid router in plain PyTorch, ``all_to_all_single``
      dispatch and combine, a Python loop over the eight local experts with
      dense SwiGLU MLPs, ``index_copy`` / ``index_add`` to place the rows.

      .. container:: scroll-x

         .. raw:: html
            :file: profile_dsv3_naive.svg

      The expert loop turns into 16 small GEMM pairs per pass and, in the
      backward, into as much copy, index and zero-fill work (7.2 ms of
      PyTorch elementwise kernels plus 1.5 ms of memsets) as GEMM time; the
      all-to-all itself is 2.9 ms.

   .. tab:: Plain PyTorch + TE grouped GEMM

      ``--impl naive_grouped``: **17.9 ms/iter**, 17.4 ms of kernels (forward
      6.9 ms, backward 10.5 ms). Same router and all-to-all, but the received
      rows are sorted by expert and run through one
      ``GroupedLinear -> ScaledSwiGLU -> GroupedLinear`` stack.

      .. container:: scroll-x

         .. raw:: html
            :file: profile_dsv3_naive_grouped.svg

      Replacing the loop by grouped GEMMs removes 10 ms: the expert GEMMs
      shrink to 5.0 ms and the elementwise overhead to 0.7 ms. What remains
      above the TE layer is the all-to-all (2.9 ms), the sort / gather /
      scatter around it (1.7 ms) and the host syncs for the split sizes,
      visible as gaps before each all-to-all.

   .. tab:: Optimized with TE (BF16)

      ``DeepSeekV3MoE`` with NCCL expert parallelism: **14.0 ms/iter**,
      14.4 ms of kernels (forward 5.2 ms, backward
      9.2 ms).

      .. container:: scroll-x

         .. raw:: html
            :file: profile_dsv3_te_bf16.svg

      NCCL EP dispatch and combine replace the all-to-all plus the sort /
      gather / scatter and their host syncs (1.7 ms of EP kernels in
      total), the fused router is a single kernel, and the stream stays busy:
      over 99 % of the iteration is kernel time. The expert GEMMs and the
      dense MLA projections are now the bulk of the layer.

   .. tab:: Optimized with TE (MXFP8, fused grouped MLP)

      Same layer under ``te.autocast(recipe=MXFP8BlockScaling())`` with the
      cuDNN fused grouped MLP (``split_alignment=256`` on the expert
      ``GroupedLinear`` ops): **10.3 ms/iter**, 10.1 ms of
      kernels (forward 4.1 ms, backward 6.0 ms).

      .. container:: scroll-x

         .. raw:: html
            :file: profile_dsv3_te_mxfp8.svg

      The fused grouped MLP (fc1 + SwiGLU, fc2, dGLU and wgrad as cuDNN kernels)
      takes 3.1 ms where the BF16 variant spent 5.7 ms of grouped GEMMs plus
      0.9 ms of SwiGLU, and the dense MLA and shared-expert GEMMs drop from
      3.8 ms to 2.0 ms; the price is 1.2 ms of MXFP8 quantization kernels.
      The EP exchange (1.7 ms) and the cuDNN attention (1.0 ms) are unchanged,
      so they now make up a quarter of the iteration. The gaps in the forward
      row, before the attention and before the EP dispatch, are the only idle
      time in the four traces: 1.3 ms in which the GPU waits for the host,
      because the MXFP8 forward launches many short quantization and routing
      kernels and the stream runs dry at this small per-GPU token count.

TE MoE vs. plain PyTorch MoE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same layer, 8 local experts per GPU, only the MoE block differs. ``naive``
is an expert-parallel MoE written in plain PyTorch (``all_to_all_single``
dispatch and combine, a Python loop over the local experts with dense SwiGLU
MLPs); ``naive_grouped`` keeps the all-to-all but runs the experts as one TE
grouped GEMM; ``te`` is ``DeepSeekV3MoE`` with NCCL expert parallelism.
Median of three runs, ms per iteration.

.. list-table::
   :header-rows: 1

   * - MoE implementation
     - 4 GB200 (32 experts)
     - 4 GB300 (32 experts)
     - 8 GB300 (64 experts)
   * - ``naive``: all_to_all + loop over experts
     - 28.61
     - 27.21
     - 27.70
   * - ``naive_grouped``: all_to_all + TE grouped GEMM
     - 17.79
     - 16.72
     - 17.13
   * - ``te``: NCCL EP + grouped GEMM
     - 13.68
     - 12.37
     - 14.09
   * - ``te``, MXFP8 (unfused grouped GEMM)
     - 10.99
     - 10.32
     - 12.11
   * - ``te``, MXFP8 fused grouped MLP
     - 10.11
     - 9.49
     - 10.45

Usage
-----

The natural way to run the layer is with the routed experts sharded over all
GPUs (expert parallelism) and the dense parts replicated or sharded by the
surrounding framework. One process per GPU, launched with ``torchrun``:

.. code-block:: python

    import os
    import torch
    import torch.distributed as dist
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.ep import ep_bootstrap, ep_finalize
    from transformer_engine.pytorch.models import DeepSeekV3Layer, DeepSeekV3MoE

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    ep_group = dist.new_group(backend="nccl")  # every rank; one expert group
    ep_size = dist.get_world_size(ep_group)

    hidden, topk = 7168, 8
    num_local_experts = 32            # 256 experts on 8 GPUs
    num_experts = num_local_experts * ep_size
    tokens_per_rank = 4096            # upper bound on local tokens per forward

    # NCCL EP: once per process, before building any EP module.
    ep_bootstrap(
        ep_group,
        num_experts=num_experts,
        max_tokens_per_rank=tokens_per_rank,
        hidden_dim=hidden,
        num_topk=topk,
        recv_capacity_per_rank=DeepSeekV3MoE.ep_recv_capacity(
            ep_size, tokens_per_rank, topk, num_local_experts
        ),
    )

    layer = DeepSeekV3Layer(
        hidden_size=hidden,
        num_attention_heads=128,
        num_experts=num_experts,
        moe_ffn_hidden_size=2048,
        topk=topk,
        num_groups=8,
        group_topk=4,
        shared_expert_ffn_hidden_size=2048,
        params_dtype=torch.bfloat16,
        ep_group=ep_group,
        ep_max_tokens_per_rank=tokens_per_rank,
    )
    ep_buffer = layer.mlp.make_ep_buffer()  # reuse across steps

    x = torch.randn(tokens_per_rank // 4, 4, hidden, dtype=torch.bfloat16, device="cuda")
    for step in range(num_steps):
        with te.autocast(recipe=te.MXFP8BlockScaling()):
            y = layer(x, ep_buffer=ep_buffer)
        y.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        # Aux-loss-free load balancing: token counts must be summed over the
        # ranks that saw different tokens (here: the whole EP group).
        dist.all_reduce(layer.mlp._last_tokens_per_expert, group=ep_group)
        layer.mlp.update_expert_bias()

    ep_finalize()
    dist.destroy_process_group()

.. code-block:: bash

    torchrun --nproc-per-node=8 train.py

Notes:

* Every rank owns ``num_experts // ep_size`` experts; ``ep_max_tokens_per_rank``
  bounds the local tokens of one forward and sizes the receive buffers.
  Tokens are exchanged in bf16, so the input of an EP layer must be bf16.
* An ``EpBuffer`` holds one forward's routing state until its backward has
  run; reuse one buffer per layer, or let the layer create one per call.
* Tensor parallelism of the attention is independent: pass ``tp_group`` /
  ``tp_size`` and the MLA up- and output projections are column- / row-sharded.
* Without ``ep_group`` the same layer runs all experts locally; this is the
  single-GPU and test configuration. ``examples/pytorch/deepseek_v3`` is a
  complete expert-parallel benchmark with a plain-PyTorch baseline.
