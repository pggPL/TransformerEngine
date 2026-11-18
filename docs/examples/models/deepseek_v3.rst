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


``nsys`` traces of rank 0 on one GB300 node (4 GPUs, expert parallelism over
all four, 64 routed experts, top-k 8, 4096 tokens per GPU), redrawn as one
iteration of the layer with the forward and the backward pass on separate
rows. Every rectangle is one kernel on the compute stream, placed at its
measured time; kernels are attributed to forward or backward by the host
call that launched them. The images are wider than the page, scroll them
sideways. Hovering a kernel shows its name and duration.

**BF16, unfused grouped GEMM**

.. container:: scroll-x

   .. image:: profile_dsv3_te_bf16.svg
      :alt: One iteration of DeepSeekV3Layer in BF16, forward and backward kernels

**MXFP8, fused cuDNN grouped MLP**

.. container:: scroll-x

   .. image:: profile_dsv3_te_mxfp8.svg
      :alt: One iteration of DeepSeekV3Layer in MXFP8 with the fused grouped MLP

The traces are from an earlier revision of the layer than the tables below:
the two grey groups, PyTorch copies (the concatenations that assembled the
MLA key and value tensors) and memsets (zeroing the EP receive buffers), have
since been removed by the fused MLA RoPE kernels and the EP buffer changes,
which is where most of the gap between these 16-17 ms and the 9.5-12.4 ms
below comes from. What remains is the shape of the layer: the expert GEMMs
(or the fused grouped MLP), one NCCL EP exchange in each direction per pass,
the cuDNN attention and the dense MLA projections. The backward is longer
than the forward because every GEMM runs twice (data and weight gradients).
In the MXFP8 variant the fused grouped MLP takes 2.7 ms instead of 4.9 ms of
grouped GEMMs plus 0.8 ms of SwiGLU, at the price of 1.4 ms of quantization
kernels; the forward EP combine is long because the first collective after
the experts absorbs the load imbalance between ranks.

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

Global throughput of the ``te`` variants, in millions of tokens per second:

.. list-table::
   :header-rows: 1

   * - Precision
     - 4 GB200
     - 4 GB300
     - 8 GB300
   * - BF16
     - 1.20
     - 1.33
     - 2.32
   * - MXFP8
     - 1.49
     - 1.59
     - 2.71
   * - MXFP8 fused
     - 1.62
     - 1.73
     - 3.14

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
