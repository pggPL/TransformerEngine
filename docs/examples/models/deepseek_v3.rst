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
  latent normalizations are fused into the up-projections; one
  :class:`~transformer_engine.pytorch.Linear` computes both latents and the
  shared rope head (``qkv_down_proj``), and column and row tensor
  parallelism applies to the up and output projections.

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
  (FP8 current scaling, MXFP8, NVFP4).

Profile on 1 GB200 node
-----------------------

``nsys`` traces of rank 0 on one GB200 node: DeepSeek-V3 dimensions, 16384
tokens per GPU, expert parallelism over the 4 GPUs (8 local experts per GPU,
top-k 8). Each tab shows one iteration as a forward and a backward image with
time running downwards:

- one rectangle per kernel, labelled with the layer operation it executes
  (for quantization: the precision and the operand); gaps are GPU idle time,
- hover a kernel for its group, name and duration,
- "expert weights" says whether the experts use one grouped
  ``single_grouped_weight`` parameter or per-expert parameters,
- ms/iter are from runs without the profiler.

.. list-table::
   :header-rows: 1

   * - Precision
     - ms / iter
     - Mtok / s
     - expert weights
   * - BF16
     - 59.2
     - 1.11
     - grouped
   * - FP8 current scaling
     - 45.3
     - 1.45
     - grouped
   * - MXFP8, fused grouped MLP
     - 42.9
     - 1.53
     - grouped
   * - NVFP4, fused grouped MLP
     - 36.5
     - 1.79
     - per expert (grouped NVFP4 weight quantization not available yet)

.. tabs::

   .. tab:: BF16

      No quantization; the experts run as BF16 grouped GEMMs with a separate
      SwiGLU kernel: **59.2 ms/iter**.

      **Forward: 21.4 ms**

      .. raw:: html
         :file: profile_dsv3_bf16_fwd.svg

      **Backward: 37.9 ms**

      .. raw:: html
         :file: profile_dsv3_bf16_bwd.svg

   .. tab:: FP8 current scaling

      ``te.autocast(recipe=Float8CurrentScaling())``: per-tensor FP8 with an
      amax reduction before every cast, unfused grouped GEMMs:
      **45.3 ms/iter**.

      **Forward: 16.4 ms**

      .. raw:: html
         :file: profile_dsv3_fp8_cs_fwd.svg

      **Backward: 29.0 ms**

      .. raw:: html
         :file: profile_dsv3_fp8_cs_bwd.svg

   .. tab:: MXFP8 (fused grouped MLP)

      ``te.autocast(recipe=MXFP8BlockScaling())`` with the cuDNN fused grouped
      MLP: **42.9 ms/iter**.

      **Forward: 15.4 ms**

      .. raw:: html
         :file: profile_dsv3_mxfp8_fwd.svg

      **Backward: 27.9 ms**

      .. raw:: html
         :file: profile_dsv3_mxfp8_bwd.svg

   .. tab:: NVFP4 (fused grouped MLP)

      ``te.autocast(recipe=NVFP4BlockScaling())`` with the cuDNN fused grouped
      MLP; inputs and gradients are quantized by the random Hadamard transform
      kernel, expert weights per expert: **36.5 ms/iter**.

      **Forward: 12.8 ms**

      .. raw:: html
         :file: profile_dsv3_nvfp4_fwd.svg

      **Backward: 23.7 ms**

      .. raw:: html
         :file: profile_dsv3_nvfp4_bwd.svg

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
