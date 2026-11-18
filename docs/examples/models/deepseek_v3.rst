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
all four, 8 local experts per GPU (32 routed experts), top-k 8, 16384 tokens
per GPU, DeepSeek-V3 dimensions, this revision of
``examples/pytorch/deepseek_v3`` with 5 warmup and 10 timed iterations. Each
tab shows one iteration as a forward and a backward image on a common time
scale; every rectangle is one kernel on the compute
stream at its measured time, attributed to forward or backward by the host
call that launched it, and gaps are GPU idle time. Hover a kernel for its
group, name and duration; the images are wider than the page, scroll them
sideways. The ms/iter numbers are from runs without the profiler.

.. tabs::

   .. tab:: Baseline: plain PyTorch

      ``--impl naive``: **96.2 ms/iter** (0.68 Mtok/s), 95.3 ms of kernels,
      1.4 ms of the iteration with an idle GPU. Sigmoid router in plain
      PyTorch, ``all_to_all_single`` dispatch and combine, a Python loop over
      the eight local experts with dense SwiGLU MLPs, ``index_copy`` /
      ``index_add`` to place the rows.

      **Forward: 32.6 ms of kernels, 1.0 ms idle**

      .. container:: scroll-x

         .. raw:: html
            :file: profile_dsv3_naive_fwd.svg

      **Backward: 62.7 ms of kernels, 0.35 ms idle**

      .. container:: scroll-x

         .. raw:: html
            :file: profile_dsv3_naive_bwd.svg

      Dense GEMMs are a third of the time (33.7 ms: 16 small GEMM pairs per
      pass from the expert loop plus the MLA projections), the two
      all-to-alls 10.0 ms, the cuDNN attention 9.0 ms, the sort / gather /
      index kernels of the dispatch 8.5 ms and zero-fills 3.4 ms. The grey
      "PyTorch copies / elementwise" group, 21.3 ms per iteration, is the
      glue around the expert loop in ``NaiveMoE.forward``:

      * 9.0 ms, 47 device-to-device ``memcpy``: ``y_recv = y_recv.index_copy(0, sel, ...)``
        is out of place, so every expert iteration first clones the whole
        receive buffer (8 clones per forward, and their backward
        counterparts), plus the copies around the autograd
        ``all_to_all_single``.
      * 8.2 ms, 29 bf16 ``add`` kernels, 8.6 ms of them in the backward:
        autograd summing the gradients that the eight ``index_copy`` steps
        and the eight ``x_recv[sel]`` gathers contribute to the same
        full-size ``y_recv`` / ``x_recv`` tensors, plus the residual adds and
        ``out + shared_w2(...)``.
      * 3.2 ms, 51 bf16 ``mul`` kernels and 1.6 ms of ``silu`` / ``silu_backward``:
        ``F.silu(a) * g`` and ``* p_recv[sel, None]`` run as separate
        elementwise kernels per expert and for the shared expert, forward and
        backward, where TE uses one fused ``ScaledSwiGLU`` kernel.
      * 4.4 ms of strided ``elementwise_kernel`` copies: ``p_recv[...].to(x.dtype)``
        casts, ``.contiguous()`` before the all-to-alls, and the strided
        gradient copies of ``chunk`` and indexing in the backward.
      * 0.3 ms ``cat``: the backward of ``h.chunk(2)`` in ``_swiglu``.

   .. tab:: Optimized with TE (MXFP8, fused grouped MLP)

      ``DeepSeekV3MoE`` with NCCL expert parallelism under
      ``te.autocast(recipe=MXFP8BlockScaling())`` and the cuDNN fused grouped
      MLP (``split_alignment=256`` on the expert ``GroupedLinear`` ops):
      **43.3 ms/iter** (1.51 Mtok/s), 43.4 ms of kernels, 0.1 ms of the
      iteration with an idle GPU.

      **Forward: 15.7 ms of kernels, 0.09 ms idle**

      .. container:: scroll-x

         .. raw:: html
            :file: profile_dsv3_te_mxfp8_fwd.svg

      **Backward: 27.7 ms of kernels, 0.04 ms idle**

      .. container:: scroll-x

         .. raw:: html
            :file: profile_dsv3_te_mxfp8_bwd.svg

      The experts are twelve cuDNN kernels per iteration (fc1 + SwiGLU, fc2,
      dGLU, wgrad) totalling 11.5 ms, there is no separate activation kernel,
      and the quantization that feeds them costs 3.6 ms. NCCL EP dispatch and
      combine (4.6 ms) plus their local permutes (2.2 ms) replace the
      all-to-all, the sort / scatter work and the host syncs of the baseline.
      At this shape the attention (9.5 ms) and the dense MLA projections
      (7.8 ms) are as large as the expert MLP, and the stream is never idle.

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
