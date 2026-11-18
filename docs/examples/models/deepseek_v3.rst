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

The diagram traces one layer as a plain PyTorch implementation would run
it, operation by operation, and marks which Transformer Engine module or
kernel takes each group of operations over.

.. image:: deepseek_v3_layer.svg
   :width: 100%
   :alt: deepseek_v3 layer: baseline operations and the TE features covering them

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
* Shared expert as a dense ``Linear -> SwiGLU -> Linear`` operation sequence.

Precision

* Every GEMM runs under :class:`~transformer_engine.pytorch.autocast` with any
  :doc:`low-precision recipe </features/low_precision_training/index>`
  (FP8 delayed and current scaling, FP8 block scaling, MXFP8, NVFP4).

Usage
-----

.. code-block:: python

    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.models import DeepSeekV3Layer

    layer = DeepSeekV3Layer(
        hidden_size=7168,
        num_attention_heads=128,
        num_experts=256,
        moe_ffn_hidden_size=2048,
        topk=8,
        num_groups=8,
        group_topk=4,
        shared_expert_ffn_hidden_size=2048,
        params_dtype=torch.bfloat16,
    )

    x = torch.randn(4096, 1, 7168, dtype=torch.bfloat16, device="cuda")
    with te.autocast(recipe=te.MXFP8BlockScaling()):
        y = layer(x)
    y.sum().backward()
    layer.mlp.update_expert_bias()  # aux-loss-free load balancing, once per step

With ``ep_group`` the routed experts are sharded across the group and tokens
are exchanged over NCCL expert parallelism; call
:func:`~transformer_engine.pytorch.ep.ep_bootstrap` once per process first.
``examples/pytorch/deepseek_v3`` is a complete expert-parallel benchmark
of the layer.

Where the time goes
-------------------

All numbers are for one ``DeepSeekV3Layer`` with DeepSeek-V3 dimensions
(hidden 7168, 128 heads, MLA 1536 / 512 / 128 / 64 / 128, expert ffn 2048,
shared expert ffn 2048), top-k 8, 4096 tokens per GPU, forward + backward,
bf16 unless noted. Timings are wall-clock with CUDA synchronization, no CUDA
graphs. Reproduce them with ``examples/pytorch/deepseek_v3``; the full setup
(software versions, environment variables) is in that example's README.

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

Kernel breakdown of the fastest variant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``te``, MXFP8 fused grouped MLP, 8 GB300: per GPU and iteration, from
``nsys stats --report cuda_gpu_kern_sum`` (10.8 ms of kernel time; 10.5 ms
per iteration without the profiler; no memsets or copies left).

.. list-table::
   :header-rows: 1

   * - Group
     - ms
     - Kernels
   * - NCCL EP all-to-all
     - 2.3
     - dispatch (1.05) and combine (1.29), each twice per iteration (fwd + bwd)
   * - NCCL EP local permute
     - 0.6
     - staging buffer to expert-major layout, zero-filled padding
   * - rank skew wait
     - 0.5
     - routing-map all-gather absorbs load imbalance between ranks
   * - fused grouped MLP (cuDNN, MXFP8)
     - 2.4
     - fc1 + SwiGLU fwd (0.53), fc2 fwd (0.82), dGLU bwd (0.31), wgrad (0.78)
   * - MXFP8 quantization
     - 1.1
     - recv buffer (0.4), dense GEMM inputs (0.7)
   * - dense MXFP8 GEMMs
     - 1.4
     - MLA projections, shared expert
   * - attention (cuDNN fused)
     - 0.9
     - fprop (0.18), bprop (0.51), dq / dO helpers
   * - RMSNorm, RoPE, residual adds
     - 0.8
     - rmsnorm (0.39), MLA RoPE (0.18), adds (0.2)

For comparison, ``naive`` spends 3.4 ms in all-to-all, 6.5 ms in eight
separate GEMM pairs and about 12 ms in indexing, zero-fills, copies and adds;
``naive_grouped`` removes the loop (about 10.5 ms) and NCCL EP removes the
remaining all-to-all plus sort / gather / scatter overhead (2.9 ms on 8 GPUs).

Other implementations of the same layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measured with an earlier revision of ``DeepSeekV3Layer`` (before the EP
buffer and alignment changes above), 64 routed experts, top-k 8, sequence
length 4096, ms per layer, forward + backward.

.. list-table::
   :header-rows: 1

   * - Implementation
     - Setup
     - ms / layer
     - Speedup
   * - Hugging Face ``DeepseekV3DecoderLayer`` (SDPA, bf16)
     - 1 GB300, no EP
     - 56.4
     - 1x
   * - ``DeepSeekV3Layer`` (bf16)
     - 1 GB300, no EP
     - 13.8
     - 4.1x
   * - TorchTitan DeepSeek-V3 layer (FSDP2 + EP all-to-all, FlexAttention, ``torch._grouped_mm``)
     - 4 GB300, EP 4, 4096 tokens per GPU
     - ~54
     - 1x
   * - ``DeepSeekV3Layer`` (bf16)
     - 4 GB300, EP 4, 4096 tokens per GPU
     - 15.1
     - 3.6x
   * - ``DeepSeekV3Layer`` (MXFP8, fused grouped MLP)
     - 4 GB300, EP 4, 4096 tokens per GPU
     - 12.1
     - 4.4x

The Hugging Face layer and ``DeepSeekV3Layer`` were loaded with the same
weights; forward outputs and input gradients agree to bf16 precision. The
TorchTitan time is the difference between 8-layer and 4-layer models divided
by four, so it includes TorchTitan's per-layer FSDP2 all-gather and
reduce-scatter (``dp_shard=4``), which the TE benchmark does not perform.
