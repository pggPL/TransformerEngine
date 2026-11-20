..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Kimi-K3
=======

:class:`~transformer_engine.pytorch.models.KimiK3Layer` is one layer of the
Kimi-K3 architecture. The model alternates three Kimi Delta Attention (KDA)
layers with one multi-latent attention layer, uses a dense SiTU-GLU MLP in the
first layer and a latent MoE afterwards, and replaces the plain residual
stream with the Attention Residuals (AttnRes) highway.

Features used
-------------

Attention

* :class:`~transformer_engine.pytorch.KimiDeltaAttention`: the KDA
  recurrence through the cuDNN frontend kernel (Blackwell GPUs), with the
  model's feature map computed in the kernel: L2 normalization of queries and
  keys, the sigmoid of the write strength and the safe decay gate
  ``gate_lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``.
  :class:`~transformer_engine.pytorch.models.KimiLinearAttention` adds the
  projections around it and two fused Triton kernels: SiLU of the causal short
  convolution, written directly as the q, k and v tensors in the layout the
  recurrence consumes, and the per-head RMSNorm times the sigmoid gate on its
  output. Both compute in float32 and replace the layout copies, dtype casts
  and separate elementwise kernels of the plain PyTorch formulation.
* :class:`~transformer_engine.pytorch.models.MultiLatentAttention` from the
  DeepSeek-V3 layer in its NoPE variant (``rope=False``) with a sigmoid output
  gate (``output_gate=True``): the same cuDNN fused attention with
  asymmetric head sizes, fused latent RMSNorms and one down-projection GEMM.
  The keys are assembled by the fused MLA RoPE kernel without rotation, and
  the output gate is one fused kernel.

MoE (:class:`~transformer_engine.pytorch.models.KimiK3MoE`)

* :class:`~transformer_engine.pytorch.models.DeepSeekV3MoE` for routing,
  permutation, experts and expert parallelism, running on the latent with the
  router reading the full hidden state (``router_input``). Everything listed
  for the :doc:`DeepSeek-V3 MoE <deepseek_v3>` applies.
* SiTU-GLU experts: :class:`~transformer_engine.pytorch.ops.ScaledSiTUGLU`
  in the expert sequence and :class:`~transformer_engine.pytorch.ops.SiTUGLU`
  in the dense MLP and shared expert. Each is one kernel for the gated
  product (and one for its backward) that writes its output directly in the
  quantized format the next GEMM consumes; the GEMMs around them stay
  separate kernels. The :ref:`fused grouped MLP <moe-grouped-mlp>` supports
  SiTU-GLU as well.
* Quantile balancing: the :ref:`fused router <moe-router>` accumulates the
  required-bias histogram (``qb_histogram``) in the same kernel as the top-k
  selection, and ``update_expert_bias()`` recovers the
  ``topk / num_experts`` quantile per expert
  (``expert_bias_update="quantile"``).
* :class:`~transformer_engine.pytorch.RMSNorm` after the experts and
  :class:`~transformer_engine.pytorch.Linear` for the latent down and up
  projections.

Residual stream

* AttnRes pooling (:func:`~transformer_engine.pytorch.models.kimi_k3.attn_res_pool`):
  a float32 softmax over the committed block residuals and the running prefix
  sum before each sub-layer, with the block commits handled by the layer.

Precision

* Every GEMM runs under :class:`~transformer_engine.pytorch.autocast` with any
  :doc:`low-precision recipe </features/low_precision_training/index>`; the
  KDA recurrence, the AttnRes pooling, the convolution and the gates stay in
  bf16 / float32.

Profile on 1 GB200 node
-----------------------

``nsys`` traces of rank 0 on one GB200 node: Kimi-K3 dimensions (hidden 7168,
96 heads x 128, latent 3584, expert ffn 3072, top-k 16, shared expert ffn
6144), 16384 tokens per GPU, expert parallelism over the 4 GPUs (16 local
experts per GPU). The model alternates 3 KDA layers with 1 MLA layer; both
layer types share the same latent MoE and differ only in the attention block,
so each is profiled separately. Each tab shows one iteration as a forward and
a backward image with time running downwards:

- one rectangle per kernel, labelled with the layer operation it executes
  (for quantization: the precision and the operand); gaps are GPU idle time,
- hover a kernel for its group, name and duration,
- "expert weights" says whether the experts use one grouped
  ``single_grouped_weight`` parameter or per-expert parameters,
- ms/iter are from runs without the profiler.

.. list-table::
   :header-rows: 1

   * - Layer
     - Precision
     - ms / iter
     - Mtok / s
     - expert weights
   * - KDA
     - BF16
     - 162.9
     - 0.40
     - grouped
   * - KDA
     - FP8 current scaling
     - 130.0
     - 0.50
     - grouped
   * - KDA
     - MXFP8, fused grouped MLP
     - 110.3
     - 0.59
     - grouped
   * - KDA
     - NVFP4, fused grouped MLP
     - 89.5
     - 0.73
     - per expert (grouped NVFP4 weight quantization not available yet)
   * - MLA
     - BF16
     - 113.9
     - 0.58
     - grouped
   * - MLA
     - FP8 current scaling
     - 87.4
     - 0.75
     - grouped
   * - MLA
     - MXFP8, fused grouped MLP
     - 61.9
     - 1.06
     - grouped
   * - MLA
     - NVFP4, fused grouped MLP
     - 48.0
     - 1.37
     - per expert (grouped NVFP4 weight quantization not available yet)

.. tabs::

   .. tab:: KDA layer

      .. tabs::

         .. tab:: BF16

            No quantization; the experts run as BF16 grouped GEMMs with a separate
            SiTU-GLU kernel: **162.9 ms/iter**.

            **Forward: 59.7 ms**

            .. raw:: html
               :file: profile_kimi_kda_bf16_fwd.svg

            **Backward: 103.7 ms**

            .. raw:: html
               :file: profile_kimi_kda_bf16_bwd.svg

         .. tab:: FP8 current scaling

            ``te.autocast(recipe=Float8CurrentScaling())``: per-tensor FP8 with an
            amax reduction before every cast, unfused grouped GEMMs: **130.0 ms/iter**.

            **Forward: 50.9 ms**

            .. raw:: html
               :file: profile_kimi_kda_fp8_cs_fwd.svg

            **Backward: 79.5 ms**

            .. raw:: html
               :file: profile_kimi_kda_fp8_cs_bwd.svg

         .. tab:: MXFP8 (fused grouped MLP)

            ``te.autocast(recipe=MXFP8BlockScaling())`` with the cuDNN fused grouped
            MLP: **110.3 ms/iter**.

            **Forward: 39.7 ms**

            .. raw:: html
               :file: profile_kimi_kda_mxfp8_fwd.svg

            **Backward: 70.4 ms**

            .. raw:: html
               :file: profile_kimi_kda_mxfp8_bwd.svg

         .. tab:: NVFP4 (fused grouped MLP)

            ``te.autocast(recipe=NVFP4BlockScaling())`` with the cuDNN fused grouped
            MLP; inputs and gradients are quantized by the random Hadamard
            transform kernel, expert weights per expert: **89.5 ms/iter**.

            **Forward: 33.2 ms**

            .. raw:: html
               :file: profile_kimi_kda_nvfp4_fwd.svg

            **Backward: 56.7 ms**

            .. raw:: html
               :file: profile_kimi_kda_nvfp4_bwd.svg

   .. tab:: MLA layer

      .. tabs::

         .. tab:: BF16

            No quantization; the experts run as BF16 grouped GEMMs with a separate
            SiTU-GLU kernel: **113.9 ms/iter**.

            **Forward: 44.4 ms**

            .. raw:: html
               :file: profile_kimi_mla_bf16_fwd.svg

            **Backward: 70.4 ms**

            .. raw:: html
               :file: profile_kimi_mla_bf16_bwd.svg

         .. tab:: FP8 current scaling

            ``te.autocast(recipe=Float8CurrentScaling())``: per-tensor FP8 with an
            amax reduction before every cast, unfused grouped GEMMs: **87.4 ms/iter**.

            **Forward: 35.1 ms**

            .. raw:: html
               :file: profile_kimi_mla_fp8_cs_fwd.svg

            **Backward: 53.0 ms**

            .. raw:: html
               :file: profile_kimi_mla_fp8_cs_bwd.svg

         .. tab:: MXFP8 (fused grouped MLP)

            ``te.autocast(recipe=MXFP8BlockScaling())`` with the cuDNN fused grouped
            MLP: **61.9 ms/iter**.

            **Forward: 23.4 ms**

            .. raw:: html
               :file: profile_kimi_mla_mxfp8_fwd.svg

            **Backward: 38.3 ms**

            .. raw:: html
               :file: profile_kimi_mla_mxfp8_bwd.svg

         .. tab:: NVFP4 (fused grouped MLP)

            ``te.autocast(recipe=NVFP4BlockScaling())`` with the cuDNN fused grouped
            MLP; inputs and gradients are quantized by the random Hadamard
            transform kernel, expert weights per expert: **48.0 ms/iter**.

            **Forward: 19.1 ms**

            .. raw:: html
               :file: profile_kimi_mla_nvfp4_fwd.svg

            **Backward: 28.8 ms**

            .. raw:: html
               :file: profile_kimi_mla_nvfp4_bwd.svg

The GPU is idle for at most 0.3 ms per iteration in every trace. The MoE
dominates both layer types; the KDA layer is slower than the MLA layer by the
PyTorch side of the KDA block (float32 short convolution, casts, gated output
norm), not by the cuDNN KDA kernel itself.

Usage
-----

Because the AttnRes highway carries state between layers,
``KimiK3Layer.forward`` takes and returns the committed block residuals
alongside the hidden states. ``layer_number`` (1-based) decides which layers
commit a block. The routed experts of the MoE layers are sharded over all
GPUs with NCCL expert parallelism; one process per GPU, launched with
``torchrun``:

.. code-block:: python

    import os
    import torch
    import torch.distributed as dist
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.ep import ep_bootstrap, ep_finalize
    from transformer_engine.pytorch.models import DeepSeekV3MoE, KimiK3Layer

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    ep_group = dist.new_group(backend="nccl")
    ep_size = dist.get_world_size(ep_group)

    hidden, latent, topk = 7168, 3584, 16
    num_local_experts = 112           # 896 experts on 8 GPUs
    num_experts = num_local_experts * ep_size
    tokens_per_rank = 4096

    # The routed experts run on the latent, so the EP buffers are sized by it.
    ep_bootstrap(
        ep_group,
        num_experts=num_experts,
        max_tokens_per_rank=tokens_per_rank,
        hidden_dim=latent,
        num_topk=topk,
        recv_capacity_per_rank=DeepSeekV3MoE.ep_recv_capacity(
            ep_size, tokens_per_rank, topk, num_local_experts
        ),
    )

    def make_layer(layer_number):
        attention_type = "mla" if layer_number % 4 == 0 else "kda"
        attention_kwargs = {} if attention_type == "kda" else dict(
            q_lora_rank=1536, kv_lora_rank=512,
            qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        )
        mlp_kwargs = (
            dict(ffn_hidden_size=33792)
            if layer_number == 1
            else dict(
                num_experts=num_experts, moe_ffn_hidden_size=3072, topk=topk,
                latent_size=latent, shared_expert_ffn_hidden_size=6144,
                ep_group=ep_group, ep_max_tokens_per_rank=tokens_per_rank,
            )
        )
        return KimiK3Layer(
            hidden_size=hidden,
            num_attention_heads=96,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_res_block_size=12,
            params_dtype=torch.bfloat16,
            **attention_kwargs,
            **mlp_kwargs,
        )

    layers = torch.nn.ModuleList(make_layer(i + 1) for i in range(num_layers))
    moe_layers = layers[1:]
    ep_buffers = [layer.mlp.make_ep_buffer() for layer in moe_layers]

    x = torch.randn(tokens_per_rank // 4, 4, hidden, dtype=torch.bfloat16, device="cuda")
    for step in range(num_steps):
        hidden_states, block_residuals = x, None
        with te.autocast(recipe=te.MXFP8BlockScaling()):
            for i, layer in enumerate(layers):
                hidden_states, block_residuals = layer(
                    hidden_states,
                    block_residuals=block_residuals,
                    ep_buffer=ep_buffers[i - 1] if i > 0 else None,
                )
        hidden_states.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        # Quantile balancing: pool the required-bias histograms of all ranks,
        # then replace the biases. Every rank ends up with the same bias.
        for layer in moe_layers:
            dist.all_reduce(layer.mlp.routed_experts.qb_histogram, group=ep_group)
            layer.mlp.update_expert_bias()

    ep_finalize()
    dist.destroy_process_group()

.. code-block:: bash

    torchrun --nproc-per-node=8 train.py

Notes:

* ``ep_bootstrap`` is called once per process with the expert input size,
  which for the latent MoE is ``latent_size``, not ``hidden_size``.
* The KDA layers and the MLA layers hold no expert state and need nothing
  from the EP group. The MLA layers accept ``tp_group`` / ``tp_size``; the KDA
  block does not support tensor parallelism yet.
* The histogram all-reduce covers every rank whose tokens were routed by the
  same bias, i.e. the EP group plus any data-parallel replicas. Without
  ``ep_group`` the layer runs all experts locally, as in the tests.
