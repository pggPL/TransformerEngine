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
  projections, the causal short convolution and the gated output norm around
  it.
* :class:`~transformer_engine.pytorch.models.MultiLatentAttention` from the
  DeepSeek-V3 layer in its NoPE variant (``rope=False``) with a sigmoid output
  gate (``output_gate=True``): the same cuDNN fused attention with
  asymmetric head sizes and fused latent RMSNorms.

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

One ``KimiK3Layer`` (layer 2 of the model: attention plus latent MoE) with
Kimi-K3 dimensions (hidden 7168, 96 heads x 128, MLA 1536 / 512 / 128 / 64 /
128 NoPE, latent 3584, expert ffn 3072, top-k 16, shared expert ffn 6144) on
one GB200 node, 4 GPUs, expert parallelism over all four with 16 local experts
each (64 in total; the model has 896), 16384 tokens per GPU, forward + backward,
bf16 parameters, wall-clock with CUDA synchronization. With no committed block
residuals the AttnRes pooling is the identity, so the traces hold the
attention block and the MoE. Reproduce with ``examples/pytorch/kimi_k3``.

.. list-table::
   :header-rows: 1

   * - Layer (ms per iteration, fwd + bwd)
     - BF16
     - MXFP8, fused grouped MLP
   * - KDA layer
     - 160.2 ms (0.41 Mtok/s)
     - 107.1 ms (0.61 Mtok/s)
   * - MLA layer
     - 111.2 ms (0.59 Mtok/s)
     - 60.5 ms (1.08 Mtok/s)

The ``nsys`` traces of rank 0 below are redrawn as one iteration, forward and
backward pass as separate images on the same time scale. Every rectangle is
one kernel on the compute stream at its measured time; gaps are GPU idle time.
Hover a kernel for its group, name and duration; the images scroll sideways.

.. tabs::

   .. tab:: KDA layer (BF16)

      160.2 ms per iteration, 0.41 Mtok/s. Without the fused grouped MLP the SiTU-GLU of the experts (14 ms forward, 20 ms backward) is the single largest group, ahead of the dense projections and the grouped GEMMs; the KDA kernel itself is 5 ms of the backward and under 1 ms of the forward.

      **Forward:** 58.5 ms

      .. container:: scroll-x

         .. raw:: html
            :file: profile_kimi_kda_bf16_fwd.svg

      **Backward:** 102.6 ms

      .. container:: scroll-x

         .. raw:: html
            :file: profile_kimi_kda_bf16_bwd.svg

   .. tab:: KDA layer (MXFP8, fused grouped MLP)

      107.1 ms per iteration, 0.61 Mtok/s. The MoE costs the same as in the
      MLA layer; the difference is the PyTorch side of the KDA block (float32
      short convolution, layout copies, casts and the gated output norm),
      while the KDA kernel itself takes 6 ms.

      **Forward:** 40.0 ms

      .. container:: scroll-x

         .. raw:: html
            :file: profile_kimi_kda_mxfp8_fwd.svg

      **Backward:** 71.5 ms

      .. container:: scroll-x

         .. raw:: html
            :file: profile_kimi_kda_mxfp8_bwd.svg

   .. tab:: MLA layer (BF16)

      111.2 ms per iteration, 0.59 Mtok/s. Same MoE as the KDA layer; the attention block costs 1.4 ms forward and 4.9 ms backward in cuDNN fused attention plus the MLA projections, so the layer is about 90 % MoE.

      **Forward:** 44.9 ms

      .. container:: scroll-x

         .. raw:: html
            :file: profile_kimi_mla_bf16_fwd.svg

      **Backward:** 64.3 ms

      .. container:: scroll-x

         .. raw:: html
            :file: profile_kimi_mla_bf16_bwd.svg

   .. tab:: MLA layer (MXFP8, fused grouped MLP)

      60.5 ms per iteration, 1.08 Mtok/s. The fused grouped MLP replaces 30 ms of grouped GEMMs and 32 ms of SiTU-GLU with 18 ms, at the cost of 3.8 ms of MXFP8 quantization; the two NCCL EP exchanges per pass take 3.2 ms in total.

      **Forward:** 23.4 ms

      .. container:: scroll-x

         .. raw:: html
            :file: profile_kimi_mla_mxfp8_fwd.svg

      **Backward:** 36.9 ms

      .. container:: scroll-x

         .. raw:: html
            :file: profile_kimi_mla_mxfp8_bwd.svg

Across the four traces the GPU is idle for at most 0.2 ms per iteration, so
the timelines are kernel time. The MoE dominates both layer types: the
expert SiTU-GLU is not fused in BF16 and costs as much as the grouped GEMMs
themselves, which is what the fused cuDNN grouped MLP removes in MXFP8. The
KDA layer is 1.5-1.8x slower than the MLA layer, and the difference is not
the recurrence: the cuDNN KDA kernel accounts for 6 ms of the 111 ms MXFP8
iteration, while the PyTorch-side pieces of the block (the float32 short
convolution, the casts around it, the gated per-head RMSNorm) account for
about 47 ms. Those are the next fusion targets for this layer.

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
