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
  in the dense MLP and shared expert, both fused forward and backward kernels;
  the :ref:`fused grouped MLP <moe-grouped-mlp>` supports SiTU-GLU as well.
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

Usage
-----

Because the AttnRes highway carries state between layers,
``KimiK3Layer.forward`` takes and returns the committed block residuals
alongside the hidden states. ``layer_number`` (1-based) decides which layers
commit a block.

.. code-block:: python

    import torch
    from transformer_engine.pytorch.models import KimiK3Layer

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
                num_experts=896, moe_ffn_hidden_size=3072, topk=16,
                latent_size=3584, shared_expert_ffn_hidden_size=6144,
            )
        )
        return KimiK3Layer(
            hidden_size=7168,
            num_attention_heads=96,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_res_block_size=12,
            params_dtype=torch.bfloat16,
            **attention_kwargs,
            **mlp_kwargs,
        )

    layers = torch.nn.ModuleList(make_layer(i + 1) for i in range(8))

    hidden = torch.randn(4096, 1, 7168, dtype=torch.bfloat16, device="cuda")
    block_residuals = None
    for layer in layers:
        hidden, block_residuals = layer(hidden, block_residuals=block_residuals)
    hidden.sum().backward()
    for layer in layers[1:]:
        layer.mlp.update_expert_bias()  # quantile balancing, once per step

Expert parallelism works as for the DeepSeek-V3 layer: pass ``ep_group`` and
``ep_max_tokens_per_rank``, and forward an ``ep_buffer`` if you want to
reuse one across layers.

Where the time goes
-------------------

The KDA kernel requires a Blackwell GPU, so timings of the Kimi-K3 layer will
be added once the layer has been measured there. The MoE and the MLA layers
share their kernels with the :doc:`DeepSeek-V3 layer <deepseek_v3>`, whose
measurements apply to those parts.
