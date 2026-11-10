..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Jax
===

Pre-defined Variable of Logical Axes
------------------------------------
Variables are available in `transformer_engine.jax.sharding`.

* BATCH_AXES: The logical axis of batch dimension. It is usually sharded along DP + FSDP on Mesh.
* SEQLEN_AXES: The logical axis of sequence length dimension. It is usually not sharded.
* SEQLEN_TP_AXES: The logical axis of sequence length dimension. It is usually sharded along TP on Mesh.
* HEAD_AXES: The logical axis of head dimension of MHA. It is usually sharded along TP on Mesh.
* HIDDEN_AXES: The logical axis of hidden dimension. It is usually not sharded.
* HIDDEN_TP_AXES: The logical axis of hidden dimension. It is usually sharded along TP on Mesh.
* JOINED_AXES: The logical axis of non-defined dimension. It is usually not sharded.


Checkpointing
-------------
When using checkpointing with Transformer Engine JAX, please be aware of the checkpointing policy being applied to your model. Any JAX checkpointing policy using `dot`, such as `jax.checkpoint_policies.dots_with_no_batch_dims`, may not work with GEMMs provided by Transformer Engine as they do not always use the `jax.lax.dot_general` primitive. Instead, you can use `transformer_engine.jax.checkpoint_policies.dots_and_te_gemms_with_no_batch_dims` or similar policies that are designed to work with Transformer Engine's GEMMs and `jax.lax.dot_general` GEMMs. You may also use any JAX policies that do not filter by primitive, such as `jax.checkpoint_policies.save_only_these_names` or `jax.checkpoint_policies.everything_saveable`.

Modules
-------
.. autoapiclass:: transformer_engine.jax.flax.TransformerLayerType
.. autoapiclass:: transformer_engine.jax.MeshResource()


.. autoapifunction:: transformer_engine.jax.fp8_autocast
.. autoapifunction:: transformer_engine.jax.autocast
.. autoapifunction:: transformer_engine.jax.update_collections
.. autoapifunction:: transformer_engine.jax.grouped_situglu


.. autoapiclass:: transformer_engine.jax.flax.LayerNorm(epsilon=1e-6, layernorm_type='layernorm', **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.DenseGeneral(features, layernorm_type='layernorm', use_bias=False, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.LayerNormDenseGeneral(features, layernorm_type='layernorm', epsilon=1e-6, use_bias=False, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.LayerNormMLP(intermediate_dim=2048, layernorm_type='layernorm', epsilon=1e-6, use_bias=False, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.RelativePositionBiases(num_buckets, max_distance, num_heads, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.DotProductAttention(head_dim, num_heads, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.MultiHeadAttention(head_dim, num_heads, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.TransformerLayer(hidden_size=512, mlp_hidden_size=2048, num_attention_heads=8, **kwargs)
  :members: __call__

.. autoapifunction:: transformer_engine.jax.flax.extend_logical_axis_rules


Mixture of Experts
------------------
Router, routing kernels, and grouped dense for MoE layers. See
:doc:`Mixture of Experts <../features/mixture_of_experts/mixture_of_experts>` for an overview.

.. autoapifunction:: transformer_engine.jax.router.fused_topk_with_score_function

.. autoapifunction:: transformer_engine.jax.router.fused_topk_with_score_function_qb

.. autoapifunction:: transformer_engine.jax.router.fused_moe_aux_loss

.. autoapifunction:: transformer_engine.jax.permutation.sort_chunks_by_index

.. autoapifunction:: transformer_engine.jax.dense.grouped_dense

.. autoapiclass:: transformer_engine.jax.moe.SiTUActivation

``SiTUActivation`` uses native grouped activation kernels between the expert
GEMMs, including their backward pass. Device group counts bound active rows;
unused receive capacity is zero-filled without evaluating the activation there.
FP32, FP16 and BF16 activation outputs are supported. MXFP8 expert execution
uses grouped quantization after the activation. The standalone
``grouped_situglu(x, group_sizes, beta1=4.0, beta2=25.0)`` function has the same
math and automatic differentiation support. It expects contiguous gate/up
halves in each row of ``x[capacity, 2H]``. Counts must be nonnegative device
int32 values with a sum no greater than capacity. Distributed expert callers
should pass local arrays and counts inside ``shard_map``.

.. autoapifunction:: transformer_engine.jax.moe.moe

.. autoapiclass:: transformer_engine.jax.moe.RouterComputationInfo

Latent MoE models can pass full-width hidden states through
``RouterComputationInfo(router_input=hidden_states, ...)`` while calling
``moe(latent_states, router_info, ...)`` with narrower expert activations.
Batch and sequence dimensions must agree. Router parameters use the router
width; expert parameters and EP bootstrap buffers use the latent width.
The custom VJP returns gradients to both inputs, including when the latent
is a differentiable projection of the router input. The default
``router_input=None`` preserves routing on the expert input.

Quantile Balancing statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``RouterComputationInfo.qb_bin_bounds`` to an FP32 ``[lower, upper]`` array
to collect QB statistics during MoE routing. Sigmoid routing, an FP32 expert
bias, no routing groups and zero auxiliary loss are required. The existing CUDA
QB kernels provide ``fused_atomic`` and ``two_kernel`` modes, selected by
``qb_histogram_mode``. The number of bins defaults to 1000.

The bounds must be finite and ordered; their values are read on device without
host validation. Kimi uses ``[min(bias)-1, max(bias)+1]``. Keep both bias and
bounds unchanged throughout every microbatch in a training step. With no mask, all token rows must be valid. Optional ``qb_token_mask``
(``[batch, sequence]``) excludes padding: CUDA computes scores and cutoff,
while a JAX histogram counts only valid rows. This masked path does not use
CUDA histogram accumulation. The standalone router accepts ``token_mask``
with shape ``[tokens]`` for the same behavior.

With QB enabled, ``moe`` returns a fifth result, an int32 histogram array of
shape ``[token_shards, experts, bins]``. Its leading axis is sharded across the
compound DP/FSDP/EP token axes. The forward does not reduce these histograms
or update biases. Accumulate per-shard counts across microbatches and only then
sum the leading axis to obtain each expert's global-batch distribution. The
training framework derives and mean-centers the new bias from that distribution,
applying it on the next step. Default calls keep the four-result API.

The standalone QB router returns ``(probs, routing_map, histogram)`` and accepts
an optional initial histogram for functional accumulation. Its FFI declares
input/output aliasing for the CUDA kernel's in-place histogram writes. For SPMD,
place the standalone operation inside ``shard_map`` with the expert axis
replicated and expose local histograms as a sharded leading axis, as MoE does.
Neither histogram collection nor bias/bounds inputs have a gradient. The router
reuses the ordinary sigmoid routing backward for gradients through its weights.
Use fresh forward statistics or explicit scan carry state; an external mutable
histogram would be counted twice when activation checkpointing recomputes it.

The Flax wrapper accepts ``qb_bin_bounds`` on each call and exposes
``qb_num_bins`` and ``qb_histogram_mode`` as module configuration. It returns
the same optional fifth result. Step-level bias updates, checkpointing that
bias state, and TPU implementations belong to the training framework.

.. autoapifunction:: transformer_engine.jax.ep.ep_bootstrap

.. autoapifunction:: transformer_engine.jax.ep.ep_finalize

.. autoapifunction:: transformer_engine.jax.ep.ep_dispatch

.. autoapifunction:: transformer_engine.jax.ep.ep_combine
