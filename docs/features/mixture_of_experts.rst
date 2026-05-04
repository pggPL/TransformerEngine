..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Mixture of Experts
==================

Mixture of Experts (MoE) layers replace a dense feed-forward network with a set
of expert networks and a router that sends each token to one or more experts.
This keeps the activated parameter count per token small while allowing the
model to scale to many more total parameters.

Efficient MoE execution is mostly about data movement and batching. Tokens must
be dispatched so that tokens assigned to the same expert are contiguous, expert
linear layers must be executed without launching many tiny GEMMs, and expert
outputs must be combined back into the original token order. Transformer Engine
provides optimized building blocks for these steps.

Grouped Linear
--------------

``GroupedLinear`` applies several independent linear transformations in one
grouped GEMM call. In an MoE layer, each group typically corresponds to one
expert, and the input is the concatenation of the token blocks routed to those
experts. The ``m_splits`` argument passed to ``forward`` gives the number of
tokens in each expert block.

This is equivalent to splitting the input along the token dimension, applying a
separate linear layer to each split, and concatenating the outputs, but it avoids
the overhead of launching one GEMM per expert. This is especially important when
token counts per expert are small or imbalanced.

The PyTorch module is available as
``transformer_engine.pytorch.GroupedLinear``. The operation-fuser variant is
available as ``transformer_engine.pytorch.ops.GroupedLinear`` for users building
custom fused graphs.

``GroupedLinear`` supports Transformer Engine's low precision execution paths,
including FP8 autocast where supported. Tensor-parallel weight shapes can be
configured through the module arguments, but MoE dispatch and combine
communications are handled outside of the module.

Permutation Kernels
-------------------

Permutation kernels implement the token dispatch and combine stages around the
expert computation. Dispatch takes the original token tensor and a routing map,
then produces an expert-contiguous tensor suitable for grouped GEMMs. Combine
takes the expert outputs, restores the original token order, and optionally
merges multiple expert contributions using router probabilities.

For PyTorch, Transformer Engine exposes:

* ``transformer_engine.pytorch.moe_permute`` for token dispatch.
* ``transformer_engine.pytorch.moe_permute_with_probs`` for dispatching tokens
  and router probabilities together.
* ``transformer_engine.pytorch.moe_permute_and_pad_with_probs`` for dispatching
  with per-expert padding to satisfy alignment requirements.
* ``transformer_engine.pytorch.moe_unpermute`` for combining expert outputs.
* ``transformer_engine.pytorch.moe_sort_chunks_by_index`` and
  ``transformer_engine.pytorch.moe_sort_chunks_by_index_with_probs`` for
  reordering already chunked token buffers.

For JAX, the same dispatch/combine pattern is available through
``transformer_engine.jax.permutation.token_dispatch``,
``transformer_engine.jax.permutation.token_combine``, and
``transformer_engine.jax.permutation.sort_chunks_by_index``.

The permutation APIs support routing maps that describe which experts receive
each token. PyTorch supports both mask-based maps and index-based maps for
``moe_permute``/``moe_unpermute``; probability-aware and padding-aware variants
use mask-based routing maps. JAX dispatch accepts mask-based routing maps and can
compute padding internally when an alignment size is provided.
