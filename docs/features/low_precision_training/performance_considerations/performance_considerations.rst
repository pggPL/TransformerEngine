..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Performance Considerations
===================================

.. _handling_transposes:

Handling transposes
-------------------

During the backward pass of a Linear layer, matrix multiplication requires transposed versions of both the input and weight tensors. 
Specifically:

* To compute the weight gradient ``dW``, we need the transposed input: ``dW = input^T × output_grad``
* To compute the input gradient ``dX``, we need the transposed weight: ``dX = output_grad × weight^T``

Hopper and previous devices need to have physically transposed tensor in memory to perform such a GEMM.
Blackwell and later devices support more GEMM layouts, so the tensor does not need to be physically transposed.
However, for some recipes the quantization needs to be different for both of these tensors.
Thus we will not use names "transposed" and "non-transposed" anymore, but rather *columnwise* and *rowwise* usages.
It may happen that both of them will refer to the same tensor - like in FP8 on Blackwell.

Note that performing rowwise and columnwise quantization at the same time
allows to perform some fusions, which usually lead to better performance.
We showcase 3 example scenarios of producing quantized tensors in rowwise and columnwise usages,
TE will use best possible fusion for given recipe and TE module configuration:

1. Computation of quantized tensor in both rowwise and columnwise usages in a single kernel in forward pass. This is the fastest one,
   but since the columnwise usage is saved for backward pass, it may lead to increased memory usage, 
   if high precision tensor also needs to be saved for backward - for example if it is attention output which is saved anyways.
2. Computation of quantized tensor in rowwise usage in forward pass and fused quantization to produce columnwise usage in backward pass. 
   This is usually slower than the previous one, since high precision tensor needs to be read twice.
   It is used for example when high precision tensor is gathered both in forward and in backward 
   and quantized tensor gather is not implemented for such recipe.
3. Computation of quantized tensor in rowwise usage in forward pass and transpose to columnwise usage in backward pass. 
   This is not possible for all recipes.

Transformer Engine uses the best possible fusion internally, so user do not need to worry about the details.
We showcase this issue in the documentation to understand memory consequences of different fusion scenarios.

.. raw:: html
   :file: img/transpose_fusion.svg

*Figure 4: Three scenarios of producing quantized tensors in rowwise and columnwise usages.*



Memory usage
------------

As mentioned earlier, Transformer Engine stores parameters in high precision 
and casts them to lower precision before the GEMM.
Optimizer step is applied by default to high precision parameters - 
applying optimizer to lower precision parameters directly can potentially lead
to accuracy degradation, thus this is not the default behavior.

One can see that low precision training will not decrease memory usage by default - 
we need to store high precision parameters of the model anyways. Moreover, TE by default
stores quantized weight in columnwise usage for backward pass - this adds additional memory usage.
Therefore, it's important to note that low precision training does not always lead to lower memory usage—the main 
objective is speed.

Note that for some use cases, this is not optimal - 
for example in inference high precision weights are not needed. \
Depending on framework, Transformer Engine 
has some mechanisms of using only low precision parameters and weight caching.

.. tabs::

   .. tab:: PyTorch

      Let's see how FP8 training affects memory usage in Transformer Engine for PyTorch.
      At first, the forward pass in higher precision is run.

      .. literalinclude:: memory_usage_pytorch.py
         :language: python
         :start-after: # START_MEMORY_USAGE_1
         :end-before: # END_MEMORY_USAGE_1
      
      .. container:: program-output
      
         .. literalinclude:: ../introduction/pytorch_out
            :language: text
            :start-after: # START_MEMORY_USAGE_1
            :end-before: # END_MEMORY_USAGE_1
      
      We can see that the layer size is ``1024 * 1024 * 2 (2 bytes per parameter) = 2MB``.
      Memory after forward pass is ``2 MB (weight) + 2 MB (input) + 2 MB (output) = 6 MB``.
      
      Now let's see what is the memory usage in FP8 training.

      
      .. literalinclude:: memory_usage_pytorch.py
         :language: python
         :start-after: # START_MEMORY_USAGE_2
         :end-before: # END_MEMORY_USAGE_2
      
      .. container:: program-output
      
         .. literalinclude:: ../introduction/pytorch_out
            :language: text
            :start-after: # START_MEMORY_USAGE_2
            :end-before: # END_MEMORY_USAGE_2
      
      Total memory usage is ``2 MB (weight) + 1 MB (weight in FP8) + 2 MB (input) + 1 MB (input in FP8) + 2 MB (output) = 8 MB``.
      
      .. literalinclude:: memory_usage_pytorch.py
         :language: python
         :start-after: # START_MEMORY_USAGE_3
         :end-before: # END_MEMORY_USAGE_3

      Total memory usage is ``1 MB (weight in FP8) + 2 MB (input) + 1 MB (input in FP8) + 2 MB (output) = 6 MB``.
      Note that columnwise usage of FP8 weight is not computed, since the layer was initialized with ``torch.no_grad()`` context.
      It will be computed on the first backward pass from the rowwise FP8 weight.

      .. container:: program-output
      
         .. literalinclude:: ../introduction/pytorch_out
            :language: text
            :start-after: # START_MEMORY_USAGE_3
            :end-before: # END_MEMORY_USAGE_3
      

   .. tab:: JAX

      Let's see how FP8 training affects memory usage in Transformer Engine for JAX.
      At first, the forward pass in higher precision is run.

      .. literalinclude:: memory_usage_jax.py
         :language: python
         :start-after: # START_MEMORY_USAGE_1
         :end-before: # END_MEMORY_USAGE_1
      
      .. container:: program-output
      
         .. literalinclude:: ../introduction/jax_out
            :language: text
            :start-after: # START_MEMORY_USAGE_1
            :end-before: # END_MEMORY_USAGE_1
      
      We can see that the layer size is ``1024 * 1024 * 2 (2 bytes per parameter) = 2MB``.
      Memory after forward pass is ``2 MB (weight) + 2 MB (input) + 2 MB (output) = 6 MB``.
      
      Now let's see what is the memory usage in FP8 training with master weights in BF16.

      
      .. literalinclude:: memory_usage_jax.py
         :language: python
         :start-after: # START_MEMORY_USAGE_2
         :end-before: # END_MEMORY_USAGE_2
      
      .. container:: program-output
      
         .. literalinclude:: ../introduction/jax_out
            :language: text
            :start-after: # START_MEMORY_USAGE_2
            :end-before: # END_MEMORY_USAGE_2
      
      Total memory usage is ``2 MB (weight) + 1 MB (weight in FP8) + 2 MB (input) + 1 MB (input in FP8) + 2 MB (output) = 8 MB``.
      
      Finally, let's initialize the layer with FP8 weights directly by using ``fp8_autocast`` during initialization.

      .. literalinclude:: memory_usage_jax.py
         :language: python
         :start-after: # START_MEMORY_USAGE_3
         :end-before: # END_MEMORY_USAGE_3

      .. container:: program-output
      
         .. literalinclude:: ../introduction/jax_out
            :language: text
            :start-after: # START_MEMORY_USAGE_3
            :end-before: # END_MEMORY_USAGE_3

      Total memory usage is ``1 MB (weight in FP8) + 2 MB (input) + 1 MB (input in FP8) + 2 MB (output) = 6 MB``.
      This approach reduces memory footprint by storing weights directly in FP8 format.

Fused layers
------------


Transformer Engine provides fused layers like ``LayerNormLinear`` and ``LayerNormMLP``, 
which enables various optimizations. Here, one of such optimizations is presented
- fusion of layer norm and quantization.

Quantizing the input before the GEMM operation can add noticeable overhead. 
For example, in a typical sequence where a LayerNorm is followed by a Linear layer, 
the LayerNorm produces an FP32 output, 
which must then be quantized by the Linear layer prior to the GEMM. 
Transformer Engine's fused LayerNormLinear layer optimizes this by combining 
the layer normalization and quantization operations, 
so the output from LayerNorm is returned in quantized form, 
reducing unnecessary quantization steps.


.. raw:: html
   :file: img/fused_layers.svg

*Figure 5: Comparison of separate LayerNorm and Linear layers versus fused LayerNormLinear layer, showing reduced quantization overhead.*


Let's see how we can use fused layers in different frameworks.

.. tabs::

   .. tab:: PyTorch

      In PyTorch, Transformer Engine provides fused layers like ``LayerNormLinear`` and ``LayerNormMLP``.
      These layers combine normalization and linear operations with optimized quantization.

      .. literalinclude:: fused_layers_pytorch.py
         :language: python
         :start-after: # START_FUSED_LAYERS
         :end-before: # END_FUSED_LAYERS
      
      The fused ``LayerNormLinear`` layer is particularly efficient in FP8 training because 
      it avoids an intermediate quantization step. The LayerNorm output is directly quantized 
      for the GEMM operation, reducing memory bandwidth and improving performance.

   .. tab:: JAX

      In JAX, Transformer Engine provides fused layers like ``LayerNormDenseGeneral`` and ``LayerNormMLP``.
      These layers combine normalization and dense operations with optimized quantization.

      .. literalinclude:: fused_layers_jax.py
         :language: python
         :start-after: # START_FUSED_LAYERS
         :end-before: # END_FUSED_LAYERS
      
      The fused ``LayerNormDenseGeneral`` layer is particularly efficient in FP8 training because 
      it avoids an intermediate quantization step. The LayerNorm output is directly quantized 
      for the GEMM operation, reducing memory bandwidth and improving performance.


Distributed training
--------------------

Transformer Engine supports distirbuted training and some of collective operations are done inside TE modules.
Since TE does not expose lower precision outside the modules, user usually does not need to be concerned
by the interplay of communication and low precision - most of the logic is handled internally.

As we said earlier, in most common scenarios, only GEMMs are affected by low precision. There are 6 tensors related
to each GEMM operation - weight, input, output, weight gradient, input gradient and output gradient.
Weight gradient, input gradient and output are returned in higher precision, so low precision training does not affect them.
Weights are not communicated inside TE modules in most scenarios -- there are exceptions like FSDP,
but they will not be covered in this part of the documentation.

So there are 2 tensors left: input and output gradient. For tensor parallel, communication
is performed in high precision. The only case in which tensors are communicated in lower precision 
is all gather related to the sequence parallelism.

Transformer Engine supports all-gather of quantized tensors for most recipes. 
The extent of this support may differ depending on the recipe - for example gathering of
columnwise quantized tensors is not always supported. There are multiple benefits
of using quantized all-gather, benefiting from the fact that quantized tensor is smaller than high precision tensor:

1. Reduced memory usage - no need to store high precision tensor for backward pass.
2. Reduced communication overhead - no need to gather high precision tensor.
3. Reduced computation overhead - quantization is parallelized between multiple GPUs.


Below we present all-gather scenarios for input and gradient tensors showing
rowwise and columnwise quantized tensors. Note that
they may vary depending on recipe and TE modules configuration.

.. raw:: html
   :file: img/sequence_parallel_quantization.svg

*Figure 6: All-gather of quantized tensors for input and gradient tensors. 
Note that this is one scenario and it may vary depending on the recipe and TE modules configuration.*


