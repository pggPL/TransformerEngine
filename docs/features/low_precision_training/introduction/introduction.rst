..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Introduction
===================================

The main feature of Transformer Engine is enabling low precision training. 
While the standard floating-point format on CPUs is FP32, 
NVIDIA GPUs support lower precision formats designed to accelerate training.
In this chapter, we introduce the general concepts of low precision training support in Transformer Engine.


Training in BF16/FP16
---------------------

NVIDIA GPUs support FP16 format since Pascal generation and BF16 format since Ampere generation.
Let's compare the differences between these two formats and FP32 format.

.. raw:: html
   :file: img/fp_formats_comparison.svg

*Figure 1: Comparison of FP32, BF16, and FP16 floating-point formats showing bit allocation for sign, exponent, and mantissa.*

The key differences between these formats are:

* **FP32** (32 bits total): 1 sign bit + 8 exponent bits + 23 mantissa bits - standard single-precision format
* **BF16** (16 bits total): 1 sign bit + 8 exponent bits + 7 mantissa bits - maintains FP32's exponent range but reduced precision
* **FP16** (16 bits total): 1 sign bit + 5 exponent bits + 10 mantissa bits - reduced range but higher precision than BF16

BF16's advantage is that it shares the same exponent range as FP32, 
making it easier to convert between the two formats without overflow/underflow issues. 
FP16 offers better precision for smaller values but has a more limited dynamic range,
which results in the need to perform loss scaling to avoid overflow/underflow—see `this paper on loss scaling <https://arxiv.org/pdf/1710.03740>`__ for more details.


Not all operations can be performed in reduced precision. In practice, modern deep learning frameworks use *mixed precision training*,
where computationally intensive operations like matrix multiplications run in low precision, 
while numerically sensitive operations remain in high precision. 
Transformer Engine also always runs some operations in FP32 precision - like normalization layers.
Moreover, usually optimizer state is also stored in FP32 precision.

Another issue is whether to keep FP32 master weights and cast them between every operation or use lower precision weights.
The motivation for using master weights in FP32 is that sum of low precision weight with low precision update tensor 
may lead to numerical instability.

Transformer Engine supports FP32 and FP16/BF16 mixed precision training. It also supports using master weights in FP32 precision
and using lower precision weights.

.. tabs::

   .. tab:: PyTorch

      In PyTorch one can set weights precision by using ``params_dtype`` argument in any TE layer constructor.
      To run computation in lower precision, one can use ``torch.cuda.amp.autocast`` context manager.
      If parameters are set to be in lower precision and no autocast is used, then lower precision is used for computation.
      Input is casted to lower precision before the computation inside the layer.
      Output precision is the same as autocast precision.

      .. literalinclude:: bf16_fp16_training_pytorch.py
         :language: python
         :start-after: # START_BF16_FP16_TRAINING
         :end-before: # END_BF16_FP16_TRAINING
      

   .. tab:: JAX

      In JAX one can set weights precision by using ``dtype`` argument in any TE layer constructor.
      Computation is performed in precision of the input tensor.
      So for training with master weights in FP32 and computation in BF16, 
      one needs to cast input to BF16.

      .. literalinclude:: bf16_fp16_training_jax.py
         :language: python
         :start-after: # START_BF16_FP16_TRAINING
         :end-before: # END_BF16_FP16_TRAINING
      


Even lower precisions (FP8 etc.)
------------------------------------

BF16/FP16 is not the smallest precision you get used to. Transformer Engine supports training in even lower precisions - like for example FP8.
The logic of FP8 and other 8- or 4-bit precision is more complicated than BF16/FP16.
Transformer Engine abstracts this complexity by recipe objects and autocast context manager.
Let's look how fp8 training works in supported frameworks.

.. tabs::

   .. tab:: PyTorch

      For PyTorch one needs to define recipe object.
      Then run forward pass inside `autocast` context manager with proper recipe as an argument.

      .. literalinclude:: fp8_autocast_pytorch.py
         :language: python
         :start-after: # START_FP8_AUTOCAST
         :end-before: # END_FP8_AUTOCAST
      

   .. tab:: JAX

      For JAX FP8 training:
      
      1. Set up ``global_shard_guard(MeshResource())`` context (required even for single GPU)
      2. Create FP8 recipe using ``get_delayed_scaling_recipe()``
      3. Wrap model initialization and training in ``te.fp8_autocast(enabled=True, recipe=recipe, mesh_resource=...)``
      
      Model initialization must happen inside the autocast context 
      to properly capture FP8 metadata in the parameter tree.

      .. literalinclude:: fp8_autocast_jax.py
         :language: python
         :start-after: # START_FP8_AUTOCAST
         :end-before: # END_FP8_AUTOCAST

The diagram below shows which operations in a Transformer layer run in lower precision versus higher precision by default:

.. raw:: html
   :file: img/mixed_precision_operations.svg

*Figure 2: Default single-device forward pass of TransformerLayer operations precision - only GEMMs are in lower precision.*

We can see that mostly linear operations benefit from running in lower precision. Let's see how 
this layers work internally.

.. raw:: html
   :file: img/fp8_linear_flow.svg

*Figure 3: Single-device forward pass of Linear layer data flow showing quantization and dequantization steps.*

We can see that:

1. Weights are stored in higher precision and cast to lower precision before the GEMM.
2. Input is cast to lower precision before the GEMM.
3. Output is in higher precision.
4. Output gradient is casted into lower precision.
5. Weight and input in both rowwise and columnwise usages are in lower precision.
6. Gradient of weight and input are returned in higher precision.

