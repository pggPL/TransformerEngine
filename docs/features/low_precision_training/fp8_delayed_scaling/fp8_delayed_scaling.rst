..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

FP8 Delayed Scaling
===================================

FP8 Delayed Scaling is historically the first recipe provided by Transformer Engine.
It is the improvement on FP8 Current Scaling recipe, which aims to provide better performance,
with potentially higher quantization error.

These recipes both use the same data formats, and the only substantial difference is in quantization process.

Let's remind us how quantization in FP8 Current Scaling works. To quantize a tensor to FP8,
two tensor reads were needed - one to compute the amax, and one to apply the scaling factor and cast to FP8.
The core idea of delayed scaling is not to compute the amax on the fly,
but rather try to estimate it from the amax history. This is the origin of the names *current scaling*
and *delayed scaling*. The first one uses the current amax value, 
while the second one uses the history of amax values to compute the scaling factor.

Quantization Process
--------------------

The only substantial difference is in quantization process. In FP8 Delayed Scaling the steps are:

1. Each module stores ``amax_history`` tensor for each of the tensors in each of the GEMMs.
   Size of this tensor is ``amax_history_len``, which is a hyperparameter of the recipe.
2. The scaling factor is computed from the ``amax_history`` tensor using one of two predefined algorithms:
   ``max`` (default, uses maximum from entire history) or ``most_recent`` (uses only the most recent value).
   Alternatively, a custom callable can be provided for custom amax computation logic.
   The formula is ``new_scaling_factor = (FP8_MAX / amax) / (2 ^ margin)``, 
   where ``FP8_MAX`` is the maximum representable value of the FP8 data format, 
   and ``margin`` is a hyperparameter of the recipe - default is 0. Note that this does not need a tensor read.
3. The quantization is performed using the computed scaling factor. 
   If some value exceeds fp8 range, it is clipped to FP8_MAX or -FP8_MAX.
   needs one tensor read.
4. The ``amax_history`` tensor is updated with the new amax value.

Note that only one tensor read is needed in the quantization process, 
which is an improvement over FP8 Current Scaling 
where two tensor reads were needed.

.. raw:: html
   :file: img/scaling_comparison.svg

*Figure 1. Comparison of FP8 Current Scaling and FP8 Delayed Scaling quantization processes.*

Amax History Management
-----------------------

Let's look closer how amax history is stored and updated.

1. For each module initialization, the two ``amax_history`` tensors are created and initialized to 0 
   (one for tensors used in forward pass, and one for tensors used in backward pass).
2. Each quantization updates element 0 of the ``amax_history`` tensor with the new amax value.
   During the first forward pass in an autocast context, each module registers both its forward 
   and backward ``amax_history`` tensors to global buffers corresponding to that autocast context 
   (identified by recipe and distributed group). This registration happens only once - subsequent 
   forward passes reuse the same registered tensors.
3. After the exit from the autocast context,
   amax history tensors related to tensors used in forward pass are rotated by -1: 
   the first entry shifts to the last, the last entry shifts to the second to last, etc.
   Then the first element is set to 0.
   This is done by one custom CUDA kernel - which is called once per autocast context.
   Here also the reduction of amax between different GPUs is performed, more details in the next section.
4. In backward pass, each quantization updates element 0 of the ``amax_history`` tensor with the new amax value.
   The backward ``amax_history`` tensors are already registered in global buffers from step 2.
5. After the last backward pass for a TE module in the autocast context completes, 
   the amax history tensors for backward pass are rotated as in step 3.
  

Here's how to use FP8 Delayed Scaling in PyTorch and JAX:

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: pytorch_delayed_scaling_example.py
         :language: python
         :start-after: # START_DELAYED_SCALING_EXAMPLE
         :end-before: # END_DELAYED_SCALING_EXAMPLE

   .. tab:: JAX

      .. literalinclude:: jax_delayed_scaling_example.py
         :language: python
         :start-after: # START_DELAYED_SCALING_EXAMPLE
         :end-before: # END_DELAYED_SCALING_EXAMPLE


Distributed Training
--------------------

Since FP8 Delayed Scaling uses the same data formats as FP8 Current Scaling,
similarly as in the delayed scaling case, the transpose gather
is not supported. Altough, the amax reduction works slightly differently in different frameworks.

**Configuration examples:**

.. tabs::

   .. tab:: PyTorch

      One can enable or disable amax reduction across all GPUs by setting ``reduce_amax`` parameter in the recipe.
      Note that sequence parallelism and context parallelism require amax reduction to be enabled.

      Group on which amax reduction is performed is specified in the ``fp8_group`` parameter of ``fp8_autocast``.
      We advise to reduce amax across all GPUs for which tensor is sharded - across data parallel too.

      .. literalinclude:: pytorch_delayed_scaling_distributed_example.py
         :language: python
         :start-after: # START_AMAX_REDUCTION_EXAMPLE
         :end-before: # END_AMAX_REDUCTION_EXAMPLE


   .. tab:: JAX

      For JAX, amax reduction is always enabled in FP8 Delayed Scaling.
      The scope of this reduction is all parallelism axes except pipeline parallelism (TP, SP, DP/FSDP).
      This is managed automatically internally.

      .. literalinclude:: jax_delayed_scaling_distributed_example.py
         :language: python
         :start-after: # START_AMAX_REDUCTION_EXAMPLE
         :end-before: # END_AMAX_REDUCTION_EXAMPLE

Supported devices
-----------------

Ada and later (SM 8.9+)