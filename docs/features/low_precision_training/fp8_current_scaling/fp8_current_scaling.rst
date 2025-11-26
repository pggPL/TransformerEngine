..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

FP8 Current Scaling
===================================

FP8 current scaling is the simplest low precision recipe provided by Transformer Engine. 
Let's start with what the FP8 data type in fact is.


FP8 data type
-------------

The FP8 datatype, introduced in Ada architecture, is actually 2 distinct datatypes, useful in different parts of the training of neural networks:

* E4M3 - it consists of 1 sign bit, 4 exponent bits and 3 bits of mantissa. It can store values up to +/-448 and `nan`.
* E5M2 - it consists of 1 sign bit, 5 exponent bits and 2 bits of mantissa. It can store values up to +/-57344, +/- `inf` and `nan`. The tradeoff of the increased dynamic range is lower precision of the stored values.

.. raw:: html
   :file: img/fp8_formats.svg

*Figure 1: Structure of the floating point datatypes. All of the values shown (in FP16, BF16, FP8 E4M3 and FP8 E5M2) are the closest representations of value 0.3952.*


During neural network training both of these types may be utilized. 
Typically forward activations and weights require more precision, so E4M3 datatype is best used during forward pass. 
In the backward pass, however, gradients flowing through the network typically are less susceptible to the loss of precision, but require higher dynamic range. 
Therefore they are best stored using E5M2 data format. 


Scaling factors
---------------


FP8's limited dynamic range is insufficient for many tensors. 
To address this, scaling factors are used. In FP8 Current Scaling there is one float32 scale factor per tensor.
The representation of tensor :math:`A` in FP8 precision is given by:

.. math::
    A = A_{fp8} \cdot s

where :math:`A_{fp8}` is an FP8 tensor and :math:`s` is a scalar 32-bit float.

Let's look more closely at how quantization to FP8 with scaling factor is implemented in
the FP8 Current Scaling recipe.

.. raw:: html
   :file: img/fp8_scaling_concept.svg

*Figure 3: Quantization to FP8 consists of amax computation, scaling to fit the FP8 range and casting to the respective FP8 format.*

Quantization to FP8 consists of 3 steps:

1. Computation of the absolute maximum value of the tensor - we refer to it as ``amax``.
2. Applying the scaling factor of ``fp8_max / amax`` to the tensor, to fit it into the FP8 range
3. Casting into the respective FP8 format using *Round To Nearest Even (RTNE)*. Values round to the nearest representable FP8 value. When exactly halfway between two values, rounds to the one with even mantissa to minimize systematic bias.

From a performance perspective, quantization implies that the tensor needs to be read twice – first to compute amax, then to apply the scaling factor and quantize to FP8.

.. raw:: html
   :file: img/fp8_cast_process.svg

*Figure 4: FP8 quantization with current scaling recipe - two tensor reads are needed, one to compute amax and one to apply the scaling factor and cast to FP8.*

Hardware support
----------------

The Ada architecture introduced FP8 support in Tensor Cores, enabling efficient low-precision computation. 
Tensor Cores support every combination of E4M3 and E5M2 formats as inputs, allowing flexible precision choices for different operands.
The inputs to an FP8 Tensor Core operation consist of chunks of FP8 tensors along with their corresponding scaling factors.
The Tensor Core performs the matrix multiplication in FP8 precision and produces output in higher precision (FP16, BF16, or FP32).

.. raw:: html
   :file: img/fp8_tensor_core.svg

*Figure 5: FP8 Tensor Cores process two input tensors (A and B) with their respective scaling factors and perform matrix multiplication to accumulate higher-precision output.*


Transpose handling
------------------

On Blackwell and later, hardware supports multiple GEMM layouts, so no transpose of the FP8 tensor is needed.
So columnwise tensor is in fact the same tensor as rowwise tensor. In this case,
quantization happens usually only in forward pass and is saved for
backward pass.

For Hopper and Ada, transpose of the FP8 tensor is needed in backward pass. 
Making columnwise tensor (transpose) from rowwise tensor is supported for FP8 tensors.
So all 3 options from :ref:`introduction Transpose handling section <handling_transposes>` are supported.

Distributed training 
--------------------

All gather of transposed FP8 tensors is not supported. 
Thus if gathered transpose is needed, the non-transposed tensor is gathered and then it is transposed.
Note that this only applies to Hopper and Ada, since Blackwell and later do not need transposed tensors.

For tensors, which are gathered - input and gradient in sequence parallelism,
amax reduction is performed before quantization. 
If before synchronization there was ``amax_1`` on node 1, 
``amax_2`` on node 2, etc., after synchronization there will be ``max(amax_1, amax_2, ...)`` on all nodes.
To make quantized all-gather possible, 
all nodes must use the same scaling factor, which is computed from the synchronized global amax.


.. raw:: html
   :file: img/fp8_current_scaling_all_gather.svg

*Figure 7: Quantization and all-gather flow for FP8 current scaling showing amax computation and synchronization.*


Supported devices
-----------------

Ada and later (SM 8.9+)

Examples
--------

Here's how to use FP8 Current Scaling recipe in PyTorch and JAX:

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: pytorch_current_scaling_example.py
         :language: python
         :start-after: # START_CURRENT_SCALING_EXAMPLE
         :end-before: # END_CURRENT_SCALING_EXAMPLE

   .. tab:: JAX

      .. literalinclude:: jax_current_scaling_example.py
         :language: python
         :start-after: # START_CURRENT_SCALING_EXAMPLE
         :end-before: # END_CURRENT_SCALING_EXAMPLE