..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

MXFP8
=====


Hardware support for the MXFP8 (mixed FP8 format) first appeared in the Blackwell GPU.
This recipe is an improved version of FP8 blockwise scaling. With the native hardware features introduced in SM 10.0,
it is possible to efficiently use one scaling factor per 32 consecutive values.



Data Format
-----------

MXFP8 uses the same blockwise scaling principle as FP8 Blockwise Scaling, but with hardware-optimized parameters
that enable more efficient execution on Blackwell GPUs.

The representation of tensor ``A`` in MXFP8 precision is given by:

.. code-block:: python
    
    A[i] = A_fp8[i] * s[block(i)]

where

* ``A_fp8`` is an FP8 tensor in E4M3 format,
* ``s`` is an array of E8M0 scaling factors,
* ``block(i)`` determines which block index ``i`` belongs to.

**FP8 tensor**

MXFP8 uses the E4M3 FP8 format exclusively for both forward and backward passes.
The finer-grained scaling (one factor per 32 values instead of 128) provides sufficient dynamic range
without requiring the E5M2 format.


**Block size**

Block size is **32** - significantly smaller than the 128 used in FP8 Blockwise Scaling.
This provides more granular control over quantization precision.

Blocks are one-dimensional, containing 32 consecutive values. No 2D scaling is performed.

For optimal performance, tensor dimensions should be divisible by 32.


**Scaling factors**

Scaling factors use the **E8M0 format** (8 exponent bits, 0 mantissa bits), which inherently represents
powers of 2. This is different from FP8 Blockwise Scaling, which uses 32-bit floating point numbers
that are optionally constrained to powers of 2.

The scaling factor for each block is computed as follows:

1. Find the maximum absolute value (``amax``) across all 32 elements in the block
2. Compute the E8M0 exponent using the formula: ``exponent = float_to_e8m0(amax / max_norm)``
   where ``max_norm = 448`` (the maximum representable value in E4M3 format)
   
   - The ``float_to_e8m0`` conversion extracts the 8-bit exponent from the float32 representation
   - Uses **round-up** (toward positive infinity) to ensure values are not clipped too aggressively
   - Applies **saturation** to handle edge cases (NaN → 0xFF, inf → 0xFE, zero → 0x00)
   
3. The resulting scaling factor is ``s = 2^exponent``

This ensures that the largest value in each block, when divided by the scaling factor and
converted to E4M3, will fit within the FP8 range without saturation.


.. raw:: html
   :file: img/fp8_1d_scaling.svg

*Figure 1. MXFP8 uses one E8M0 scaling factor per 32 consecutive elements, providing fine-grained
quantization control with compact scaling factor representation.*


Swizzling scaling factors
-------------------------

To optimize memory access patterns and improve performance, scaling factors for both input tensors are stored in a swizzled layout before the GEMM operation.

.. raw:: html
   :file: img/mxfp8_swizzle_both_tensors.svg

*Figure 2. Both input tensors undergo swizzling before GEMM to optimize memory access for Blackwell Tensor Cores.*

Swizzling is a two-step transformation process designed for Blackwell Tensor Cores:

1. **Tile Transposition**: Scaling factors are reorganized into **128x4 tiles** (representing 128 rows and 4 columns of scaling factors).
   This results in a **block-linear layout** in memory, ensuring that all scaling factors for a specific computation tile are stored contiguously.

.. raw:: html
   :file: img/mxfp8_tile_transposition.svg

*Figure 3. Tile Transposition: The logical grid of scaling factors is broken into 128x4 tiles, which are then stored linearly in memory.*

2. **Internal Interleaving**: Within each 128x4 block, the scaling factors are **interleaved**.
   Instead of storing data column-by-column or row-by-row locally, the elements are shuffled to align with the specific thread access patterns of the hardware, minimizing bank conflicts.

.. raw:: html
   :file: img/mxfp8_tile_internal.svg

*Figure 4. Inside the Tile: Scaling factors within a tile are interleaved to optimize Tensor Core access.*

The swizzling pattern is hardware-specific and designed to align with the access patterns of Blackwell Tensor Cores.


Handling transposes
-------------------

Unlike FP8 Blockwise Scaling on Hopper, Blackwell architecture supports more GEMM layouts (TN, NT, NN),
so explicit transposition of tensors is often not required for hardware compatibility.

However, MXFP8 uses 1D scaling with 1 scaling factor per 32 consecutive values.
This means that rowwise and columnwise quantized tensors are numerically different.
Consequently, transposing a 1D quantized tensor is not supported, as the rowwise tensor has scaling factors
aligned with rows, while a columnwise tensor would need scaling factors aligned with columns.
Computing one from the other would lead to significant precision loss.
Thus, a quantized tensor can only be obtained accurately from higher precision data.

.. raw:: html
   :file: img/mxfp8_transpose_handling.svg

*Figure 5. Transpose handling in MXFP8: While the tensor shape is transposed, the internal scaling block orientation
is different (horizontal vs vertical strips of 32 elements). Direct transposition is invalid because scaling factors
would not align with the new dimension axis.*


Distributed training
--------------------

**Scale synchronization**

The blockwise scaled tensor does not need any scale synchronization among the nodes. 
This is because each scaling factor is local to the 32 elements,
not like in FP8 Current/Delayed Scaling where the scale is global for the whole tensor, 
which may be sharded.

**Quantized all-gather**

Gather of columnwise tensor is supported and is used since:

- as mentioned in the previous section, it is not supported to compute columnwise quantized tensor from rowwise quantized one,
- high precision tensor is not gathered in most cases due to performance reasons,

Since Blackwell architecture supports more GEMM layouts (TN, NT, NN), there is no need to transpose 
data or scaling factors to a specific layout required by the hardware (unlike on Hopper). This makes the gathering process trivial.


Examples
--------

Here's how to use MXFP8 recipe in PyTorch and JAX:

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: pytorch_mxfp8_example.py
         :language: python
         :start-after: # START_MXFP8_EXAMPLE
         :end-before: # END_MXFP8_EXAMPLE

   .. tab:: JAX

      .. literalinclude:: jax_mxfp8_example.py
         :language: python
         :start-after: # START_MXFP8_EXAMPLE
         :end-before: # END_MXFP8_EXAMPLE


Supported devices
-----------------

Blackwell and later (SM 10.0+)