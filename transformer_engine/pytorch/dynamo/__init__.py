# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile glue for Transformer Engine."""

from .quantizer_opaque import register_value_opaque_quantizer, is_value_opaque_quantizer
from .tensor import empty_tensor
from .custom_op import (
    register_custom_op,
    register_custom_op_with_autograd,
    TensorOrQuantized,
)

__all__ = [
    "register_value_opaque_quantizer",
    "is_value_opaque_quantizer",
    "empty_tensor",
    "register_custom_op",
    "register_custom_op_with_autograd",
    "TensorOrQuantized",
]
