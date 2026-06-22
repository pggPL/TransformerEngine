# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile glue for Transformer Engine."""

from .quantizer_opaque import register_value_opaque_quantizer, is_value_opaque_quantizer
from .traceable_alloc import make_empty_traceable

__all__ = [
    "register_value_opaque_quantizer",
    "is_value_opaque_quantizer",
    "make_empty_traceable",
]
