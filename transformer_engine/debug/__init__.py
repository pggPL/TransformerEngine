# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

try:
    from . import pytorch
except ImportError as e:
    pass

from .debug_state import set_weight_tensor_tp_group_reduce
