# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Union
import sys


class TEDebugState:
    """
    A class to manage the state of debug layers.
    """

    layer_count = 1
    layers_initialized = {}
    weight_tensor_tp_group_reduce = True
    debug_enabled = None
    initialized = False

    @classmethod
    def initialize(cls):
        if "nvdlfw_inspect" in sys.modules:
            import nvdlfw_inspect.api as nvinspect_api

            if cls.debug_enabled is False and nvinspect_api.DEBUG_MANAGER is not None:
                raise RuntimeError(
                    "[nv_dlfw_inspect] nv_dlfw_inspect module should be initialized before"
                    " initialization of the first TE module"
                )
            cls.debug_enabled = nvinspect_api.DEBUG_MANAGER is not None

    @classmethod
    def reset(cls):
        from .features.utils.stats_buffer import STATS_BUFFERS, StatsBuffers

        STATS_BUFFERS.reset()
        cls.debug_enabled = None
        cls.layers_initialized.clear()

    @classmethod
    def get_layer_count(cls):
        """
        Layer counter is used when layer names are not provided to modules by user.
        """
        lc = cls.layer_count
        cls.layer_count += 1
        return lc

    @classmethod
    def set_weight_tensor_tp_group_reduce(cls, enabled):
        cls.weight_tensor_tp_group_reduce = enabled


def set_weight_tensor_tp_group_reduce(enabled):
    TEDebugState.set_weight_tensor_tp_group_reduce(enabled)
