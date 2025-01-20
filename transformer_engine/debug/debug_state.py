import torch

from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Union
import sys



from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS, StatsBuffers

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
        if cls.debug_enabled is False and "nv_dlfw_inspect" in sys.modules:
            raise RuntimeError("[nv_dlfw_inspect] nv_dlfw_inspect module should be initialized before initialization of the first TE module")
        cls.debug_enabled = "nv_dlfw_inspect" in sys.modules
        initialized = True

    @classmethod
    def reset(cls):
        STATS_BUFFERS.reset()
        cls.layers_initialized.clear()
    
    @classmethod
    def get_layer_count(cls):
        """
        Layer counter is used when layer names are not provided to modules by user.
        """
        lc = cls.layer_count
        cls.layer_count += 1
        return lc

    def num_of_features_for_layer(self, layer_name):
        pass

    @classmethod
    def set_weight_tensor_tp_group_reduce(cls, enabled):
        cls.weight_tensor_tp_group_reduce = enabled
    

def set_weight_tensor_tp_group_reduce(enabled):
    TEDebugState.set_weight_tensor_tp_group_reduce(enabled)