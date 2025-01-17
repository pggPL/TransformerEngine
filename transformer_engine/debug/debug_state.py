import torch

from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Union

try:
    import nvtorch_inspect.api as nvinspect_api
except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError("ERROR: Could not locate nvtorch_inspect package. Make sure it is installed correctly.")

@dataclass
class FP8GemmState:
    FPROP: bool = None
    DGRAD: bool = None
    WGRAD: bool = None

@dataclass
class DelayedScalingState:
    FPROP_ACTIVATION: bool = None
    FPROP_WEIGHT: bool = None
    DGRAD_GRADIENT: bool = None
    DGRAD_WEIGHT: bool = None
    WGRAD_GRADIENT: bool = None
    WGRAD_ACTIVATION: bool = None


class DebugLayerState:
    """
    A class to manage the state of debug layers.
    """
    layer_count = 1
    layers_initialized = {}
    weight_tensor_tp_group_reduce = True


    @classmethod
    def reset(cls):
        cls.layers_initialized.clear()
    
    @classmethod
    def initialize_state(cls, name, fp8_enabled: bool):
        if name not in cls.layers_initialized:
            fp8_gemm_state = FP8GemmState(
                FPROP=nvinspect_api.transformer_engine.is_fp8_gemm_enabled(name, gemm="fprop", fp8_enabled=fp8_enabled)["ret"],
                DGRAD=nvinspect_api.transformer_engine.is_fp8_gemm_enabled(name, gemm="dgrad", fp8_enabled=fp8_enabled)["ret"],
                WGRAD=nvinspect_api.transformer_engine.is_fp8_gemm_enabled(name, gemm="wgrad", fp8_enabled=fp8_enabled)["ret"]
            )

            delayed_scaling_state = DelayedScalingState()
            if fp8_gemm_state.FPROP:
                delayed_scaling_state.FPROP_ACTIVATION=nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled(name, gemm="fprop", tensor_name="activation", fp8_enabled=fp8_enabled)["ret"]
                delayed_scaling_state.FPROP_WEIGHT=nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled(name, gemm="fprop", tensor_name="weight", fp8_enabled=fp8_enabled)["ret"]
            else:
                delayed_scaling_state.FPROP_ACTIVATION = False
                delayed_scaling_state.FPROP_WEIGHT = False

            if fp8_gemm_state.DGRAD:
                delayed_scaling_state.DGRAD_GRADIENT=nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled(name, gemm="dgrad", tensor_name="gradient", fp8_enabled=fp8_enabled)["ret"]
                delayed_scaling_state.DGRAD_WEIGHT=nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled(name, gemm="dgrad", tensor_name="weight", fp8_enabled=fp8_enabled)["ret"]
            else:
                delayed_scaling_state.DGRAD_GRADIENT = False
                delayed_scaling_state.DGRAD_WEIGHT = False
            
            if fp8_gemm_state.WGRAD:
                delayed_scaling_state.WGRAD_GRADIENT=nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled(name, gemm="wgrad", tensor_name="gradient", fp8_enabled=fp8_enabled)["ret"]
                delayed_scaling_state.WGRAD_ACTIVATION=nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled(name, gemm="wgrad", tensor_name="activation", fp8_enabled=fp8_enabled)["ret"]
            else:
                delayed_scaling_state.WGRAD_GRADIENT = False
                delayed_scaling_state.WGRAD_ACTIVATION = False
                
            cls.layers_initialized[name] = namedtuple('LayerState', ['DelayedScaling', 'FP8Gemm'])(delayed_scaling_state, fp8_gemm_state)
    
    @classmethod
    def get(cls, name: str):
        return cls.layers_initialized[name]
    
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
    DebugLayerState.set_weight_tensor_tp_group_reduce(enabled)