# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations
from typing import Optional, Tuple, Iterable, Any, Dict
import torch

from .quantized_tensor import QuantizedTensor, Quantizer, _IdentityFunc

try:
    import nvtorch_inspect.api as nvinspect_api
except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError("ERROR: Could not locate nvtorch_inspect package. Make sure it is installed correctly.")

class DebugQuantizer(Quantizer):
    def __init__(self, layer_name, tensor_name, iteration, parent_quantizer, tp_group):
        self.layer_name = layer_name
        self.tensor_name = tensor_name
        self.iteration = iteration
        self._parent_quantizer = parent_quantizer
        self.tp_group = tp_group
    
    def quantize(self, tensor, *, out = None):
        args = {
            "layer_name": self.layer_name, 
            "tensor": tensor, 
            "tensor_name": self.tensor_name, 
            "iteration": self.iteration
        }
        nvinspect_api.transformer_engine.save_stats_for_logging(*args)
        tensor = nvinspect_api.transformer_engine.process_tensor(*args)
        if self._parent_quantizer is None:
            if out is not None:
                out.copy(tensor)
                tensor = out
        else:
            quantized_tensor = self._parent_quantizer(tensor)
            args["tensor"] = quantized_tensor
            tensor = nvinspect_api.transformer_engine.process_quantized_tensor(*args)
            nvinspect_api.transformer_engine.save_fp8_stats_for_logging(*args)
        
        # i should return tensor packed in DebugQuantizedTensor
        return DebugQuantizedTensor(
            _tensor=tensor,
            _quantized_tensor=quantized_tensor if self._parent_quantizer is not None else None,
            _layer_name=self.layer_name
        )

    
    def get_weight_workspace(self, tensor, quantized_tensor):
        args = {
            "layer_name": self.layer_name, 
            "tensor": tensor, 
            "tensor_name": self.tensor_name, 
            "iteration": self.iteration
        }
        red_group = nvinspect_api.get_tensor_reduction_group()
        skip_reduction = not DebugLayerState.weight_tensor_tp_group_reduce
        nvinspect_api.set_tensor_reduction_group(self.tp_group)
        nvinspect_api.transformer_engine.save_stats_for_logging(*args)
        args["tensor"] = quantized_tensor
        nvinspect_api.transformer_engine.save_stats_for_logging_quantized(*args)
        nvinspect_api.set_tensor_reduction_group(red_group)
        return tensor

    def process_after_quantization(self, tensor): # this is incorrect
        args = {
            "layer_name": self.layer_name, 
            "tensor": tensor, 
            "tensor_name": self.tensor_name, 
            "iteration": self.iteration
        }
        tensor = nvinspect_api.transformer_engine.process_quantized_tensor(*args)
        if isinstance(tensor, QuantizedTensor):
            nvinspect_api.transformer_engine.save_stats_for_logging_quantized(*args)
        else:
            nvinspect_api.transformer_engine.save_stats_for_logging(*args)
        return tensor
    
    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> QuantizedTensor:
        if self._parent_quantizer is not None:
            return self._parent_quantizer(shape, dtype=dtype, device=device)
        else:
            return torch.empty(shape, dtype=dtype, device=device)
    
    def calibrate(self, tensor: torch.Tensor):
        if self._parent_quantizer is not None:
            return self._parent_quantizer.caluubrate(tensor)
    
    def update_quantized(
        self, src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        if self._parent_quantizer is not None:
            return self.update_quantized(src, dst, noop_flag=noop_flag)
    
    def is_fp8(self):
        return self._parent_quantizer is not None
        


class DebugQuantizedTensor(QuantizedTensor):
    _tensor = None 
    _quantized_tensor = None
    _layer_name: str
    # saves both: high precision and fp8 tensor
    def prepare_for_save(self):
        return super.prepare_for_saving(self._tensor, self._quantized_tensor)
    
    def restore_from_saved(self, tensors):
        if self._tensor is not None:
            self._tensor = tensors[0]
            tensors = tensors[1:]
        if self._quantized_tensor is not None:
            tensors = self._quantized_tensor.restore_from_saved(tensors)
        return tensors

    def get_tensor(self, gemm_name):
        if nvinspect_api.transformer_engine.fp8_gemm(self._layer_name, gemm_name):
            return self._quantized_tensor
        else:
            # We do this if to log information about each high precision gemm once.
            if [self.gemm_name, self.tensor_name] in [
                ["fprop", "weight"], ["wgrad", "gradient"], ["dgrad", "gradient"]]:
                nvinspect_api.log_message(f"{self.gemm_name}: High Precision", self._layer_name)
            return self._tensor
