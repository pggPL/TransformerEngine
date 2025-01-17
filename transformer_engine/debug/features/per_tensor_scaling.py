# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from transformer_engine.debug.features.api import TEConfigAPIMapper
import nvtorch_inspect.api as nvinspect_api
from nvtorch_inspect.registry import Registry, api_method
from nvtorch_inspect.utils import append_parent_docstring

import transformer_engine_torch as tex
from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.fp8 import _default_sf_compute
from transformer_engine.pytorch.tensor import Float8Quantizer


def per_tensor_cast(tensor: torch.Tensor, 
             fp8_dtype,
             margin=0,
             out=None) -> torch.Tensor:
    
    assert tensor.dtype in (torch.float, torch.float16, torch.bfloat16), "[NVTORCH INSPECT ERROR] Unsupported tensor type for per tensor scaling"
    assert tensor.is_cuda, "[NVTORCH INSPECT ERROR] Must be a GPU tensor."
    assert fp8_dtype in {tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2}, "[NVTORCH INSPECT ERROR] Only 2 FP8 types: E4M3 and E5M2 are supported in TE."

    if fp8_dtype == "E4M3":
        fp8_max = Format.E4M3.value.max_fwd
        fp8_dtype = tex.DType.kFloat8E4M3
    else:
        fp8_max = Format.E5M2.value.max_fwd
        fp8_dtype = tex.DType.kFloat8E5M2
    amax = torch.max(torch.abs(tensor)).float()
    one = torch.ones(1, device=tensor.device)
    scale = _default_sf_compute(amax, one, fp8_max, margin)

    quantizer = Float8Quantizer(scale, amax, fp8_dtype)

    quantizer

    if out is not None:
        quantizer.update_quantized(tensor, out)
        return None
    else:
        return quantizer(tensor)


@Registry.register_feature(namespace="transformer_engine")
@append_parent_docstring(parent=TEConfigAPIMapper)
class PerTensorScaling(TEConfigAPIMapper):
    """
    Per Tensor Current Scaling feature for FP8 tensor in Transformer engine.

    If this feature is enabled, delayed scaling is disabled for the specified FP8 Tensor and GEMM.

    APIs:
    1. transformer_engine.is_fp8_delayed_scaling_enabled:
    - When using this api, you would need to pass args:
    -- layer_name: this is matched with the layer description in the config file
    -- gemm: this is matched with one of the gemms in the config field, and passed as a kwarg. For example, gemm='gemm1'
    -- tensor_name: this is matched with one of the tensors in the config field for a given gemm, and passed as a kwarg. For example, tensor_name='tensor1'
    -- fp8_enabled: bool. this notifies if the original gemm execution is in FP8 or not, and passed as a kwarg. For example, fp8_enabled=True
    NOTE: The above api is needed in transformer engine to disable the delayed scaling for a given tensor

    2. transformer_engine.process_tensor
    - When using this api, you would need to pass args:
    -- layer_name: this is matched with the layer description in the config file
    -- gemm: this is matched with one of the gemms in the config field, and passed as a kwarg. For example, gemm='gemm1'
    -- tensor_name: this is matched with one of the tensors in the config field for a given gemm, and passed as a kwarg. For example, tensor_name='tensor1'
    -- tensor: the tensor to process, and passed as a kwarg. For example, tensor={torch tensor}
    -- fp8_enabled: bool. this notifies if the original gemm execution is in FP8 or not, and passed as a kwarg. For example, fp8_enabled=True

    Config:
    To enable the feature in yaml config:
    transformer_engine:
      per_tensor_scaling:
        enabled: True
        feature_properties:
        ...

    Config fields:
    This feature works at a tensor level, you can set the following properties for each tensor:
    - margin: int, default is 0, amax = amax * (2^margin)

    Gemm and Tensor structure is described below:
    """
    def _get_margin_default(self):
        return 0

    @api_method
    def fp8_gemm(self, config, layer_name, **kwargs):
        return False

    @api_method
    def process_tensor(self, config, layer_name, **kwargs):
        for key in config.keys():
            if key not in ["gemm", "tensor", "margin"]:
                raise ValueError(f"[NVTORCH INSPECT ERROR] Unexpected key in config: \"{key}\".")
       
        # todo[pgadzinski] - change this first assert into sth meaningful
        assert kwargs["fp8_enabled"], f"[NVTORCH INSPECT ERROR] Feature={self.__class__.__name__}, API=process_tensor: Per tensor scaling feature should only be used during FP8 training."        
        assert "fp8_dtype" in kwargs, f"[NVTORCH INSPECT ERROR] Feature={self.__class__.__name__}, API=process_tensor: Provide FP8 dtype when using process_tensor for per_tensor_scaling. {layer_name}"
        
        nvinspect_api.log_message(f"Feature={self.__class__.__name__}, API=process_tensor: {kwargs['gemm']}, {kwargs['tensor_name']}: Per Tensor Scaling", layer_name)
        
        margin = config.get('margin', self._get_margin_default())
        fp8_tensor = per_tensor_cast(kwargs["tensor"], kwargs["fp8_dtype"], margin=margin, out=kwargs.get("out", None))
        return fp8_tensor
