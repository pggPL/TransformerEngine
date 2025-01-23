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
import nvdlfw_inspect.api as nvinspect_api
from nvdlfw_inspect.registry import Registry, api_method
from nvdlfw_inspect.utils import append_parent_docstring


import transformer_engine_torch as tex
from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.tensor import Float8Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.fp8 import _default_sf_compute
from transformer_engine.pytorch.constants import TE_DType



def fake_quantize_fp8(tensor: torch.Tensor, fp8_format, margin=0):
    assert tensor.dtype in (torch.float, torch.float16, torch.bfloat16), "[NVTORCH INSPECT ERROR] Unsupported tensor type."
    assert tensor.is_cuda, "[NVTORCH INSPECT ERROR] Must be a GPU tensor."
    assert fp8_format in {"E4M3", "E5M2", "MXE4M3", "MXE5M2"},\
          "[NVTORCH INSPECT ERROR] Only 4 FP8 types: E4M3, E5M2, MXE4M3, MXE5M2 are supported in TE."
    if fp8_format in ["E4M3", "E5M2"]:
        if fp8_format == "E4M3":
            fp8_max = Format.E4M3.value.max_fwd
            fp8_dtype = tex.DType.kFloat8E4M3
        else:
            fp8_max = Format.E5M2.value.max_fwd
            fp8_dtype = tex.DType.kFloat8E5M2
        amax = torch.max(torch.abs(tensor)).float()
        one = torch.ones(1, device=tensor.device)
        scale = _default_sf_compute(amax, one, fp8_max, margin)

        quantizer = Float8Quantizer(scale, amax, fp8_dtype)
    else:
        MXFP8Quantizer(fp8_dtype=fp8_format)
    return quantizer(tensor).dequantize()


@Registry.register_feature(namespace="transformer_engine")
@append_parent_docstring(parent=TEConfigAPIMapper)
class FakeQuantFp8(TEConfigAPIMapper):
    """
    Fake Quantization feature in Transformer engine. 
    
    Fake quantization in this case refers to casting a tensor to FP8 and back to original dtype.

    This feature can be enabled for both FP8 training and High Precision training.
    - High precision training: tensor is cast to FP8 and back. Runs high precision GEMM.
    - FP8 training: FP8 GEMM and delayed scaling are disabled for the specified GEMM. A tensor is cast to FP8 with current scaling and back. Runs high precision GEMM.

    APIs:
    1. transformer_engine.is_fp8_gemm_enabled:
    - When using this api, you would need to pass args:
    -- layer_name: this is matched with the layer description in the config file
    -- gemm: this is matched with one of the gemms in the config field, and passed as a kwarg. For example, gemm='gemm1'
    -- fp8_enabled: bool. this notifies if the original gemm execution is in FP8 or not, and passed as a kwarg. For example, fp8_enabled=True
    NOTE: The above api is needed in transformer engine to disable the execution of Gemm in FP8

    2. transformer_engine.process_tensor
    - When using this api, you would need to pass args:
    -- layer_name: : this is matched with the layer description in the config file
    -- gemm: this is matched with one of the gemms in the config field, and passed as a kwarg. For example, gemm='gemm1'
    -- tensor_name: this is matched with one of the tensors in the config field for a given gemm, and passed as a kwarg. For example, tensor_name='tensor1'
    -- tensor: the tensor to process, and passed as a kwarg. For example, tensor={torch tensor}
    -- fp8_enabled: bool. this notifies if the original gemm execution is in FP8 or not, and passed as a kwarg. For example, fp8_enabled=True

    Config:
    To enable the feature in yaml config:
    transformer_engine:
      fake_quant_fp8:
        enabled: True
        feature_properties:
        ...
    
    Config fields:
    This feature works at a tensor level, you can set the following properties for each tensor:
    - quant_format: Dictionary containing tensor names to FP8 formats. Options: {'E4M3', 'E5M2', 'MXE4M3', 'MXE5M2'}
    - margin: int, default is 0

    Gemm and Tensor structure is described below:
    """
    def _supported_formats(self):
        return ["E4M3", "E5M2", "MXE4M3", "MXE5M2"]

    def _get_margin_default(self):
        return 0
    
    @api_method
    def fp8_gemm(self, config, layer_name, gemm):
        return False

    @api_method
    def use_process_tensor(self, config, layer_name, tensor_name, gemm):
        return True

    @api_method
    def process_tensor(self, config, layer_name, gemm, tensor_name, tensor, default_quantizer=None, out=None):
        for key in config.keys():
            if key not in ["gemm", "tensor", "quant_format", "margin"]:
                raise ValueError(f"[NVTORCH INSPECT ERROR] Unexpected key in config: \"{key}\".")
        
        if "quant_format" not in config:
            raise ValueError(f"[NVTORCH INSPECT ERROR] Feature={self.__class__.__name__}, API=process_tensor: quant_format missing for Tensor: {tensor_name} in the config yaml for FakeQuantFp8 feature which is a required field")
        if config["quant_format"] not in self._supported_formats():
            raise ValueError(f"[NVTORCH INSPECT ERROR] Feature={self.__class__.__name__}, API=process_tensor: quant_format: {config['quant_format']} for Tensor: {tensor_name} in the config yaml for FakeQuantFp8 feature is not supported")

        quant_format = config["quant_format"]
        margin = config.get('margin', self._get_margin_default())
        q_tensor = fake_quantize_fp8(tensor, quant_format, margin=margin)
        return q_tensor