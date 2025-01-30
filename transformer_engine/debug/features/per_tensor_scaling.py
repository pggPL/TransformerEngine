# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from transformer_engine.pytorch.fp8 import _default_sf_compute
from transformer_engine.pytorch.tensor import Float8Quantizer


def per_tensor_cast(tensor: torch.Tensor, fp8_dtype, margin=0, out=None) -> torch.Tensor:

    assert tensor.dtype in (
        torch.float,
        torch.float16,
        torch.bfloat16,
    ), "[NVTORCH INSPECT ERROR] Unsupported tensor type for per tensor scaling"
    assert tensor.is_cuda, "[NVTORCH INSPECT ERROR] Must be a GPU tensor."
    assert fp8_dtype in {
        tex.DType.kFloat8E4M3,
        tex.DType.kFloat8E5M2,
    }, "[NVTORCH INSPECT ERROR] Only 2 FP8 types: E4M3 and E5M2 are supported in TE."

    if fp8_dtype == tex.DType.kFloat8E4M3:
        fp8_max = Format.E4M3.value.max_fwd
    else:
        fp8_max = Format.E5M2.value.max_fwd
    amax = torch.max(torch.abs(tensor)).float()
    one = torch.ones(1, device=tensor.device)

    scale = _default_sf_compute(amax, one, fp8_max, margin)

    quantizer = Float8Quantizer(scale, amax, fp8_dtype)

    if out is not None:
        quantizer.update_quantized(tensor, out)
        return None
    else:
        return quantizer(tensor)


@Registry.register_feature(namespace="transformer_engine")
class PerTensorScaling(TEConfigAPIMapper):
    """
    Per Tensor Current Scaling feature for FP8 tensor in Transformer engine.

    If this feature is enabled, delayed scaling is disabled for the specified FP8 Tensor and GEMM.

    Config:
    To enable the feature in yaml config:
    transformer_engine:
        per_tensor_scaling:
        enabled: True
        ...

    Config fields:
    This feature works at a tensor level, you can set the following properties for each tensor:
    - margin: int, default is 0, amax = amax * (2^margin)
    - tensors/tensors_struct: tensors list or tensors_struct - please look into the Transformer Engine Precision Debug Tools documentation for more information.

    Gemm and Tensor structure is described below:
    """

    def _get_margin_default(self):
        return 0

    @api_method
    def fp8_gemm(self, config, layer_name, gemm):
        assert config["gemm"] == gemm
        return True

    @api_method
    def use_process_tensor(self, config, layer_name, gemm, tensor_name):
        return True

    @api_method
    def process_tensor(
        self, config, layer_name, gemm, tensor_name, tensor, default_quantizer=None, out=None
    ):
        for key in config.keys():
            if key not in ["gemm", "tensor", "margin"]:
                raise ValueError(f'[NVTORCH INSPECT ERROR] Unexpected key in config: "{key}".')

        assert default_quantizer is not None, (
            f"[NVTORCH INSPECT ERROR] Feature={self.__class__.__name__}, API=process_tensor:"
            f" Provide FP8 dtype when using process_tensor for per_tensor_scaling. {layer_name}"
        )

        nvinspect_api.log_message(
            f"Feature={self.__class__.__name__}, API=process_tensor: {gemm}, {tensor_name}",
            layer_name,
            extra_cachable_args=(gemm, tensor_name),
        )

        margin = config.get("margin", self._get_margin_default())
        fp8_tensor = per_tensor_cast(tensor, default_quantizer.dtype, margin=margin, out=out)
        return fp8_tensor
