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

from transformer_engine.debug.features.api import TEConfigAPIMapper

import nvdlfw_inspect.api  as nvinspect_api
from nvdlfw_inspect.registry import Registry, api_method
from nvdlfw_inspect.utils import append_parent_docstring


@Registry.register_feature(namespace="transformer_engine")
class DisableFp8Gemm(TEConfigAPIMapper):
    """
    Feature to disable FP8 GEMM in transformer engine.

    Config: 

    To enable the feature in yaml config:
    transformer_engine:
      disable_fp8_gemm:
        enabled: True
        gemms: gemms list - please look into the Transformer Engine Precision Debug Tools documentation for more information.
    """

    @api_method
    def fp8_gemm(self, config, layer_name, gemm, iteration):
        for key in config:
            if key != 'gemm':
                raise ValueError(f"[NVTORCH INSPECT ERROR] Unexpected key in config: \"{key}\".")
                
        return False