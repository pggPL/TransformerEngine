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

from transformer_engine.debug.features.api import TEConfigAPIMapper

import nvdlfw_inspect.api  as nvinspect_api
from nvdlfw_inspect.registry import Registry, api_method
from nvdlfw_inspect.utils import append_parent_docstring


@Registry.register_feature(namespace="transformer_engine")
@append_parent_docstring(parent=TEConfigAPIMapper)
class DisableFp8Gemm(TEConfigAPIMapper):
    """
    Feature to disable FP8 GEMM in transformer engine.

    APIs:

    1. transformer_engine.is_fp8_gemm_disabled
    - When using this api, you would need to pass args:
    -- layer_name: this is matched with the layer description in the config file
    -- gemm: this is matched with one of the gemms in the config field, and passed as a kwarg. For example, gemm='gemm1'
    -- fp8_enabled: bool. this notifies if the original gemm execution is in FP8 or not, and passed as a kwarg. For example, fp8_enabled=True

    Config: 

    To enable the feature in yaml config:
    transformer_engine:
      disable_fp8_gemm:
        enabled: True
        feature_properties:
        ...
    """

    @api_method
    def fp8_gemm(self, config, layer_name, gemm):
        for key in config:
            if key != 'gemm':
                raise ValueError(f"[NVTORCH INSPECT ERROR] Unexpected key in config: \"{key}\".")
                
        if config["gemm"] == gemm:
            nvinspect_api.log_message(
                f"Feature={self.__class__.__name__}, API=is_fp8_gemm_enabled: {gemm}: FP8 GEMM: False", layer_name)
            return False
        return True