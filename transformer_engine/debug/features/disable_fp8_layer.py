# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.#
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

import nvdlfw_inspect.api as nvinspect_api
from nvdlfw_inspect.registry import Registry, api_method


@Registry.register_feature(namespace="transformer_engine")
class DisableFp8Layer:
    """
    Feature to disable FP8 for entire layer or set of layers in Transformer Engine.

    Config:

    To enable the feature in yaml config:
    transformer_engine:
      disable_fp8_layer:
        enabled: True

    """

    @api_method
    def fp8_gemm(self, config, layer_name, *args, **kwargs):
        for key in config:
            if key not in ["enabled", "gemm"]:
                raise ValueError(f'[NVTORCH INSPECT ERROR] Unexpected key in config: "{key}".')
        # If FP8 training, disable FP8 for the selected layers if this feature is enabled in config.
        nvinspect_api.log_message(f"FP8 Disabled", layer_name)
        return False

    def parse_config_and_api(self, config, **kwargs):
        # Determines whether to run the API
        # DisableFp8Layer is the only feature provided by the Transformer Engine
        # which does not inherit from TEConfigAPIMapper.
        #
        # Explanation of the parse_config_and_api can be found in the nvidia-dlframework-inspect documentation.
        return config["enabled"], None
