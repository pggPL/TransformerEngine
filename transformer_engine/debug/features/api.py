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

import copy
from typing import Any
from nvtorch_inspect.base import BaseNamespaceAPI, BaseConfigAPIMapper
from nvtorch_inspect.registry import Registry

class TEConfigAPIMapper(BaseConfigAPIMapper):
    gemm_config_docstring = '''
    Supported yaml config structure:
    1. gemms: [gemm1, gemm2]
       feature_param1: value
       feature_param2: value

    2. gemms_struct:
         - gemm: gemm1
           feature_param1: value
           feature_param2: value
         - gemm: gemm2
           feature_param1: value
           feature_param2: value
    '''

    gemm_and_tensor_config_docstring = '''
    Supported yaml config structure:
    1. gemms: [gemm1, gemm2]
       tensors: [tensor1, tensor2]
       tensor_feature_param1: value
       gemm_feature_param1: value

    2. gemms: [gemm1, gemm2]
       tensors_struct:
        - tensor: tensor1
          tensor_feature_param1: value
        - tensor: tensor2
          tensor_feature_param2: value
        gemm_feature_param1: value

    3. gemms_struct:
        - gemm: gemm1
          tensors: [tensor1, tensor2]
          tensor_feature_param1: value
          gemm_feature_param1: value
        - gemm: gemm2
          tensors_struct:
          - tensor: tensor1
            tensor_feature_param1: value
          - tensor: tensor2
            tensor_feature_param2: value
          gemm_feature_param1: value
    '''
    def parse_config_and_api(self, config, **kwargs):
        # Process the config and returns True if the config and api args match, along with processed config.
        processed_config = None
        config_copy = copy.deepcopy(config)
        gemm_parsing = kwargs.get('gemm_parsing', False)
        tensor_parsing = kwargs.get('tensor_parsing', False)

        if gemm_parsing:
            # parse with GEMM and/or tensor
            processed_config = self._process_transformer_engine_config(config_copy, **kwargs)
        elif tensor_parsing:
            # parse with only tensor
            processed_config = self._process_tensor_config(config_copy, kwargs["tensor_name"])

        if not processed_config:
            return False, None
        
        if "enabled" in processed_config:
            processed_config.pop("enabled")
        return True, processed_config
    
    def _validate_gemm(self, gemm):
        assert gemm in ['fprop', 'wgrad', 'dgrad'], f"[NVTORCH INSPECT ERROR] Invalid gemm: {gemm}. It must be one of the ['fprop', 'wgrad', 'dgrad']."

    def _process_transformer_engine_config(self, config, **kwargs):
        '''
        Return config specific to a particular tensor name and gemm that matches the api args.
        '''
        if "gemms_struct" in config:
            for cfg in config["gemms_struct"]:
                self._validate_gemm(cfg["gemm"])
                if cfg["gemm"] == kwargs["gemm"]:
                    if kwargs['tensor_parsing']:
                        cfg = self._process_tensor_config(cfg, kwargs["tensor_name"])
                        if not cfg: 
                            return None
                    cfg_copy = copy.deepcopy(cfg)
                    config.pop("gemms_struct")
                    assert "enabled" not in cfg_copy, "[NVTORCH INSPECT ERROR] Enabled field should not be part of gemms_struct"
                    config.update(cfg_copy)
                    return config
        elif "gemms" in config:
            for gemm in config["gemms"]:
                self._validate_gemm(gemm)
            if kwargs["gemm"] in config["gemms"]:
                if kwargs['tensor_parsing']:
                    cfg = self._process_tensor_config(config, kwargs["tensor_name"])
                    if not cfg:
                        return None
                config["gemm"] = kwargs["gemm"]
                config.pop("gemms")
                return config
        else:
            raise ValueError(f"[NVTORCH INSPECT ERROR] Provide 'gemms_struct: List[Dict]' or 'gemms: List[str]' in the config yaml")


class TEDefaultFeatures:
    def is_fp8_enabled_for_layer(self, config, layer_name, **kwargs):
        """
        Check whether a given layer runs in FP8. 
        """
        return {"ret": True} if kwargs["fp8_enabled"] else {"ret": False}
    
    def is_fp8_gemm_enabled(self, config, layer_name, **kwargs):
        """
        Check whether a given GEMM runs in FP8. 
        """
        return {"ret": True} if kwargs["fp8_enabled"] else {"ret": False}

    def is_fp8_delayed_scaling_enabled(self, config, layer_name, **kwargs):
        """
        Check whether a given GEMM runs in delayed scaling. 
        """
        return {"ret": True} if kwargs["fp8_enabled"] else {"ret": False}

    def process_tensor(self, config, layer_name, **kwargs):
        """
        API to process a tensor. This must return a tensor.
        """
        return {"tensor": kwargs["tensor"]}


@Registry.register_namespace_api(namespace="transformer_engine")
class TransformerEngineAPI (BaseNamespaceAPI):
    """
    Transformer Engine API class that contains default APIs that are invoked when a config is not provided
    or a layer is not selected in the config.
    TransformerEngine specific features must override these APIs wherever required.
    The overridden APIs will be invoked whenever the corresponding feature is enabled in the config.
    """
    def __init__(self):
        BaseNamespaceAPI.__init__(self)
        self._default_api_impl = TEDefaultFeatures()
        self._cacheable_api_kwargs_map = {
            "save_stats_for_logging": ["tensor_name"],
            "save_fp8_stats_for_logging": ["tensor_name"],
            "is_fp8_gemm_enabled": ["gemm"],
            "process_tensor": ["tensor_name", "gemm"],
            "is_fp8_delayed_scaling_enabled": ["tensor_name", "gemm"],
        }

    def is_multiple_feature_invocation_allowed(self, api_name):
        """
        Check if API allowes executing multiple features for a single call
        """
        if api_name in {"is_fp8_gemm_enabled", "is_fp8_delayed_scaling_enabled"}:
            return True
        return False

    def input_assertions_hook(self, api_name, **kwargs):
        """
        These args must be passed as kwargs in the API call for all TransformerEngine specific APIs.
        """
        required_kwargs = {
            "save_stats_for_logging": ["tensor_name", "tensor"],
            "save_fp8_stats_for_logging": ["tensor_name", "tensor"],
            "is_fp8_gemm_enabled": ["gemm", "fp8_enabled"],
            "is_fp8_enabled_for_layer": ["fp8_enabled"],
            "is_fp8_delayed_scaling_enabled": ["fp8_enabled", "gemm", "tensor_name"],
            "process_tensor": ["tensor", "fp8_enabled", "gemm", "tensor_name"],
            "default": ["fp8_enabled", "tensor_name", "gemm"]
        }

        if api_name in required_kwargs:
            for kwarg in required_kwargs[api_name]:
                assert kwarg in kwargs, f"[NVTORCH INSPECT ERROR] Cannot route API, too ambiguous. Provide {kwarg} in {api_name}."
        else:
            for kwarg in required_kwargs["default"]:
                assert kwarg in kwargs, f"[NVTORCH INSPECT ERROR] Cannot route API, too ambiguous. Provide {kwarg} in {api_name}."


    def routing_condition(self, api_name, config, layer_name, feature_obj, **kwargs):
        """
        Overridden APIs are selected based on the GEMM name in the config and kwargs.
        """
        if api_name in {"is_fp8_enabled_for_layer"}:
            return True, None

        tensor_parsing = False if api_name in {"is_fp8_gemm_enabled"} else True
        gemm_parsing = False if api_name in {"log_tensor_stats", "log_fp8_tensor_stats"} else True
        status, modified_config = feature_obj.parse_config_and_api(config, gemm_parsing=gemm_parsing, tensor_parsing=tensor_parsing, **kwargs)
        return status, modified_config

    def output_assertions_hook(self, api_name, ret, **kwargs):
        if api_name in {"process_tensor"}:
            assert "tensor" in ret, f"This API {api_name} must return a tensor."
        

        
        
        