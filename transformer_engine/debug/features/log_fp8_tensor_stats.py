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

from transformer_engine.debug.features.utils.stats_computation import STATS
from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
from transformer_engine.pytorch.tensor._internal.float8_tensor_base  import Float8TensorBase
from transformer_engine.pytorch.tensor._internal.mxfp8_tensor_base  import MXFP8TensorBase


import nvdlfw_inspect.api as nvinspect_api
from nvdlfw_inspect.debug_features.log_tensor_stats import LogTensorStats as BaseLogTensorStats
from nvdlfw_inspect.registry import Registry, api_method
from nvdlfw_inspect.utils import append_parent_docstring


@Registry.register_feature(namespace="transformer_engine")
@append_parent_docstring(parent=BaseLogTensorStats)
class LogFp8TensorStats(BaseLogTensorStats):
    """
    Log FP8 tensor statistics in Transformer engine.

    APIs:
    1. transformer_engine.log_fp8_tensor_stats
    - When using this api, you would need to pass args:
    -- layer_name: this is matched with the layer description in the config file
    -- tensor_name: this is matched with one of the tensors in the config field, and passed as a kwarg. For example, tensor_name='tensor1'
    -- tensor: the tensor to process, and passed as a kwarg. For example, tensor={torch tensor}
    - Optional kwargs:
    -- skip_reduction (default: False): skip reduction of tensor stats across GPU ranks. Each GPU rank will log its local stats. 
    if skip_reduction is not set, this api only checks for DDP and reduces tensor on last rank.
    -- reduction_group: (default: None): Provide torch distributed process group for collecting tensor from GPU ranks that are part of this group.
    (For enabling tensor reduction across different parallelisms DP, TP, PP, etc)
    -- iteration: option to pass training step for logging. if using step() api of this tool in the training loop, this arg is not needed.

    Config:

    To enable the feature in yaml config:
    transformer_engine:
      log_fp8_tensor_stats:
        enabled: True
        feature_properties:
        ...

    Config fields:
    This feature works at a tensor level, you can set the following properties for each tensor:
    - stats: List[str], type of statistics to log. Options: {underflows, overflows}
    - freq: int, logging frequency in training steps. Default = 1.
    - start_step: int, train step to start logging. Default = 0.
    - end_step: int, train step to end logging. Default = -1 (don't stop logging once started).
    Tensor structure is described below:
    """
    def _get_supported_stats_list(self):
        return {"underflows", "overflows"}

    @api_method
    def save_stats_for_logging_quantized(self, config, layer_name, tensor_name, tensor, skip_reduction=False, reduction_group=None, iteration=None):
        assert type(tensor) in [Float8Tensor, Float8TensorBase, MXFP8Tensor, MXFP8TensorBase],\
            f"[NVTORCH INSPECT ERROR] Tensor {tensor_name} must be quantized tensor when using log_fp8_tensor_stats. Use log_tensor_stats for high precision tensors."

        tensor = tensor._data
        options = (config.get('start_step', None), config.get('end_step', None), config.get('start_end_list', None),)

        FP8GlobalStateManager.debug_tool = True
        
        stats_with_percent = []
        for stat in config['stats']:
            assert stat in ["underflows", "overflows"], f"[NVTORCH INSPECT ERROR] Statistic {stat} is not supported."
            stats_with_percent.append(stat + "%")
        stats = stats_with_percent

        for stat in stats:
            STATS_BUFFERS.try_add_buffer(layer_name, tensor_name, stat, options, reduction_group)

        if not self._check_params(config, layer_name, iteration=iteration):
            return {}

        iteration = super()._get_current_iteration(iteration=iteration)

        STATS_BUFFERS.feed(layer_name, tensor_name, stats, options, tensor, iteration, skip_reduction)