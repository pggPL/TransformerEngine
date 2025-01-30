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


from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
from transformer_engine.pytorch.tensor._internal.float8_tensor_base import Float8TensorBase
from transformer_engine.pytorch.tensor._internal.mxfp8_tensor_base import MXFP8TensorBase
import nvdlfw_inspect.api as nvinspect_api
from transformer_engine.debug.debug_state import TEDebugState


from nvdlfw_inspect.debug_features.log_tensor_stats import LogTensorStats as BaseLogTensorStats
from nvdlfw_inspect.registry import Registry, api_method
from nvdlfw_inspect.utils import append_parent_docstring


@Registry.register_feature(namespace="transformer_engine")
class LogFp8TensorStats(BaseLogTensorStats):
    """
    Log FP8 tensor statistics in Transformer engine.

    Config:

    To enable the feature in yaml config:
    transformer_engine:
      log_tensor_stats:
        enabled: True
        ...

    Config fields:
    This feature works at a tensor level, you can set the following properties for each tensor:
    - stats: List[str], type of statistics to log. Options: {min, max, mean, std, l1_norm, l2_norm, cur_amax, dynamic_range}
    - freq: int, logging frequency in training steps. Default = 1.
    - start_step: int, train step to start logging. Default = 0.
    - end_step: int, train step to end logging. Default = -1 (don't stop logging once started).
    - tensors/tensors_struct: tensors list or tensors_struct - please look into the Transformer Engine Precision Debug Tools documentation for more information.
    """

    def _get_supported_stats_list(self):
        return {"underflows%", "overflows%"}

    @api_method
    def use_look_at_tensor_before_process(self, config, layer_name, tensor_name, iteration=None):
        return self._check_params(config, layer_name, iteration=iteration)

    @api_method
    def look_at_tensor_after_process(
        self, config, layer_name, tensor_name, tensor, rowwise, iteration
    ):
        assert type(tensor) in [Float8Tensor, Float8TensorBase, MXFP8Tensor, MXFP8TensorBase], (
            f"[NVTORCH INSPECT ERROR] Tensor {tensor_name} must be quantized tensor when using"
            " log_fp8_tensor_stats.                Use log_tensor_stats for high precision"
            " tensors."
        )

        tensor = tensor._data
        options = (
            config.get("start_step", None),
            config.get("end_step", None),
            config.get("start_end_list", None),
            "fp8",
        )
        skip_reduction = False
        reduction_group = nvinspect_api.get_tensor_reduction_group()
        if self.tensor_name == "weight":
            if TEDebugState.weight_tensor_tp_group_reduce:
                reduction_group = self.tp_group
            else:
                skip_reduction = True

        FP8GlobalStateManager.debug_tool = True

        for stat in config["stats"]:
            assert (
                stat in self._get_supported_stats_list()
            ), f"[NVTORCH INSPECT ERROR] Statistic {stat} is not supported."

        STATS_BUFFERS.try_add_buffer(
            layer_name, tensor_name, config["stats"], options, reduction_group
        )

        if not self._check_params(config, layer_name, iteration=iteration):
            return {}

        iteration = super()._get_current_iteration(iteration=iteration)

        STATS_BUFFERS.feed(layer_name, tensor_name, options, tensor, iteration, skip_reduction)

        nvinspect_api.log_message(
            f"Feature={self.__class__.__name__}, API=look_at_tensor_after_process: {tensor_name}",
            layer_name,
            extra_cachable_args=(tensor_name),
        )

    @api_method
    def step(self):
        STATS_BUFFERS.log_stats()
