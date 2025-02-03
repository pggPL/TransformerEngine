# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

from transformer_engine.debug.features.utils.stats_computation import STATS
from nvdlfw_inspect.debug_features.log_tensor_stats import LogTensorStats as BaseLogTensorStats
from nvdlfw_inspect.registry import Registry, api_method
from nvdlfw_inspect.utils import append_parent_docstring
import nvdlfw_inspect.api as nvinspect_api
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
from transformer_engine.pytorch.tensor._internal.float8_tensor_base import Float8TensorBase
from transformer_engine.pytorch.tensor._internal.mxfp8_tensor_base import MXFP8TensorBase
from transformer_engine.debug.debug_state import TEDebugState

from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager


@Registry.register_feature(namespace="transformer_engine")
class LogTensorStats(BaseLogTensorStats):
    """
    Log tensor statistics in transformer engine.

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

    def _get_supported_stats_list():
        return BaseLogTensorStats._get_supported_stats_list(None) | {"cur_amax", "dynamic_range"}

    @api_method
    def use_look_at_tensor_before_process(self, config, layer_name, tensor_name, iteration):
        return self._check_params(config, layer_name, iteration=iteration)

    @api_method
    def look_at_tensor_before_process(
        self, config, layer_name, tensor_name, tensor, rowwise, iteration
    ):
        assert (
            type(tensor) not in [Float8Tensor, Float8TensorBase, MXFP8Tensor, MXFP8TensorBase]
            and tensor.dtype != torch.uint8
        ), (
            f"[NVTORCH INSPECT ERROR] Tensor {tensor_name} must be in high precision when using"
            " log_tensor_stats. Use log_fp8_tensor_stats for FP8 tensors."
        )
        if not rowwise:
            return None # tensor was already seen rowwise in the other gemm
        FP8GlobalStateManager.debug_tool = True
        options = (
            config.get("start_step", None),
            config.get("end_step", None),
            config.get("start_end_list", None),
        )
        skip_reduction = False
        reduction_group = nvinspect_api.get_tensor_reduction_group()
        if tensor_name == "weight":
            if TEDebugState.weight_tensor_tp_group_reduce:
                pass
                # reduction_group = self.tp_group
            else:
                skip_reduction = True

        STATS_BUFFERS.try_add_buffer(
            layer_name, tensor_name, config["stats"], options, reduction_group
        )

        if not self._check_params(config, layer_name, iteration=iteration):
            return {}
        for stat in config["stats"]:
            assert (
                stat in STATS.keys()
            ), f"[NVTORCH INSPECT ERROR] Statistic {stat} is not supported."

        iteration = super()._get_current_iteration(iteration=iteration)
        STATS_BUFFERS.feed(layer_name, tensor_name, options, tensor, iteration, skip_reduction)

        nvinspect_api.log_message(
            f"Feature={self.__class__.__name__}, API=look_at_tensor_before_process: {tensor_name}",
            layer_name,
            extra_cachable_args=(tensor_name),
        )

    @api_method
    def step(self):
        STATS_BUFFERS.log_stats()
