# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from transformer_engine.debug.features.api import TEConfigAPIMapper

import nvdlfw_inspect.api as nvinspect_api
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
            if key != "gemm":
                raise ValueError(f'[NVTORCH INSPECT ERROR] Unexpected key in config: "{key}".')

        return False
