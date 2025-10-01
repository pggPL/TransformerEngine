# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LogFp8TensorStats Feature support for nvidia-dlframework-inspect"""


from transformer_engine.debug.features.log_quantized_tensor_stats import LogQuantizedTensorStats
from nvdlfw_inspect.registry import Registry


@Registry.register_feature(namespace="transformer_engine")
class LogFp8TensorStats(LogQuantizedTensorStats):
    """
        This class is functionally identical to LogQuantizedTensorStats and exists to ensure backward compatibility.
    """
