# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LogFp8TensorStats Feature support for nvidia-dlframework-inspect"""


from nvdlfw_inspect.debug_features.log_quantized_tensor_stats import LogQuantizedTensorStats
from nvdlfw_inspect.registry import Registry



@Registry.register_feature(namespace="transformer_engine")
class LogFp8TensorStats(LogQuantizedTensorStats):
    """
    The same as LogQuantizedTensorStats, for backward compatibility.
    """
