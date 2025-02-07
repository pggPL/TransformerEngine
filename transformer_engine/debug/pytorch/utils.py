# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utils functions for the debug module."""


def any_feature_enabled(quantizers):
    """Returns True if at least one API call is made from DebugQuantizer."""
    for q in quantizers:
        if q.any_feature_enabled():
            return True
    return False


class DebugQuantizerBase:
    """
    Parent class to DebugQuantizer. It is used to recoginze if object is instance of
    DebugQuantizer withouth the loading of the debuq_quantization.py
    """


class DebugQuantizedTensorBase:
    """
    Parent class to DebugQuantizedTensor. It is used to recoginze if object is instance of
    DebugQuantizedTensor withouth the loading of the debuq_quantization.py
    """
