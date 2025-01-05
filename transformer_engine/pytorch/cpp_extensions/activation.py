# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for activation extensions"""
from typing import Optional, Union

import torch

import transformer_engine_torch as tex
from ._common import canonicalize_fp8_scales

__all__ = ["gelu", "relu", "reglu", "geglu", "swiglu", "qgelu", "srelu"]


def gelu(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GeLU with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
        allow_multiple_offsets=False,
    )

    # Launch kernel
    return tex.gelu(
        inp,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
    )


def relu(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ReLU with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
        allow_multiple_offsets=False,
    )

    # Launch kernel
    return tex.relu(
        inp,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
    )


def geglu(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GeGLU with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
        allow_multiple_offsets=False,
    )

    # Launch kernel
    return tex.geglu(
        inp,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
    )


def reglu(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ReGLU with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
        allow_multiple_offsets=False,
    )

    # Launch kernel
    return tex.reglu(
        inp,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
    )


def swiglu(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SwiGLU with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
        allow_multiple_offsets=False,
    )

    # Launch kernel
    return tex.swiglu(
        inp,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
    )


def qgelu(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """QuickGELU with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
        allow_multiple_offsets=False,
    )

    # Launch kernel
    return tex.qgelu(
        inp,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
    )


def srelu(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ReLU with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
        allow_multiple_offsets=False,
    )

    # Launch kernel
    return tex.srelu(
        inp,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
    )
