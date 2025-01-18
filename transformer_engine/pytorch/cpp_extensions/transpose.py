# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for transpose extensions"""
from typing import Optional, Tuple, Union

import torch

import transformer_engine_torch as tex
from ..constants import TE_DType
from ._common import canonicalize_fp8_scales


__all__ = [
    "fp8_cast_transpose_bgrad_fused",
    "fp8_transpose_bgrad_fused",
]


def fp8_cast_transpose_bgrad_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast + Transpose + BGRAD with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
    )

    # Launch kernel
    return tex.fused_cast_transpose_bgrad(
        inp,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
        **fp8_scales_offsets,
    )


def fp8_transpose_bgrad_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: Optional[tex.FP8TensorMeta],
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None],
    otype: tex.DType,
    grad_bias_type: torch.dtype,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transpose + BGRAD with FP8 output"""

    # Get FP8 scaling factors
    fp8_scales, fp8_scales_offsets = canonicalize_fp8_scales(
        scale=scale,
        amax=amax,
        scale_inv=scale_inv,
        fp8_meta=fp8_meta_tensor,
        fp8_meta_index=fp8_tensor,
    )

    # Launch kernel
    return tex.fused_fp8_transpose_bgrad(
        inp,
        fp8_scales["scale"],
        fp8_scales["amax"],
        fp8_scales["scale_inv"],
        otype,
        TE_DType[grad_bias_type],
        **fp8_scales_offsets,
    )
