# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Data-free descriptions of quantized operation outputs."""

from __future__ import annotations
from typing import Optional

import torch

from .tensor import empty_tensor
from ..quantized_tensor import Quantizer


def fake_bgrad(
    grad_output: torch.Tensor,
    quantizer: Optional[Quantizer] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe bias reduction and optional gradient quantization, in native order."""
    db = empty_tensor(
        shape=(grad_output.shape[-1],), dtype=grad_output.dtype, device=grad_output.device
    )
    dx = (
        grad_output
        if quantizer is None
        else empty_tensor(
            shape=grad_output.shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
            quantizer=quantizer,
        )
    )
    return db, dx
