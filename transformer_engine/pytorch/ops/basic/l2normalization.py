# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusable operation for L2 Normalization."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import os

import torch

from ...torch_version import torch_version
from ...jit import (
    l2normalization_fused,
    l2normalization_fwd_fused,
    l2normalization_backward_fused,
    set_jit_fusion_options,
    warmup_jit_l2normalization_all_dtypes,
)
from ...tensor import Quantizer
from ...quantized_tensor import QuantizedTensorStorage
from ...dynamo import TensorSpec
from ..op import BasicOperation, OperationContext
from .._common import maybe_dequantize

TensorOrQuantized = Union[torch.Tensor, QuantizedTensorStorage]


@dataclass(slots=True)
class L2NormalizationFwdArgs:
    """Flat, ``self``-free inputs to the L2-normalization forward."""

    input_: TensorOrQuantized
    eps: float
    requires_grad: bool


@dataclass(slots=True)
class L2NormalizationBwdArgs:
    """Flat inputs to the L2-normalization backward."""

    grad_output: Optional[torch.Tensor] = None
    saved_input: Optional[TensorOrQuantized] = None
    rsqrt_norm: Optional[torch.Tensor] = None
    eps: Optional[float] = None


class L2Normalization(BasicOperation):
    r"""L2 Normalization

    Applies L2 normalization over the last dimension of input tensors.
    This is a parameter-free normalization that scales each vector to unit L2 norm.

    .. math::
        y = \frac{x}{\sqrt{\sum_{i} x_i^2 + \varepsilon}}

    This operation is used e.g. for query-key normalization in attention mechanisms.

    Parameters
    ----------
    eps : float, default = 1e-6
        A value added to the denominator for numerical stability
    seq_length : int, default = None
        sequence length of input samples. Needed for JIT Warmup, a technique where jit fused
        functions are warmed up before training to ensure same kernels are used for forward
        propagation and activation recompute phase.
    micro_batch_size : int, default = None
        batch size per training step. Needed for JIT Warmup, a technique where jit
        fused functions are warmed up before training to ensure same kernels are
        used for forward propagation and activation recompute phase.

    """

    def __init__(
        self,
        *,
        eps: float = 1e-6,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.eps: float = eps

        # JIT warmup for L2Normalization fused operations
        if seq_length and micro_batch_size:
            if (
                torch.cuda.is_available()
                and torch_version() >= (2, 0, 0)
                and bool(int(os.getenv("NVTE_TORCH_COMPILE", "1")))
            ):
                set_jit_fusion_options()
                # For L2Normalization, we don't know the hidden size until forward pass,
                # but we can warm up with common sizes. For QK normalization, this will be
                # the attention head dimension (hidden_size_per_attention_head), not the full
                # model hidden dimension. Common head dimensions are 32, 64, 80, 96, 128, 256.
                common_hidden_sizes = [32, 64, 80, 96, 128, 256]
                for hidden_size in common_hidden_sizes:
                    warmup_jit_l2normalization_all_dtypes(hidden_size, seq_length, micro_batch_size)

    fwd_args_type = L2NormalizationFwdArgs
    bwd_args_type = L2NormalizationBwdArgs
    num_grad_inputs = 1

    @classmethod
    def forward_compute(cls, args: L2NormalizationFwdArgs):
        # L2 norm: x / sqrt(sum(x^2) + eps) = x * rsqrt(sum(x^2) + eps)
        x = maybe_dequantize(args.input_)
        if args.requires_grad:
            y, rsqrt_norm = l2normalization_fwd_fused(x, args.eps)
            # x is derived from the input by a possibly-no-op dequantize, so it
            # may *be* the input, which a custom op may not return; the backward
            # rebuilds it from the operation's input instead.
            return y, (rsqrt_norm,), {"eps": args.eps}
        y = l2normalization_fused(x, args.eps)
        return y, (), {"eps": args.eps}

    @classmethod
    def forward_fake(cls, args: L2NormalizationFwdArgs):
        x = args.input_
        shape = tuple(x.shape)
        y = TensorSpec(shape=shape, dtype=x.dtype, device=x.device)
        saved = ()
        if args.requires_grad:
            rsqrt_norm = TensorSpec(shape=shape[:-1] + (1,), dtype=torch.float32, device=x.device)
            saved = (rsqrt_norm,)
        return y, saved, {"eps": args.eps}

    @classmethod
    def backward_compute(cls, args: L2NormalizationBwdArgs):
        x = maybe_dequantize(args.saved_input)
        dy = maybe_dequantize(args.grad_output)
        # Recalculates l2_norm_squared_eps from x
        return (l2normalization_backward_fused(dy, x, args.rsqrt_norm, args.eps),)

    @classmethod
    def backward_fake(cls, args: L2NormalizationBwdArgs):
        x = args.saved_input
        return (TensorSpec(shape=tuple(x.shape), dtype=x.dtype, device=x.device),)

    def saved_for_backward(self, saved: tuple, input_: torch.Tensor) -> tuple:
        return (input_, *saved)

    def resolve_fwd_args(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> L2NormalizationFwdArgs:
        del prev_op_grad_output_quantizer, next_op_input_quantizer
        return L2NormalizationFwdArgs(input_=input_, eps=self.eps, requires_grad=requires_grad)

    def resolve_bwd_args(
        self, ctx: OperationContext, grad_output: torch.Tensor
    ) -> L2NormalizationBwdArgs:
        x, rsqrt_norm = ctx.saved_tensors
        return L2NormalizationBwdArgs(
            grad_output=grad_output,
            saved_input=x,
            rsqrt_norm=rsqrt_norm,
            eps=ctx.eps,
        )
