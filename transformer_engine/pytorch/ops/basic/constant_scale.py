# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for constant scaling."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ..op import BasicOperation, OperationContext
from .._common import maybe_dequantize
from ...tensor import Quantizer
from ...quantized_tensor import QuantizedTensorStorage
from ...dynamo import TensorSpec

TensorOrQuantized = Union[torch.Tensor, QuantizedTensorStorage]


@dataclass(slots=True)
class ConstantScaleFwdArgs:
    """Flat, ``self``-free inputs to the constant-scale forward."""

    input_: TensorOrQuantized
    scale: float


@dataclass(slots=True)
class ConstantScaleBwdArgs:
    """Flat inputs to the constant-scale backward."""

    grad_output: Optional[torch.Tensor] = None
    scale: Optional[float] = None


class ConstantScale(BasicOperation):
    """Multiply by a constant"""

    fwd_args_type = ConstantScaleFwdArgs
    bwd_args_type = ConstantScaleBwdArgs
    num_grad_inputs = 1

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    @classmethod
    def forward_compute(
        cls, args: ConstantScaleFwdArgs
    ) -> Tuple[torch.Tensor, Tuple[()], Dict[str, Any]]:
        x = maybe_dequantize(args.input_)
        return x * args.scale, (), {}

    @classmethod
    def forward_fake(
        cls, args: ConstantScaleFwdArgs
    ) -> Tuple[TensorSpec, Tuple[()], Dict[str, Any]]:
        x = args.input_
        out = TensorSpec(shape=tuple(x.shape), dtype=x.dtype, device=x.device)
        return out, (), {}

    @classmethod
    def backward_compute(cls, args: ConstantScaleBwdArgs) -> Tuple[torch.Tensor]:
        dy = maybe_dequantize(args.grad_output)
        return (dy * args.scale,)

    @classmethod
    def backward_fake(cls, args: ConstantScaleBwdArgs) -> Tuple[TensorSpec]:
        dy = args.grad_output
        return (TensorSpec(shape=tuple(dy.shape), dtype=dy.dtype, device=dy.device),)

    def resolve_fwd_args(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> ConstantScaleFwdArgs:
        del requires_grad, prev_op_grad_output_quantizer, next_op_input_quantizer
        return ConstantScaleFwdArgs(input_=input_, scale=self.scale)

    def resolve_bwd_args(
        self, ctx: OperationContext, grad_output: torch.Tensor
    ) -> ConstantScaleBwdArgs:
        del ctx
        return ConstantScaleBwdArgs(grad_output=grad_output, scale=self.scale)
