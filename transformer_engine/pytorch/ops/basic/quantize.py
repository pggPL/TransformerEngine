# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for quantization."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ...quantization import FP8GlobalStateManager
from .._common import is_quantized_tensor
from ..op import BasicOperation, OperationContext
from ...tensor import Quantizer
from ...quantized_tensor import QuantizedTensorStorage
from ...dynamo import TensorSpec

TensorOrQuantized = Union[torch.Tensor, QuantizedTensorStorage]


def _is_quantized(x: Any) -> bool:
    """Quantized-ness of a value that may be a tensor or its ``TensorSpec``."""
    if isinstance(x, TensorSpec):
        return x.quantizer is not None
    return is_quantized_tensor(x)


@dataclass(slots=True)
class QuantizeFwdArgs:
    """Flat, ``self``-free inputs to the quantize forward."""

    input_: TensorOrQuantized
    forward_quantizer: Optional[Quantizer]
    quantize_backward: bool
    backward_quantizer: Optional[Quantizer]


@dataclass(slots=True)
class QuantizeBwdArgs:
    """Flat inputs to the quantize backward."""

    grad_output: Optional[torch.Tensor] = None
    backward_quantizer: Optional[Quantizer] = None


class Quantize(BasicOperation):
    """Quantize tensor data

    Uses recipe from ``autocast`` context. When called outside
    of an ``autocast`` context, this is an identity operation.

    Parameters
    ----------
    forward : bool, default = True
        Perform quantization in forward pass
    backward : bool, default = False
        Perform quantization in backward pass

    """

    fwd_args_type = QuantizeFwdArgs
    bwd_args_type = QuantizeBwdArgs
    num_grad_inputs = 1

    def __init__(
        self,
        forward: bool = True,
        backward: bool = False,
    ) -> None:
        super().__init__()
        self._quantize_forward = forward
        self._quantize_backward = backward

    def num_quantizers(self, mode: str) -> int:
        if mode == "forward" and self._quantize_forward:
            return 1
        if mode == "backward" and self._quantize_backward:
            return 1
        return 0

    @classmethod
    def forward_compute(
        cls, args: QuantizeFwdArgs
    ) -> Tuple[Optional[TensorOrQuantized], Tuple[()], Dict[str, Any]]:
        ctx_attrs = {
            "backward_quantizer": args.backward_quantizer if args.quantize_backward else None
        }
        quantizer = args.forward_quantizer
        if quantizer is not None and not is_quantized_tensor(args.input_):
            return quantizer(args.input_), (), ctx_attrs
        # The input, unchanged; a custom op may not return its own input.
        return None, (), ctx_attrs

    @classmethod
    def forward_fake(
        cls, args: QuantizeFwdArgs
    ) -> Tuple[Optional[TensorSpec], Tuple[()], Dict[str, Any]]:
        ctx_attrs = {
            "backward_quantizer": args.backward_quantizer if args.quantize_backward else None
        }
        x = args.input_
        quantizer = args.forward_quantizer
        if quantizer is not None and not _is_quantized(x):
            out = TensorSpec(
                shape=tuple(x.shape), dtype=x.dtype, quantizer=quantizer, device=x.device
            )
            return out, (), ctx_attrs
        return None, (), ctx_attrs

    @classmethod
    def backward_compute(cls, args: QuantizeBwdArgs) -> Tuple[Optional[torch.Tensor]]:
        quantizer = args.backward_quantizer
        if quantizer is not None and not is_quantized_tensor(args.grad_output):
            return (quantizer(args.grad_output),)
        return (None,)

    @classmethod
    def backward_fake(cls, args: QuantizeBwdArgs) -> Tuple[Optional[TensorSpec]]:
        dy = args.grad_output
        quantizer = args.backward_quantizer
        if quantizer is not None and not _is_quantized(dy):
            return (
                TensorSpec(
                    shape=tuple(dy.shape), dtype=dy.dtype, quantizer=quantizer, device=dy.device
                ),
            )
        return (None,)

    def resolve_fwd_args(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> QuantizeFwdArgs:
        del requires_grad, prev_op_grad_output_quantizer, next_op_input_quantizer
        fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
        quantize_forward = fp8_enabled and self._quantize_forward
        quantize_backward = fp8_enabled and self._quantize_backward
        if fp8_enabled:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            quantize_backward = quantize_backward and recipe.backward_override is None
        return QuantizeFwdArgs(
            input_=input_,
            forward_quantizer=self.get_quantizer("forward", 0) if quantize_forward else None,
            quantize_backward=quantize_backward,
            backward_quantizer=self.get_quantizer("backward", 0) if quantize_backward else None,
        )

    def resolve_bwd_args(self, ctx: OperationContext, grad_output: torch.Tensor) -> QuantizeBwdArgs:
        return QuantizeBwdArgs(
            grad_output=grad_output,
            backward_quantizer=ctx.backward_quantizer,
        )
