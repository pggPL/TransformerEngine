# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for dropout."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import transformer_engine_torch as tex
from ...tensor import Quantizer
from ...tensor.storage.float8_tensor_storage import Float8TensorStorage
from ...quantized_tensor import QuantizedTensorStorage
from ...dynamo import TensorSpec
from .._common import maybe_autocast_dtype, maybe_dequantize
from ..op import BasicOperation, OperationContext

TensorOrQuantized = Union[torch.Tensor, QuantizedTensorStorage]


@dataclass(slots=True)
class DropoutFwdArgs:
    """Flat, ``self``-free inputs to the dropout forward."""

    input_: TensorOrQuantized
    dtype: torch.dtype
    impl: str
    dropout_probability: float
    requires_grad: bool


@dataclass(slots=True)
class DropoutBwdArgs:
    """Flat inputs to the dropout backward."""

    grad_output: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    dtype: Optional[torch.dtype] = None
    impl: Optional[str] = None
    dropout_probability: Optional[float] = None


class Dropout(BasicOperation):
    """Randomly zero out tensor entries during training

    During training, tensor entries are randomly set to zero with
    probability :math:`p` and remaining entries are scaled by
    :math:`1/(1-p)`.

    """

    fwd_args_type = DropoutFwdArgs
    bwd_args_type = DropoutBwdArgs
    num_grad_inputs = 1

    def __init__(self, p: float) -> None:
        super().__init__()
        self.dropout_probability: float = p

    @classmethod
    def forward_compute(
        cls, args: DropoutFwdArgs
    ) -> Tuple[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], ...], Dict[str, Any]]:
        impl = args.impl
        ctx_attrs = {
            "impl": impl,
            "dtype": args.dtype,
            "dropout_probability": args.dropout_probability,
        }
        if impl == "evaluation":
            # The input, unchanged; a custom op may not return its own input.
            return None, (None,) if args.requires_grad else (), ctx_attrs
        mask: torch.Tensor
        if impl == "fused":
            x = args.input_
            if not isinstance(x, Float8TensorStorage):
                x = maybe_dequantize(x, dtype=args.dtype)
            out, mask = tex.dropout_fwd(x, args.dropout_probability)
        elif impl == "unfused":
            x = maybe_dequantize(args.input_, dtype=args.dtype)
            keep_prob = 1 - args.dropout_probability
            mask = torch.empty_like(x)
            mask.bernoulli_(keep_prob)
            mask *= 1 / keep_prob
            out = x * mask
        else:
            raise ValueError(f"Unsupported forward implementation {impl}")
        return out, (mask,) if args.requires_grad else (), ctx_attrs

    @classmethod
    def forward_fake(
        cls, args: DropoutFwdArgs
    ) -> Tuple[Optional[TensorSpec], Tuple[Optional[TensorSpec], ...], Dict[str, Any]]:
        x = args.input_
        ctx_attrs = {
            "impl": args.impl,
            "dtype": args.dtype,
            "dropout_probability": args.dropout_probability,
        }
        if args.impl == "evaluation":
            return None, (None,) if args.requires_grad else (), ctx_attrs
        shape = tuple(x.shape)
        numel = 1
        for d in shape:
            numel *= d
        out = TensorSpec(shape=shape, dtype=args.dtype, device=x.device)
        if args.impl == "fused":
            mask = TensorSpec(shape=(numel // 8,), dtype=torch.uint8, device=x.device)
        else:
            mask = TensorSpec(shape=shape, dtype=args.dtype, device=x.device)
        return out, (mask,) if args.requires_grad else (), ctx_attrs

    @classmethod
    def backward_compute(cls, args: DropoutBwdArgs) -> Tuple[Optional[torch.Tensor]]:
        if args.impl == "evaluation":
            return (None,)
        dy = maybe_dequantize(args.grad_output, dtype=args.dtype)
        if args.impl == "fused":
            return (tex.dropout_bwd(dy, args.mask, args.dropout_probability),)
        if args.impl == "unfused":
            return (dy * args.mask,)
        raise ValueError(f"Unsupported backward implementation {args.impl}")

    @classmethod
    def backward_fake(cls, args: DropoutBwdArgs) -> Tuple[Optional[TensorSpec]]:
        if args.impl == "evaluation":
            return (None,)
        dy = args.grad_output
        return (TensorSpec(shape=tuple(dy.shape), dtype=args.dtype, device=dy.device),)

    def resolve_fwd_args(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> DropoutFwdArgs:
        del prev_op_grad_output_quantizer, next_op_input_quantizer
        dtype = maybe_autocast_dtype(default_dtype=input_.dtype)
        if not self.training:
            impl = "evaluation"
        elif input_.numel() % 16 == 0 and dtype in (torch.float16, torch.bfloat16):
            impl = "fused"
        else:
            impl = "unfused"
        return DropoutFwdArgs(
            input_=input_,
            dtype=dtype,
            impl=impl,
            dropout_probability=self.dropout_probability,
            requires_grad=requires_grad,
        )

    def resolve_bwd_args(self, ctx: OperationContext, grad_output: torch.Tensor) -> DropoutBwdArgs:
        (mask,) = ctx.saved_tensors
        return DropoutBwdArgs(
            grad_output=grad_output,
            mask=mask,
            dtype=ctx.dtype,
            impl=ctx.impl,
            dropout_probability=ctx.dropout_probability,
        )
