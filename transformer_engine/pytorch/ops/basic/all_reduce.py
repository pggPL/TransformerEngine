# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for all-reduce."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .._common import maybe_dequantize
from ..op import BasicOperation, OperationContext
from ...tensor import Quantizer
from ...quantized_tensor import QuantizedTensorStorage
from ...dynamo import TensorSpec

TensorOrQuantized = Union[torch.Tensor, QuantizedTensorStorage]


@dataclass(slots=True)
class AllReduceFwdArgs:
    """Flat, ``self``-free inputs to the all-reduce forward."""

    input_: TensorOrQuantized
    process_group: Optional[torch.distributed.ProcessGroup]
    process_group_size: int


@dataclass(slots=True)
class AllReduceBwdArgs:
    """Flat inputs to the all-reduce backward."""

    grad_output: Optional[torch.Tensor] = None


class AllReduce(BasicOperation):
    """All-reduce tensor

    Equivalent to summing tensors from all processes. It is assumed
    that the output is used in operations that are redundantly
    computed on all processes, and hence that gradients are identical
    between processes.

    Parameters
    ----------
    process_group : torch.distributed.ProcessGroup, default = world group
        Process group for communication

    """

    fwd_args_type = AllReduceFwdArgs
    bwd_args_type = AllReduceBwdArgs
    num_grad_inputs = 1

    def __init__(
        self,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        reduce_in_backward: bool = True,
    ) -> None:
        super().__init__()
        self.process_group: Optional[torch.distributed.ProcessGroup] = process_group
        self._reduce_in_backward: bool = reduce_in_backward

    @classmethod
    def forward_compute(
        cls, args: AllReduceFwdArgs
    ) -> Tuple[Optional[torch.Tensor], Tuple[()], Dict[str, Any]]:
        # Trivial case: the input, unchanged (a custom op may not return it).
        if args.process_group_size == 1:
            return None, (), {}

        # All-reduce into a fresh buffer: the eager implementation reduced the
        # dequantized tensor in place, which for a plain contiguous input is the
        # input itself -- a custom op may neither mutate nor return its inputs.
        x = maybe_dequantize(args.input_.contiguous())
        if x is args.input_:
            x = x.clone()
        torch.distributed.all_reduce(x, group=args.process_group)
        return x, (), {}

    @classmethod
    def forward_fake(
        cls, args: AllReduceFwdArgs
    ) -> Tuple[Optional[TensorSpec], Tuple[()], Dict[str, Any]]:
        if args.process_group_size == 1:
            return None, (), {}
        x = args.input_
        return TensorSpec(shape=tuple(x.shape), dtype=x.dtype, device=x.device), (), {}

    @classmethod
    def backward_compute(cls, args: AllReduceBwdArgs) -> Tuple[Optional[torch.Tensor]]:
        del args
        return (None,)

    @classmethod
    def backward_fake(cls, args: AllReduceBwdArgs) -> Tuple[Optional[TensorSpec]]:
        del args
        return (None,)

    def resolve_fwd_args(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> AllReduceFwdArgs:
        del requires_grad, prev_op_grad_output_quantizer, next_op_input_quantizer
        return AllReduceFwdArgs(
            input_=input_,
            process_group=self.process_group,
            process_group_size=torch.distributed.get_world_size(self.process_group),
        )

    def resolve_bwd_args(
        self, ctx: OperationContext, grad_output: torch.Tensor
    ) -> AllReduceBwdArgs:
        return AllReduceBwdArgs(grad_output=grad_output)
