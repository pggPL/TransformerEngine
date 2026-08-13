# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for reduce-scatter."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ...distributed import gather_along_first_dim
from .._common import maybe_dequantize
from ..op import BasicOperation, OperationContext
from ...tensor import Quantizer
from ...quantized_tensor import QuantizedTensorStorage
from ...dynamo import TensorSpec

TensorOrQuantized = Union[torch.Tensor, QuantizedTensorStorage]


@dataclass(slots=True)
class ReduceScatterFwdArgs:
    """Flat, ``self``-free inputs to the reduce-scatter forward."""

    input_: TensorOrQuantized
    process_group: Optional[torch.distributed.ProcessGroup]
    process_group_size: int


@dataclass(slots=True)
class ReduceScatterBwdArgs:
    """Flat inputs to the reduce-scatter backward."""

    grad_output: Optional[torch.Tensor] = None
    process_group: Optional[torch.distributed.ProcessGroup] = None
    process_group_size: Optional[int] = None


class ReduceScatter(BasicOperation):
    """Reduce-scatter tensor along outer dimension

    Equivalent to summing tensors from all processes and splitting
    along the first dimension.

    Parameters
    ----------
    process_group : torch.distributed.ProcessGroup, default = world group
        Process group for communication

    """

    fwd_args_type = ReduceScatterFwdArgs
    bwd_args_type = ReduceScatterBwdArgs
    num_grad_inputs = 1

    def __init__(
        self,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        self.process_group: Optional[torch.distributed.ProcessGroup] = process_group
        self.process_group_size: int = torch.distributed.get_world_size(process_group)

    @classmethod
    def forward_compute(
        cls, args: ReduceScatterFwdArgs
    ) -> Tuple[Optional[torch.Tensor], Tuple[()], Dict[str, Any]]:
        # Trivial case: the input, unchanged (a custom op may not return it).
        if args.process_group_size == 1:
            return None, (), {}

        # Tensor dimensions
        x = args.input_
        input_dims = x.size()
        if not input_dims or input_dims[0] % args.process_group_size != 0:
            raise RuntimeError(
                "Attempted to reduce-scatter a tensor "
                f"with shape={list(input_dims)} "
                f"over {args.process_group_size} processes"
            )
        output_dims = list(input_dims)
        output_dims[0] //= args.process_group_size

        # Perform reduce-scatter
        x = maybe_dequantize(x.contiguous())
        y = torch.empty(output_dims, dtype=x.dtype, device=x.device)
        torch.distributed.reduce_scatter_tensor(y, x, group=args.process_group)
        return y, (), {}

    @classmethod
    def forward_fake(
        cls, args: ReduceScatterFwdArgs
    ) -> Tuple[Optional[TensorSpec], Tuple[()], Dict[str, Any]]:
        if args.process_group_size == 1:
            return None, (), {}
        x = args.input_
        shape = tuple(x.shape)
        out_shape = (shape[0] // args.process_group_size, *shape[1:])
        return TensorSpec(shape=out_shape, dtype=x.dtype, device=x.device), (), {}

    @classmethod
    def backward_compute(cls, args: ReduceScatterBwdArgs) -> Tuple[Optional[torch.Tensor]]:
        # Trivial case: the incoming gradient, unchanged.
        if args.process_group_size == 1:
            return (None,)
        dx, _ = gather_along_first_dim(args.grad_output, args.process_group)
        return (dx,)

    @classmethod
    def backward_fake(cls, args: ReduceScatterBwdArgs) -> Tuple[Optional[TensorSpec]]:
        if args.process_group_size == 1:
            return (None,)
        dy = args.grad_output
        shape = tuple(dy.shape)
        dx_shape = (shape[0] * args.process_group_size, *shape[1:])
        return (TensorSpec(shape=dx_shape, dtype=dy.dtype, device=dy.device),)

    def resolve_fwd_args(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> ReduceScatterFwdArgs:
        del requires_grad, prev_op_grad_output_quantizer, next_op_input_quantizer
        return ReduceScatterFwdArgs(
            input_=input_,
            process_group=self.process_group,
            process_group_size=self.process_group_size,
        )

    def resolve_bwd_args(
        self, ctx: OperationContext, grad_output: torch.Tensor
    ) -> ReduceScatterBwdArgs:
        return ReduceScatterBwdArgs(
            grad_output=grad_output,
            process_group=self.process_group,
            process_group_size=self.process_group_size,
        )
