# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for all-gather."""

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
class AllGatherFwdArgs:
    """Flat, ``self``-free inputs to the all-gather forward."""

    input_: TensorOrQuantized
    process_group: Optional[torch.distributed.ProcessGroup]
    process_group_size: int


@dataclass(slots=True)
class AllGatherBwdArgs:
    """Flat inputs to the all-gather backward."""

    grad_output: Optional[torch.Tensor] = None
    process_group: Optional[torch.distributed.ProcessGroup] = None
    process_group_size: Optional[int] = None


class AllGather(BasicOperation):
    """All-gather tensor along outer dimension

    Equivalent to gathering tensors from all processes and
    concatenating along the first dimension.

    Parameters
    ----------
    process_group : torch.distributed.ProcessGroup, default = world group
        Process group for communication

    """

    fwd_args_type = AllGatherFwdArgs
    bwd_args_type = AllGatherBwdArgs
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
        cls, args: AllGatherFwdArgs
    ) -> Tuple[Optional[torch.Tensor], Tuple[()], Dict[str, Any]]:
        # Trivial case: the input, unchanged (a custom op may not return it).
        if args.process_group_size == 1:
            return None, (), {}
        out, _ = gather_along_first_dim(args.input_, args.process_group)
        return out, (), {}

    @classmethod
    def forward_fake(
        cls, args: AllGatherFwdArgs
    ) -> Tuple[Optional[TensorSpec], Tuple[()], Dict[str, Any]]:
        if args.process_group_size == 1:
            return None, (), {}
        x = args.input_
        shape = tuple(x.shape)
        out_shape = (shape[0] * args.process_group_size, *shape[1:])
        return TensorSpec(shape=out_shape, dtype=x.dtype, device=x.device), (), {}

    @classmethod
    def backward_compute(cls, args: AllGatherBwdArgs) -> Tuple[Optional[torch.Tensor]]:
        # Trivial case: the incoming gradient, unchanged.
        if args.process_group_size == 1:
            return (None,)

        # Tensor dimensions
        dy = args.grad_output
        output_dims = dy.size()
        if not output_dims or output_dims[0] % args.process_group_size != 0:
            raise RuntimeError(
                "Attempted to reduce-scatter a tensor "
                f"with shape={list(output_dims)} "
                f"over {args.process_group_size} processes"
            )
        input_dims = list(output_dims)
        input_dims[0] //= args.process_group_size

        # Perform reduce-scatter
        dy = maybe_dequantize(dy.contiguous())
        dx = torch.empty(input_dims, dtype=dy.dtype, device=dy.device)
        torch.distributed.reduce_scatter_tensor(dx, dy, group=args.process_group)
        return (dx,)

    @classmethod
    def backward_fake(cls, args: AllGatherBwdArgs) -> Tuple[Optional[TensorSpec]]:
        if args.process_group_size == 1:
            return (None,)
        dy = args.grad_output
        shape = tuple(dy.shape)
        dx_shape = (shape[0] // args.process_group_size, *shape[1:])
        return (TensorSpec(shape=dx_shape, dtype=dy.dtype, device=dy.device),)

    def resolve_fwd_args(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> AllGatherFwdArgs:
        del requires_grad, prev_op_grad_output_quantizer, next_op_input_quantizer
        return AllGatherFwdArgs(
            input_=input_,
            process_group=self.process_group,
            process_group_size=self.process_group_size,
        )

    def resolve_bwd_args(
        self, ctx: OperationContext, grad_output: torch.Tensor
    ) -> AllGatherBwdArgs:
        return AllGatherBwdArgs(
            grad_output=grad_output,
            process_group=self.process_group,
            process_group_size=self.process_group_size,
        )
