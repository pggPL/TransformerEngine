# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for bias."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

import transformer_engine_torch as tex
from ...quantization import FP8GlobalStateManager
from ..op import BasicOperation, OperationContext
from ...utils import canonicalize_device, canonicalize_dtype
from ...tensor import Quantizer
from ...quantized_tensor import QuantizedTensorStorage
from ...dynamo import TensorSpec, register_op_halves


TensorOrQuantized = Union[torch.Tensor, QuantizedTensorStorage]


@dataclass(slots=True)
class BiasFwdArgs:
    """Flat, ``self``-free inputs to the bias forward."""

    input_: TensorOrQuantized
    bias: torch.Tensor
    local_size: int
    grad_input_quantizer: Optional[Quantizer]


@dataclass(slots=True)
class BiasBwdArgs:
    """Flat inputs to the bias backward."""

    grad_output: Optional[torch.Tensor] = None
    grad_input_quantizer: Optional[Quantizer] = None


def _bias_forward_impl(
    args: BiasFwdArgs,
) -> Tuple[torch.Tensor, Tuple[()], Dict[str, Any]]:
    """Bias forward. Saves no tensors; backward only needs the quantizer."""
    x = args.input_
    b = args.bias.view([1] * (x.dim() - 1) + [args.local_size])
    return x + b, (), {"grad_input_quantizer": args.grad_input_quantizer}


def _bias_forward_impl_fake(
    args: BiasFwdArgs,
) -> Tuple[TensorSpec, Tuple[()], Dict[str, Any]]:
    """Allocation-free fake of :func:`_bias_forward_impl`."""
    x = args.input_
    out = TensorSpec(shape=tuple(x.shape), dtype=x.dtype, device=x.device)
    return out, (), {"grad_input_quantizer": args.grad_input_quantizer}


def _bias_backward_impl(
    args: BiasBwdArgs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bias backward: reduce the grad over all but the inner dimension."""
    dy = args.grad_output
    if dy.dim() > 1:
        quantizer = args.grad_input_quantizer
        if quantizer is None:
            db = dy.sum(tuple(range(dy.dim() - 1)))
        else:
            db, dy = tex.bgrad_quantize(dy, quantizer)
    else:
        db = dy
    return dy, db


def _bias_backward_impl_fake(
    args: BiasBwdArgs,
) -> Tuple[TensorSpec, TensorSpec]:
    """Allocation-free fake of :func:`_bias_backward_impl`.

    Mirrors its branching: with a quantizer the grad input is quantized in place
    of the reduction, otherwise both grads stay in high precision.
    """
    dy = args.grad_output
    shape = tuple(dy.shape)
    quantizer = args.grad_input_quantizer if len(shape) > 1 else None
    grad_bias_shape = (shape[-1],) if len(shape) > 1 else shape
    grad_input = TensorSpec(
        shape=shape, dtype=dy.dtype, quantizer=quantizer, device=dy.device
    )
    grad_bias = TensorSpec(shape=grad_bias_shape, dtype=dy.dtype, device=dy.device)
    return grad_input, grad_bias


_bias_ops = register_op_halves(
    op_name="bias",
    fwd_arg_type=BiasFwdArgs,
    fwd_impl=_bias_forward_impl,
    fwd_fake_impl=_bias_forward_impl_fake,
    bwd_arg_type=BiasBwdArgs,
    bwd_impl=_bias_backward_impl,
    bwd_fake_impl=_bias_backward_impl_fake,
    num_grad_inputs=2,
)


class Bias(BasicOperation):
    """Apply additive bias

    This is equivalent to the additive bias in ``torch.nn.Linear``.

    Parameters
    ----------
    size : int
        Inner dimension of input tensor
    device : torch.device, default = default CUDA device
        Tensor device
    dtype : torch.dtype, default = default dtype
        Tensor datatype
    tensor_parallel : bool, default = False
        Whether to distribute input tensor and bias tensors along
        inner dimension
    tensor_parallel_group : torch.distributed.ProcessGroup, default = world group
        Process group for tensor parallelism

    """

    def __init__(
        self,
        size: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel: bool = False,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()

        # Bias size
        self._size = size

        # Bias tensor device
        defer_param_init = False
        device = canonicalize_device(device)
        if device.type == "meta":
            defer_param_init = True
            device = canonicalize_device(None)
        self.device: torch.device = device

        # Tensor parallel configuration
        tensor_parallel_size = 1
        local_size = size
        if tensor_parallel:
            tensor_parallel_size = torch.distributed.get_world_size(tensor_parallel_group)
            tensor_parallel = tensor_parallel_size > 1
            if size % tensor_parallel_size != 0:
                raise ValueError(
                    "Invalid configuration for tensor parallelism "
                    f"({size=}, {tensor_parallel_size=})"
                )
            local_size //= tensor_parallel_size
        else:
            tensor_parallel_group = None
        self.tensor_parallel: bool = tensor_parallel
        self.tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = tensor_parallel_group
        self.tensor_parallel_size: int = tensor_parallel_size
        self.local_size: int = local_size

        # Initialize parameters if needed
        bias = torch.empty(
            local_size,
            device="meta",
            dtype=canonicalize_dtype(dtype),
        )
        bias = torch.nn.Parameter(bias)
        self.bias: torch.nn.Parameter
        self.register_parameter("bias", bias)
        if not defer_param_init:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Make sure parameter is initialized
        bias = self.bias
        if bias.device.type != "cuda":
            bias = torch.empty_like(bias, device=self.device)
        else:
            bias = bias.to(device=self.device)

        # Initialize values
        bias.zero_()

        # Save updated parameter
        if not isinstance(bias, torch.nn.Parameter):
            bias = torch.nn.Parameter(bias)
        self.bias = bias

    def pre_first_fuser_forward(self) -> None:
        super().pre_first_fuser_forward()
        if self.bias.device.type == "meta":
            self.reset_parameters()

    def resolve_fwd_args(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer],
    ) -> BiasFwdArgs:
        """Gather everything the forward needs into a flat, self-free container.

        Reads module config and global FP8 state, so it must run in the traced
        region (where Dynamo guards those reads), never inside the custom op.
        """
        grad_input_quantizer = None
        if requires_grad:
            grad_input_quantizer = prev_op_grad_output_quantizer
            if FP8GlobalStateManager.is_fp8_enabled():
                if FP8GlobalStateManager.get_fp8_recipe().backward_override is not None:
                    grad_input_quantizer = None
        return BiasFwdArgs(
            input_=input_,
            bias=self.bias,
            local_size=self.local_size,
            grad_input_quantizer=grad_input_quantizer,
        )

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        del next_op_input_quantizer  # Bias never quantizes its output
        args = self.resolve_fwd_args(
            input_,
            requires_grad=ctx.requires_grad,
            prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
        )
        out, saved, ctx_attrs = _bias_forward_impl(args)
        if ctx.requires_grad:
            ctx.save_for_backward(*saved)
            for name, value in ctx_attrs.items():
                setattr(ctx, name, value)
        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        grad_input, grad_bias = _bias_backward_impl(
            BiasBwdArgs(
                grad_output=grad_output,
                grad_input_quantizer=ctx.grad_input_quantizer,
            )
        )
        return grad_input, (grad_bias,)
