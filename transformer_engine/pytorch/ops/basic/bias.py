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
from ...dynamo import TensorSpec

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

    fwd_args_type = BiasFwdArgs
    bwd_args_type = BiasBwdArgs
    num_grad_inputs = 2  # grad input, grad bias

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

    @classmethod
    def forward_compute(cls, args: BiasFwdArgs) -> Tuple[torch.Tensor, Tuple[()], Dict[str, Any]]:
        """Add the bias. Saves no tensors; the backward only needs the quantizer."""
        x = args.input_
        b = args.bias.view([1] * (x.dim() - 1) + [args.local_size])
        return x + b, (), {"grad_input_quantizer": args.grad_input_quantizer}

    @classmethod
    def forward_fake(cls, args: BiasFwdArgs) -> Tuple[TensorSpec, Tuple[()], Dict[str, Any]]:
        x = args.input_
        out = TensorSpec(shape=tuple(x.shape), dtype=x.dtype, device=x.device)
        return out, (), {"grad_input_quantizer": args.grad_input_quantizer}

    @classmethod
    def backward_compute(cls, args: BiasBwdArgs) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Reduce the gradient over every dimension but the inner one.

        Returns ``None`` for the grad input when it is ``grad_output``
        unchanged: a custom op may not return one of its own inputs, and cloning
        would cost a full-size copy on the common (unquantized) path.
        """
        dy = args.grad_output
        if dy.dim() > 1:
            quantizer = args.grad_input_quantizer
            if quantizer is None:
                return None, dy.sum(tuple(range(dy.dim() - 1)))
            db, dy = tex.bgrad_quantize(dy, quantizer)
            return dy, db
        return None, dy

    @classmethod
    def backward_fake(cls, args: BiasBwdArgs) -> Tuple[Optional[TensorSpec], TensorSpec]:
        dy = args.grad_output
        shape = tuple(dy.shape)
        if len(shape) > 1:
            grad_bias = TensorSpec(shape=(shape[-1],), dtype=dy.dtype, device=dy.device)
            quantizer = args.grad_input_quantizer
            if quantizer is None:
                return None, grad_bias
            grad_input = TensorSpec(
                shape=shape, dtype=dy.dtype, quantizer=quantizer, device=dy.device
            )
            return grad_input, grad_bias
        return None, TensorSpec(shape=shape, dtype=dy.dtype, device=dy.device)

    def resolve_fwd_args(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> BiasFwdArgs:
        """Gather everything the forward needs into a flat, self-free container.

        Reads module config and global FP8 state, so it must run in the traced
        region (where Dynamo guards those reads), never inside the custom op.

        Bias never quantizes its output, so ``next_op_input_quantizer`` is
        accepted (every operation resolves through the same signature) and
        ignored.
        """
        del next_op_input_quantizer
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

    def resolve_bwd_args(self, ctx: OperationContext, grad_output: torch.Tensor) -> BiasBwdArgs:
        return BiasBwdArgs(
            grad_output=grad_output,
            grad_input_quantizer=ctx.grad_input_quantizer,
        )
