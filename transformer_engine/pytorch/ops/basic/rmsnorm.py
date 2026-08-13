# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusable operation for RMSNorm."""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
import math
import os
from typing import Any, Optional, Union

import torch

from transformer_engine_torch import rmsnorm_bwd, rmsnorm_fwd
from ...constants import TE_DType
from ...export import is_in_onnx_export_mode
from ...tensor import Quantizer
from ...quantized_tensor import QuantizedTensorStorage
from ...dynamo import TensorSpec
from ...utils import (
    canonicalize_device,
    canonicalize_dtype,
    devices_match,
)
from ..op import BasicOperation, OperationContext
from .._common import (
    get_fused_normalization_quantizer,
    maybe_autocast_dtype,
    maybe_dequantize,
)

TensorOrQuantized = Union[torch.Tensor, QuantizedTensorStorage]


@dataclass(slots=True)
class RMSNormFwdArgs:
    """Flat, ``self``-free inputs to the RMSNorm forward."""

    input_: TensorOrQuantized
    weight: torch.Tensor
    eps: float
    zero_centered_gamma: bool
    sm_margin: int
    dtype: torch.dtype
    output_quantizer: Optional[Quantizer]
    requires_grad: bool


@dataclass(slots=True)
class RMSNormBwdArgs:
    """Flat inputs to the RMSNorm backward."""

    grad_output: Optional[torch.Tensor] = None
    saved_input: Optional[TensorOrQuantized] = None
    rstdevs: Optional[torch.Tensor] = None
    weight: Optional[torch.Tensor] = None
    zero_centered_gamma: Optional[bool] = None
    sm_margin: Optional[int] = None
    dtype: Optional[torch.dtype] = None


class RMSNorm(BasicOperation):
    r"""Root Mean Square Layer Normalization

    Applies Root Mean Square Layer Normalization over a mini-batch of
    inputs as described in the paper
    `Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>`__ .

    .. math::
        y = \frac{x}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * \gamma

    :math:`\gamma` is a learnable affine transform parameter that
    matches the inner-most dimensions of the input tensor.

    Parameters
    ----------
    normalized_shape : int or iterable of int
        Inner dimensions of input tensor
    eps : float, default = 1e-5
        A value added to the denominator for numerical stability
    device : torch.device, default = default CUDA device
        Tensor device
    dtype : torch.dtype, default = default dtype
        Tensor datatype
    zero_centered_gamma : bool, default = False
        If ``True``, the :math:`\gamma` parameter is initialized to zero
        and the calculation changes to

            .. math::
                y = \frac{x}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * (1 + \gamma)

    sm_margin : int, default = 0
        Number of SMs to exclude when launching CUDA kernels. This
        helps overlap with other kernels, e.g. communication kernels.
        For more fine-grained control, provide a dict with the SM
        margin at each compute stage ("forward", "backward",
        "inference").

    """

    def __init__(
        self,
        normalized_shape: Iterable[int] | int,
        *,
        eps: float = 1e-5,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
        sm_margin: int = 0,
    ) -> None:
        super().__init__()
        self.eps: float = eps
        self.zero_centered_gamma: bool = zero_centered_gamma

        # Parameter shape
        if not isinstance(normalized_shape, Iterable):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)

        # Parameter device
        defer_param_init = False
        device = canonicalize_device(device)
        if device.type == "meta":
            defer_param_init = True

        # Initialize parameters if needed
        weight = torch.empty(
            normalized_shape,
            device=device,
            dtype=canonicalize_dtype(dtype),
        )
        weight = torch.nn.Parameter(weight)
        self.weight: torch.nn.Parameter
        self.register_parameter("weight", weight)
        if not defer_param_init:
            self.reset_parameters()

        # Number of SMs to exclude when launching CUDA kernels
        self._sm_margins: dict[str, int]
        if isinstance(sm_margin, dict):

            def getenv(name: str) -> int:
                return int(os.getenv(name, "0"))

            self._sm_margins = {
                "forward": sm_margin.get("forward", getenv("NVTE_FWD_LAYERNORM_SM_MARGIN")),
                "backward": sm_margin.get("backward", getenv("NVTE_BWD_LAYERNORM_SM_MARGIN")),
                "inference": sm_margin.get("inference", getenv("NVTE_INF_LAYERNORM_SM_MARGIN")),
            }
        else:

            def getenv(name: str) -> int:
                return int(os.getenv(name, str(sm_margin)))

            self._sm_margins = {
                "forward": getenv("NVTE_FWD_LAYERNORM_SM_MARGIN"),
                "backward": getenv("NVTE_BWD_LAYERNORM_SM_MARGIN"),
                "inference": getenv("NVTE_INF_LAYERNORM_SM_MARGIN"),
            }

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Parameter device
        weight = self.weight
        device = weight.device
        if device.type == "meta":
            device = canonicalize_device(None)

        # Initialize param buffers
        if not devices_match(weight.device, device):
            weight = torch.empty_like(weight, device=device)

        # Initialize values
        if self.zero_centered_gamma:
            torch.nn.init.zeros_(weight)
        else:
            torch.nn.init.ones_(weight)

        # Save updated parameter
        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight)
        self.weight = weight

    def pre_first_fuser_forward(self) -> None:
        super().pre_first_fuser_forward()
        if self.weight.device.type == "meta":
            self.reset_parameters()

    fwd_args_type = RMSNormFwdArgs
    bwd_args_type = RMSNormBwdArgs
    num_grad_inputs = 2  # grad input, grad weight

    @classmethod
    def forward_compute(cls, args: RMSNormFwdArgs):
        weight_dims = tuple(args.weight.size())
        input_dims = tuple(args.input_.size())
        inner_dim = math.prod(weight_dims)
        dtype = args.dtype
        x = maybe_dequantize(args.input_.contiguous(), dtype).view((-1, inner_dim))
        w = maybe_dequantize(args.weight, dtype).view((inner_dim,))

        y, _, rstdevs = rmsnorm_fwd(
            x,
            w,
            args.eps,
            None,
            args.output_quantizer,
            TE_DType[dtype],
            args.sm_margin,
            args.zero_centered_gamma,
        )
        out = y.view(input_dims)
        # x is a view of the (possibly no-op) dequantized input, which a custom
        # op may not return; the backward rebuilds it from the operation's input.
        saved = (rstdevs,) if args.requires_grad else ()
        return out, saved, {"dtype": dtype}

    @classmethod
    def forward_fake(cls, args: RMSNormFwdArgs):
        input_dims = tuple(args.input_.shape)
        inner_dim = math.prod(tuple(args.weight.shape))
        outer_dim = math.prod(input_dims) // inner_dim
        device = args.input_.device
        out = TensorSpec(
            shape=input_dims, dtype=args.dtype, quantizer=args.output_quantizer, device=device
        )
        saved = ()
        if args.requires_grad:
            saved = (TensorSpec(shape=(outer_dim,), dtype=torch.float32, device=device),)
        return out, saved, {"dtype": args.dtype}

    @classmethod
    def backward_compute(cls, args: RMSNormBwdArgs):
        weight_dims = tuple(args.weight.size())
        inner_dim = math.prod(weight_dims)
        dtype = args.dtype
        x = maybe_dequantize(args.saved_input.contiguous(), dtype).view((-1, inner_dim))
        dy = maybe_dequantize(args.grad_output.contiguous(), dtype).view(x.size())
        w = maybe_dequantize(args.weight, dtype).view((inner_dim,))

        dx, dw = rmsnorm_bwd(
            dy,
            x,
            args.rstdevs,
            w,
            args.sm_margin,
            args.zero_centered_gamma,
        )
        grad_input = dx.view(args.grad_output.size())
        grad_weight = dw.view(weight_dims)
        return grad_input, grad_weight

    @classmethod
    def backward_fake(cls, args: RMSNormBwdArgs):
        device = args.grad_output.device
        grad_input = TensorSpec(
            shape=tuple(args.grad_output.shape), dtype=args.dtype, device=device
        )
        grad_weight = TensorSpec(shape=tuple(args.weight.shape), dtype=args.dtype, device=device)
        return grad_input, grad_weight

    def saved_for_backward(self, saved: tuple, input_: torch.Tensor) -> tuple:
        return (input_, *saved)

    def resolve_fwd_args(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> RMSNormFwdArgs:
        del prev_op_grad_output_quantizer

        # Fall back to a high-precision output when fused quantization is unsupported.
        output_quantizer = get_fused_normalization_quantizer(next_op_input_quantizer)

        # Check tensor dims
        weight_dims = tuple(self.weight.size())
        input_dims = tuple(input_.size())
        if len(input_dims) < len(weight_dims) or input_dims[-len(weight_dims) :] != weight_dims:
            raise ValueError(
                f"Input tensor (shape={input_dims}) "
                f"and weight tensor (shape={weight_dims}) are not compatible"
            )

        return RMSNormFwdArgs(
            input_=input_,
            weight=self.weight,
            eps=self.eps,
            zero_centered_gamma=self.zero_centered_gamma,
            sm_margin=self._sm_margins["forward" if requires_grad else "inference"],
            dtype=maybe_autocast_dtype(default_dtype=self.weight.dtype),
            output_quantizer=output_quantizer,
            requires_grad=requires_grad,
        )

    def resolve_bwd_args(self, ctx: OperationContext, grad_output: torch.Tensor) -> RMSNormBwdArgs:
        saved_input, rstdevs = ctx.saved_tensors
        return RMSNormBwdArgs(
            grad_output=grad_output,
            saved_input=saved_input,
            rstdevs=rstdevs,
            weight=self.weight,
            zero_centered_gamma=self.zero_centered_gamma,
            sm_margin=self._sm_margins["backward"],
            dtype=ctx.dtype,
        )

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        **kwargs: Any,
    ) -> torch.Tensor:
        if is_in_onnx_export_mode():
            return self.op_onnx_forward(input_)
        return super().op_forward(
            ctx,
            input_,
            prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
            next_op_input_quantizer=next_op_input_quantizer,
            **kwargs,
        )

    def op_onnx_forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        """Every operand in this function has a defined ONNX translation."""
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight
        variance = input_.pow(2).mean(-1, keepdim=True)
        normalized = input_ * torch.rsqrt(variance + self.eps)
        return normalized * weight
