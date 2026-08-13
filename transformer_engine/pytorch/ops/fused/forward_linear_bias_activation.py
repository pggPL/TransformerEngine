# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for forward GEMM + bias + activation."""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...quantization import FP8GlobalStateManager
from ...tensor import Quantizer
from ..basic import BasicLinear, Bias
from ..basic.basic_linear import (
    BasicLinearFwdArgs,
    _saved_input_is_fresh,
    _saved_weight_is_fresh,
)
from .._common import is_quantized_tensor
from ..op import FusedOperation, FusibleOperation, OperationContext


@dataclass(slots=True)
class ForwardLinearBiasActivationFwdArgs(BasicLinearFwdArgs):
    """Flat inputs to the fused linear+bias forward."""

    bias: Optional[torch.Tensor] = None


@dataclass(slots=True)
class ForwardLinearBiasActivationBwdArgs:
    """Placeholder: this fused op has no backward of its own.

    The backward pass walks the basic operations' own custom ops -- that is the
    point of pipeline-level autograd -- so this container exists only to
    satisfy the registration's schema contract.
    """

    grad_output: Optional[torch.Tensor] = None


class ForwardLinearBiasActivation(FusedOperation):
    """Fused forward GEMM + bias + activation

    Bias and activation are both optional. Row tensor parallelism is
    not supported since that requires communication immediately after
    the GEMM.

    """

    def __init__(
        self,
        *,
        linear: BasicLinear,
        bias: Optional[Bias],
        activation: None,
    ) -> None:

        # Basic operations that comprise this fused operation
        op_idxs = {"linear": 0, "bias": None, "activation": None}
        ops = [linear]
        if bias is not None:
            op_idxs["bias"] = len(ops)
            ops.append(bias)
        if activation is not None:
            op_idxs["activation"] = len(ops)
            ops.append(activation)

        # Initialize base class
        super().__init__(ops)

        # Index of each basic operations
        self._op_idxs: dict[str, Optional[int]] = op_idxs

    fwd_args_type = ForwardLinearBiasActivationFwdArgs
    bwd_args_type = ForwardLinearBiasActivationBwdArgs
    num_grad_inputs = 1  # never used; the backward walks the basic ops

    @classmethod
    def forward_compute(cls, args: ForwardLinearBiasActivationFwdArgs):
        output, x_local, w = BasicLinear._functional_forward(
            input=args.input_,
            weight=args.weight,
            bias=args.bias,
            dtype=args.dtype,
            tensor_parallel_mode=args.tensor_parallel_mode,
            tensor_parallel_group=args.tensor_parallel_group,
            sequence_parallel=args.sequence_parallel,
            with_quantized_compute=args.with_quantized_compute,
            backward_override=args.backward_override,
            input_quantizer=args.input_quantizer,
            weight_quantizer=args.weight_quantizer,
            output_quantizer=args.output_quantizer,
            input_requires_grad=args.input_requires_grad,
            weight_requires_grad=args.weight_requires_grad,
        )
        saved_input = (
            x_local if _saved_input_is_fresh(args, is_quantized_tensor(args.input_)) else None
        )
        saved_weight = w if _saved_weight_is_fresh(args, is_quantized_tensor(args.weight)) else None
        return output, (saved_input, saved_weight), cls._nest_ctx_attrs(args)

    @classmethod
    def forward_fake(cls, args: ForwardLinearBiasActivationFwdArgs):
        out, saved, _ = BasicLinear.forward_fake(args)
        return out, saved, cls._nest_ctx_attrs(args)

    @staticmethod
    def _nest_ctx_attrs(args: ForwardLinearBiasActivationFwdArgs) -> Dict[str, Any]:
        """Per-basic-op context attrs; ``scatter_ctx`` unpacks them."""
        bias_grad_input_quantizer = args.grad_output_quantizer
        if args.backward_override is not None:
            bias_grad_input_quantizer = None
        return {
            "linear": {
                "with_quantized_compute": (
                    args.with_quantized_compute and args.backward_override is None
                ),
                "backward_override": args.backward_override,
                "input_quantizer": args.input_quantizer,
                "weight_quantizer": args.weight_quantizer,
                "grad_output_quantizer": args.grad_output_quantizer,
                "grad_input_quantizer": args.grad_input_quantizer,
                "dtype": args.dtype,
                "input_requires_grad": args.input_requires_grad,
                "weight_requires_grad": args.weight_requires_grad,
            },
            "bias": {"grad_input_quantizer": bias_grad_input_quantizer},
        }

    @classmethod
    def backward_compute(cls, args):
        raise RuntimeError(
            "ForwardLinearBiasActivation has no backward of its own; "
            "the backward pass walks the basic operations."
        )

    @classmethod
    def backward_fake(cls, args):
        raise RuntimeError(
            "ForwardLinearBiasActivation has no backward of its own; "
            "the backward pass walks the basic operations."
        )

    def resolve_fuser_fwd_args(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> ForwardLinearBiasActivationFwdArgs:
        del basic_op_kwargs  # the gate rejected any; Bias takes none
        linear_op = self.basic_ops[self._op_idxs["linear"]]
        linear_op_ctx = basic_op_ctxs[self._op_idxs["linear"]]
        bias = None
        if self._op_idxs["bias"] is not None:
            bias = self.basic_ops[self._op_idxs["bias"]].bias

        input_requires_grad = linear_op_ctx.requires_grad
        weight_requires_grad = linear_op_ctx.requires_grad and linear_op.weight.requires_grad

        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
        if with_quantized_compute:
            backward_override = FP8GlobalStateManager.get_fp8_recipe().backward_override
        else:
            backward_override = None

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = linear_op.weight.dtype

        return ForwardLinearBiasActivationFwdArgs(
            input_=input_,
            weight=linear_op.weight,
            dtype=dtype,
            tensor_parallel_mode=linear_op.tensor_parallel_mode,
            tensor_parallel_group=linear_op.tensor_parallel_group,
            sequence_parallel=linear_op.sequence_parallel,
            with_quantized_compute=with_quantized_compute,
            backward_override=backward_override,
            input_quantizer=linear_op.get_quantizer("forward", 0),
            weight_quantizer=linear_op.get_quantizer("forward", 1),
            output_quantizer=next_op_input_quantizer,
            grad_output_quantizer=linear_op.get_quantizer("backward", 0),
            grad_input_quantizer=prev_op_grad_output_quantizer,
            input_requires_grad=input_requires_grad,
            weight_requires_grad=weight_requires_grad,
            bias=bias,
        )

    def scatter_ctx(
        self,
        basic_op_ctxs: list[OperationContext],
        saved: tuple,
        ctx_attrs: dict[str, Any],
        input_: torch.Tensor,
    ) -> None:
        linear_op = self.basic_ops[self._op_idxs["linear"]]
        linear_op_ctx = basic_op_ctxs[self._op_idxs["linear"]]
        if linear_op_ctx.requires_grad:
            linear_op_ctx.save_for_backward(*linear_op.saved_for_backward(saved, input_))
            for name, value in ctx_attrs["linear"].items():
                setattr(linear_op_ctx, name, value)
        if self._op_idxs["bias"] is not None:
            bias_op_ctx = basic_op_ctxs[self._op_idxs["bias"]]
            if bias_op_ctx.requires_grad:
                for name, value in ctx_attrs["bias"].items():
                    setattr(bias_op_ctx, name, value)

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:

        # Get basic operations
        idx = self._op_idxs["linear"]
        linear_op = self.basic_ops[idx]
        linear_op_ctx = basic_op_ctxs[idx]
        if self._op_idxs["bias"] is None:
            bias_op = None
            bias_op_ctx = None
            bias = None
        else:
            idx = self._op_idxs["bias"]
            bias_op = self.basic_ops[idx]
            bias_op_ctx = basic_op_ctxs[idx]
            bias = bias_op.bias
            if basic_op_kwargs[idx]:
                raise ValueError("Bias operation forward does not expect keyword arguments")
        if self._op_idxs["activation"] is None:
            activation_op = None  # pylint: disable=unused-variable
        else:
            raise NotImplementedError("Activations are not yet supported")

        # Check which grads are required
        input_requires_grad = linear_op_ctx.requires_grad
        weight_requires_grad = linear_op_ctx.requires_grad and linear_op.weight.requires_grad

        # Quantizers
        input_quantizer = linear_op.get_quantizer("forward", 0)
        weight_quantizer = linear_op.get_quantizer("forward", 1)
        output_quantizer = next_op_input_quantizer
        grad_output_quantizer = linear_op.get_quantizer("backward", 0)
        grad_input_quantizer = prev_op_grad_output_quantizer
        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
        if with_quantized_compute:
            backward_override = FP8GlobalStateManager.get_fp8_recipe().backward_override
        else:
            backward_override = None

        # Get autocast dtype if needed
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = linear_op.weight.dtype

        # Linear forward
        output, x_local, w = BasicLinear._functional_forward(
            input=input_,
            weight=linear_op.weight,
            bias=bias,
            dtype=dtype,
            tensor_parallel_mode=linear_op.tensor_parallel_mode,
            tensor_parallel_group=linear_op.tensor_parallel_group,
            sequence_parallel=linear_op.sequence_parallel,
            with_quantized_compute=with_quantized_compute,
            backward_override=backward_override,
            input_quantizer=input_quantizer,
            weight_quantizer=weight_quantizer,
            output_quantizer=output_quantizer,
            input_requires_grad=input_requires_grad,
            weight_requires_grad=weight_requires_grad,
        )

        # Save state for backward pass
        if linear_op_ctx.requires_grad:
            if backward_override == "high_precision":
                saved_input = input_ if weight_requires_grad else None
                saved_weight = linear_op.weight if input_requires_grad else None
            else:
                saved_input = x_local
                saved_weight = w
            if is_cpu_offload_enabled():
                mark_activation_offload(saved_input)
            linear_op_ctx.save_for_backward(saved_input, saved_weight)
            linear_op_ctx.with_quantized_compute = (
                with_quantized_compute and backward_override is None
            )
            linear_op_ctx.backward_override = backward_override
            linear_op_ctx.input_quantizer = input_quantizer
            linear_op_ctx.weight_quantizer = weight_quantizer
            linear_op_ctx.grad_output_quantizer = grad_output_quantizer
            linear_op_ctx.grad_input_quantizer = grad_input_quantizer
            linear_op_ctx.dtype = dtype
            linear_op_ctx.input_requires_grad = input_requires_grad
            linear_op_ctx.weight_requires_grad = weight_requires_grad
        if bias_op is not None and bias_op_ctx.requires_grad:
            bias_op_ctx.grad_input_quantizer = linear_op.get_grad_output_quantizer()
            if backward_override is not None:
                bias_op_ctx.grad_input_quantizer = None

        return output, [() for _ in range(len(self.basic_ops))]

    @staticmethod
    def fuse_forward_ops(
        ops: list[FusibleOperation],
        **unused,  # pylint: disable=unused-argument
    ) -> list[FusibleOperation]:
        """Apply operation fusion for forward pass.

        Parameters
        ----------
        ops : list of FusibleOperation
            Forward pass operations.

        Returns
        -------
        ops : list of FusibleOperation
            Updated forward pass operations

        """

        # Scan through ops, fusing if possible
        out = []
        window, ops = ops[:2], ops[2:]
        while len(window) == 2:

            # Check if window matches pattern
            matches_pattern = True
            if not (isinstance(window[0], BasicLinear) and isinstance(window[1], Bias)):
                matches_pattern = False
            elif window[0].tensor_parallel_mode == "row":
                # Row tensor-parallelism requires communication after
                # the GEMM
                matches_pattern = False
            elif window[0].weight.dtype not in (torch.float16, torch.bfloat16):
                # cuBLAS only supports fused GEMM+bias+activation with
                # FP16 and BF16 output
                matches_pattern = False

            if matches_pattern:
                # Construct fused op if window matches pattern
                op = ForwardLinearBiasActivation(
                    linear=window[0],
                    bias=window[1],
                    activation=None,
                )
                window = [op]
            else:
                # Shift window if window doesn't match pattern
                out.extend(window[:-1])
                window = window[-1:]

            # Adjust window to expected size
            out.extend(window[:-2])
            window = window[-2:]
            while ops and len(window) < 2:
                window.append(ops[0])
                ops = ops[1:]

        # Return list of ops
        out.extend(window)
        return out
