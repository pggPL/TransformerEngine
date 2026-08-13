# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for SwiGLU and variants."""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch

import transformer_engine_torch as tex
from ...constants import DType
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...tensor import Float8CurrentScalingQuantizer, Quantizer
from ...quantized_tensor import QuantizedTensorStorage
from ...dynamo import TensorSpec
from ...utils import clear_tensor_data
from ..op import BasicOperation, OperationContext
from .._common import maybe_dequantize

__all__ = ["SwiGLU", "ClampedSwiGLU", "ScaledSwiGLU", "ScaledClampedQGeGLU"]

TensorOrQuantized = Union[torch.Tensor, QuantizedTensorStorage]


def _deinterleave_glu(t: torch.Tensor, interleave_size: int) -> torch.Tensor:
    """Convert block-interleaved gates/units into the concatenated layout."""
    shape = t.size()
    t = t.reshape(-1, shape[-1] // (2 * interleave_size), 2, interleave_size)
    t = t.transpose(1, 2).contiguous()
    return t.view(shape)


def _interleave_glu(t: torch.Tensor, interleave_size: int) -> torch.Tensor:
    """Inverse of :func:`_deinterleave_glu`."""
    shape = t.size()
    t = t.reshape(-1, 2, shape[-1] // (2 * interleave_size), interleave_size)
    t = t.transpose(1, 2).contiguous()
    return t.view(shape)


@dataclass(slots=True)
class SwiGLUFwdArgs:
    """Flat, ``self``-free inputs to the SwiGLU forward."""

    input_: TensorOrQuantized
    dtype: torch.dtype
    output_quantizer: Optional[Quantizer]
    cache_quantized_input: bool
    glu_interleave_size: Optional[int]
    requires_grad: bool
    prev_op_grad_output_quantizer: Optional[Quantizer]


@dataclass(slots=True)
class SwiGLUBwdArgs:
    """Flat inputs to the SwiGLU backward."""

    grad_output: Optional[torch.Tensor] = None
    saved_input: Optional[TensorOrQuantized] = None
    dtype: Optional[torch.dtype] = None
    glu_interleave_size: Optional[int] = None
    grad_input_quantizer: Optional[Quantizer] = None


@dataclass(slots=True)
class ClampedSwiGLUFwdArgs(SwiGLUFwdArgs):
    """Flat, ``self``-free inputs to the clamped-SwiGLU forward."""

    limit: float = 7.0
    alpha: float = 1.702
    glu_linear_offset: float = 1.0


@dataclass(slots=True)
class ClampedSwiGLUBwdArgs(SwiGLUBwdArgs):
    """Flat inputs to the clamped-SwiGLU backward."""

    limit: Optional[float] = None
    alpha: Optional[float] = None
    glu_linear_offset: Optional[float] = None


class SwiGLU(BasicOperation):
    r"""Swish gated linear unit

    The input tensor is split into chunks :math:``a`` and :math:``b``
    along the last dimension and the following is computed:

    .. math::

       \text{SwiGLU}(a,b) = \text{SiLU}(a) * b

    where

    .. math::

       \text{SiLU}(x) = x \sigma(x) = \frac{x}{1+\exp(-x)}

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:``a`` and
       :math:``b``. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    The Sigmoid Linear Unit (SiLU) gating function is also known as
    the swish function. See
    `GLU Variants Improve Transformer <https://arxiv.org/abs/2002.05202>`__.

    Parameters
    ----------
    cache_quantized_input : bool, default = False
        Quantize input tensor when caching for use in the backward
        pass. This will typically reduce memory usage but require
        extra compute and increase numerical error. This feature is
        highly experimental.
    glu_interleave_size : int, optional
        When set, the GLU activations will use a block interleaved
        format. Instead of interpreting the input tensor as a
        concatenation of gates and linear units (e.g.
        :math:``[a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4]``
        in the above notation), it will be interpreted
        as alternating blocks of gates and linear units (e.g.
        :math:``[a_1, a_2, b_1, b_2, a_3, a_4, b_3, b_4]``
        when the interleave size is 2). This data format is highly
        experiental and is primarily intended to support some advanced
        fused kernels.

    """

    def __init__(
        self,
        *,
        cache_quantized_input: bool = False,
        glu_interleave_size: Optional[int] = None,
    ):
        super().__init__()
        self.cache_quantized_input: bool = cache_quantized_input
        self.glu_interleave_size: Optional[int] = glu_interleave_size

    fwd_args_type = SwiGLUFwdArgs
    bwd_args_type = SwiGLUBwdArgs
    num_grad_inputs = 1

    @classmethod
    def forward_compute(cls, args: SwiGLUFwdArgs):
        x = maybe_dequantize(args.input_.contiguous(), args.dtype)

        swiglu_in = x
        if args.glu_interleave_size is not None:
            swiglu_in = _deinterleave_glu(swiglu_in, args.glu_interleave_size)

        out = tex.swiglu(swiglu_in, args.output_quantizer)

        if args.cache_quantized_input:
            input_quantizer = Float8CurrentScalingQuantizer(DType.kFloat8E4M3, x.device)
            input_quantizer.set_usage(rowwise=True, columnwise=False)
            x = input_quantizer(x)
        # Only the re-quantized input is handed back; otherwise x may *be* the
        # input (dequantize + contiguous are no-ops for a plain contiguous
        # tensor), which a custom op may not return. The backward rebuilds it
        # from the operation's input instead.
        saved = (x,) if (args.requires_grad and args.cache_quantized_input) else ()
        return out, saved, cls._ctx_attrs(args)

    @classmethod
    def forward_fake(cls, args: SwiGLUFwdArgs):
        x = args.input_
        shape = tuple(x.shape)
        out = TensorSpec(
            shape=(*shape[:-1], shape[-1] // 2),
            dtype=args.dtype,
            quantizer=args.output_quantizer,
            device=x.device,
        )
        saved = ()
        if args.requires_grad and args.cache_quantized_input:
            saved_quantizer = Float8CurrentScalingQuantizer(DType.kFloat8E4M3, x.device)
            saved_quantizer.set_usage(rowwise=True, columnwise=False)
            saved = (
                TensorSpec(
                    shape=shape, dtype=args.dtype, quantizer=saved_quantizer, device=x.device
                ),
            )
        return out, saved, cls._ctx_attrs(args)

    @staticmethod
    def _ctx_attrs(args: SwiGLUFwdArgs) -> Dict[str, Any]:
        return {
            "dtype": args.dtype,
            "prev_op_grad_output_quantizer": args.prev_op_grad_output_quantizer,
        }

    @classmethod
    def backward_compute(cls, args: SwiGLUBwdArgs):
        x = maybe_dequantize(args.saved_input.contiguous(), args.dtype)
        dy = maybe_dequantize(args.grad_output.contiguous(), args.dtype)

        swiglu_in = x
        quantizer = args.grad_input_quantizer
        if args.glu_interleave_size is not None:
            swiglu_in = _deinterleave_glu(swiglu_in, args.glu_interleave_size)
            quantizer = None

        dx = tex.dswiglu(dy, swiglu_in, quantizer)
        if args.glu_interleave_size is not None:
            dx = _interleave_glu(dx, args.glu_interleave_size)
        return (dx,)

    @classmethod
    def backward_fake(cls, args: SwiGLUBwdArgs):
        x = args.saved_input
        quantizer = args.grad_input_quantizer
        if args.glu_interleave_size is not None:
            quantizer = None
        return (
            TensorSpec(
                shape=tuple(x.shape), dtype=args.dtype, quantizer=quantizer, device=x.device
            ),
        )

    def saved_for_backward(self, saved: tuple, input_: torch.Tensor) -> tuple:
        return saved or (input_,)

    def resolve_fwd_args(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> SwiGLUFwdArgs:
        dtype: torch.dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = input_.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(f"Unsupported dtype ({dtype})")
        return SwiGLUFwdArgs(
            input_=input_,
            dtype=dtype,
            output_quantizer=next_op_input_quantizer,
            cache_quantized_input=self.cache_quantized_input,
            glu_interleave_size=self.glu_interleave_size,
            requires_grad=requires_grad,
            prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
        )

    def resolve_bwd_args(self, ctx: OperationContext, grad_output: torch.Tensor) -> SwiGLUBwdArgs:
        (saved_input,) = ctx.saved_tensors
        return SwiGLUBwdArgs(
            grad_output=grad_output,
            saved_input=saved_input,
            dtype=ctx.dtype,
            glu_interleave_size=self.glu_interleave_size,
            grad_input_quantizer=ctx.prev_op_grad_output_quantizer,
        )


class ClampedSwiGLU(BasicOperation):
    r"""GPT-OSS
    Implementation based on `GPT-OSS <https://github.com/openai/gpt-oss/blob/a0a84273e9e0c14a233cb9befdfd159c2bcfa6cd/gpt_oss/torch/model.py#L250>`__.

    This activation has two differences compared to the original SwiGLU
       1. Both gate and pre-activations are clipped based on parameter limit.
       2. Activation uses sigmoid(alpha * x) instead of sigmoid(x) used in Swish activation.

    .. warning::

       The input tensor is chunked along the last dimension to get
       gates/pre-activations which is different from GPT OSS
       implementation where the gates/pre-activations are assumed to
       be interleaved in the input tensor.

    Parameters
    ----------
    limit : float
        The clamp limit.
    alpha : float
        The scaling factor for the sigmoid function used in the activation.
    glu_linear_offset : float
        Offset added to the linear (gate) component after clamping.
        Set to ``0.0`` to disable the offset.
    cache_quantized_input : bool, default = ``False``
        Quantize input tensor when caching for use in the backward pass.
    glu_interleave_size : int, optional
        When set, the GLU activations will use an experimental block
        interleaved format. See the corresponding option in the SwiGLU
        operation for more details.

    """

    def __init__(
        self,
        *,
        limit: float = 7.0,
        alpha: float = 1.702,
        glu_linear_offset: float = 1.0,
        cache_quantized_input: bool = False,
        glu_interleave_size: Optional[int] = None,
    ):
        super().__init__()
        self.limit: float = limit
        self.alpha: float = alpha
        self.glu_linear_offset: float = glu_linear_offset
        self.cache_quantized_input: bool = cache_quantized_input
        self.glu_interleave_size: Optional[int] = glu_interleave_size

    def _tex_clamped_swiglu_forward(
        self,
        swiglu_in: torch.Tensor,
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        """Call :func:`tex.clamped_swiglu` with this op's ``limit`` / ``alpha`` / ``glu_linear_offset``."""
        return tex.clamped_swiglu(
            swiglu_in,
            next_op_input_quantizer,
            self.limit,
            self.alpha,
            self.glu_linear_offset,
        )

    def _tex_clamped_dswiglu(
        self,
        dy: torch.Tensor,
        swiglu_in: torch.Tensor,
        quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        """Call :func:`tex.clamped_dswiglu` with this op's ``limit`` / ``alpha`` / ``glu_linear_offset``."""
        return tex.clamped_dswiglu(
            dy,
            swiglu_in,
            quantizer,
            self.limit,
            self.alpha,
            self.glu_linear_offset,
        )

    fwd_args_type = ClampedSwiGLUFwdArgs
    bwd_args_type = ClampedSwiGLUBwdArgs
    num_grad_inputs = 1

    @classmethod
    def forward_compute(cls, args: ClampedSwiGLUFwdArgs):
        x = maybe_dequantize(args.input_.contiguous(), args.dtype)

        swiglu_in = x
        if args.glu_interleave_size is not None:
            swiglu_in = _deinterleave_glu(swiglu_in, args.glu_interleave_size)

        out = tex.clamped_swiglu(
            swiglu_in,
            args.output_quantizer,
            args.limit,
            args.alpha,
            args.glu_linear_offset,
        )

        if args.cache_quantized_input:
            input_quantizer = Float8CurrentScalingQuantizer(DType.kFloat8E4M3, x.device)
            input_quantizer.set_usage(rowwise=True, columnwise=False)
            x = input_quantizer(x)
        # Same saved-tensor rule as SwiGLU: only a freshly quantized input may
        # be handed back; otherwise the backward rebuilds it from the op input.
        saved = (x,) if (args.requires_grad and args.cache_quantized_input) else ()
        return out, saved, SwiGLU._ctx_attrs(args)

    @classmethod
    def forward_fake(cls, args: ClampedSwiGLUFwdArgs):
        return SwiGLU.forward_fake(args)

    @classmethod
    def backward_compute(cls, args: ClampedSwiGLUBwdArgs):
        x = maybe_dequantize(args.saved_input.contiguous(), args.dtype)
        dy = maybe_dequantize(args.grad_output.contiguous(), args.dtype)

        swiglu_in = x
        quantizer = args.grad_input_quantizer
        if args.glu_interleave_size is not None:
            swiglu_in = _deinterleave_glu(swiglu_in, args.glu_interleave_size)
            quantizer = None

        dx = tex.clamped_dswiglu(
            dy,
            swiglu_in,
            quantizer,
            args.limit,
            args.alpha,
            args.glu_linear_offset,
        )
        if args.glu_interleave_size is not None:
            dx = _interleave_glu(dx, args.glu_interleave_size)
        return (dx,)

    @classmethod
    def backward_fake(cls, args: ClampedSwiGLUBwdArgs):
        return SwiGLU.backward_fake(args)

    def saved_for_backward(self, saved: tuple, input_: torch.Tensor) -> tuple:
        return saved or (input_,)

    def resolve_fwd_args(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> ClampedSwiGLUFwdArgs:
        dtype: torch.dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = input_.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(f"Unsupported dtype ({dtype})")
        return ClampedSwiGLUFwdArgs(
            input_=input_,
            dtype=dtype,
            output_quantizer=next_op_input_quantizer,
            cache_quantized_input=self.cache_quantized_input,
            glu_interleave_size=self.glu_interleave_size,
            requires_grad=requires_grad,
            prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
            limit=self.limit,
            alpha=self.alpha,
            glu_linear_offset=self.glu_linear_offset,
        )

    def resolve_bwd_args(
        self, ctx: OperationContext, grad_output: torch.Tensor
    ) -> ClampedSwiGLUBwdArgs:
        (saved_input,) = ctx.saved_tensors
        return ClampedSwiGLUBwdArgs(
            grad_output=grad_output,
            saved_input=saved_input,
            dtype=ctx.dtype,
            glu_interleave_size=self.glu_interleave_size,
            grad_input_quantizer=ctx.prev_op_grad_output_quantizer,
            limit=self.limit,
            alpha=self.alpha,
            glu_linear_offset=self.glu_linear_offset,
        )


class _ScaledGLU(BasicOperation):
    """SwiGLU-family activation with per-row scales (fused grouped MLP middle op)."""

    num_extra_inputs: int = 1

    def __init__(
        self,
        glu_interleave_size: Optional[int] = None,
        *,
        activation_recompute_in_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.glu_interleave_size: Optional[int] = glu_interleave_size
        self.activation_recompute_in_mlp: bool = activation_recompute_in_mlp

    def _glu_forward(self, swiglu_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _glu_backward(
        self,
        grad_swiglu_out: torch.Tensor,
        swiglu_in: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def op_forward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_forward` instead of `op_forward`."
        )

    def op_backward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_backward` instead of `op_backward`."
        )

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
        if self.activation_recompute_in_mlp:
            raise RuntimeError(
                f"{self.__class__.__name__}(activation_recompute_in_mlp=True) requires the "
                "fused grouped MLP path."
            )

        extra_input = basic_op_extra_inputs[0][0]

        # Determine compute dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        elif isinstance(input_, torch.Tensor):
            dtype = input_.dtype
        else:
            dtype = extra_input.dtype

        # Make sure inputs are in correct dtype
        input_ = maybe_dequantize(input_, dtype)
        scales = maybe_dequantize(extra_input, dtype)

        # Remove gate interleaving if needed
        swiglu_in = input_
        if self.glu_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.glu_interleave_size),
                2,
                self.glu_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        swiglu_out = self._glu_forward(swiglu_in)
        out = swiglu_out * scales.unsqueeze(-1)

        # Save state for backward pass
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(input_)
            ctx.input_requires_grad = True
            ctx.extra_input_requires_grad = extra_input.requires_grad
            ctx.dtype = dtype
            ctx.save_for_backward(
                input_,
                scales if ctx.input_requires_grad else None,
            )

        return out, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        if self.activation_recompute_in_mlp:
            raise RuntimeError(
                f"{self.__class__.__name__}(activation_recompute_in_mlp=True) requires the "
                "fused grouped MLP path."
            )

        ctx = basic_op_ctxs[0]
        input_, scales = ctx.saved_tensors
        input_ = maybe_dequantize(input_, ctx.dtype)
        if scales is not None:
            scales = maybe_dequantize(scales, ctx.dtype)
        grad_output = maybe_dequantize(grad_output, ctx.dtype)

        # Remove gate interleaving if needed
        swiglu_in = input_
        if self.glu_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.glu_interleave_size),
                2,
                self.glu_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        # Compute input grad
        grad_input = None
        if ctx.input_requires_grad:
            grad_swiglu_out = grad_output * scales.unsqueeze(-1)
            grad_swiglu_in = self._glu_backward(grad_swiglu_out, swiglu_in)
            grad_input = grad_swiglu_in
            if self.glu_interleave_size is not None:
                shape = grad_input.size()
                grad_input = grad_input.reshape(
                    -1,
                    2,
                    shape[-1] // (2 * self.glu_interleave_size),
                    self.glu_interleave_size,
                )
                grad_input = grad_input.transpose(1, 2).contiguous()
                grad_input = grad_input.view(shape)

        # Compute scales grad by recomputing GLU
        grad_extra_input = None
        if ctx.extra_input_requires_grad:
            swiglu_out = self._glu_forward(swiglu_in)
            grad_extra_input = torch.linalg.vecdot(swiglu_out, grad_output)

        # Clear input tensor if possible
        clear_tensor_data(ctx.saved_tensors[0])  # input_

        return grad_input, [()], [(grad_extra_input,)]


class ScaledSwiGLU(_ScaledGLU):
    r"""SwiGLU with post-scaling (matches cuDNN grouped GEMM ``act_func="swiglu"``).

    If the GLU output has shape ``(d_1, ..., d_n)``, it is multiplied
    with an extra input tensor of shape ``(d_1, ..., d_{n-1})``.

    Parameters
    ----------
    glu_interleave_size : int, optional
        When set, the GLU activations will use an experimental block
        interleaved format. See the corresponding option in the SwiGLU
        operation for more details.
    activation_recompute_in_mlp : bool, default = ``False``
        Enable fused grouped MLP kernels to recompute activation outputs
        during backward when supported instead of saving them.

    """

    def _glu_forward(self, swiglu_in: torch.Tensor) -> torch.Tensor:
        return tex.swiglu(swiglu_in, None)

    def _glu_backward(
        self,
        grad_swiglu_out: torch.Tensor,
        swiglu_in: torch.Tensor,
    ) -> torch.Tensor:
        return tex.dswiglu(grad_swiglu_out, swiglu_in, None)


class ScaledClampedQGeGLU(_ScaledGLU):
    r"""Clamped QGeGLU with post-scaling
    (matches cuDNN grouped GEMM ``act_func="geglu"``).

    Same layout and scaling contract as :class:`ScaledSwiGLU`, but the GLU
    uses :class:`ClampedSwiGLU` numerics (default ``limit`` / ``alpha`` match
    cuDNN).

    Parameters
    ----------
    glu_interleave_size : int, optional
        When set, the GLU activations will use an experimental block
        interleaved format. See :class:`ClampedSwiGLU`.
    activation_recompute_in_mlp : bool, default = ``False``
        Enable fused grouped MLP kernels to recompute activation outputs
        during backward when supported instead of saving them.
    limit : float, default ``7.0``
        Clamp limit (see :class:`ClampedSwiGLU`).
    alpha : float, default ``1.702``
        Sigmoid scale (see :class:`ClampedSwiGLU`).
    glu_linear_offset : float, default ``1.0``
        Offset added to the linear component after clamping
        (see :class:`ClampedSwiGLU`).

    """

    def __init__(
        self,
        glu_interleave_size: Optional[int] = None,
        *,
        activation_recompute_in_mlp: bool = False,
        limit: float = 7.0,
        alpha: float = 1.702,
        glu_linear_offset: float = 1.0,
    ) -> None:
        super().__init__(
            glu_interleave_size,
            activation_recompute_in_mlp=activation_recompute_in_mlp,
        )
        self._clamped: ClampedSwiGLU = ClampedSwiGLU(
            limit=limit,
            alpha=alpha,
            glu_linear_offset=glu_linear_offset,
        )

    def _glu_forward(self, swiglu_in: torch.Tensor) -> torch.Tensor:
        return self._clamped._tex_clamped_swiglu_forward(swiglu_in, None)

    def _glu_backward(
        self,
        grad_swiglu_out: torch.Tensor,
        swiglu_in: torch.Tensor,
    ) -> torch.Tensor:
        return self._clamped._tex_clamped_dswiglu(
            grad_swiglu_out,
            swiglu_in,
            None,
        )
