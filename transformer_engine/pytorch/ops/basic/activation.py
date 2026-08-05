# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operations for activation functions."""

from __future__ import annotations
import abc
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

import transformer_engine_torch as tex
from ...constants import DType
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...dynamo import TensorSpec, register_op_halves
from ...quantized_tensor import QuantizedTensorStorage
from ...tensor.float8_tensor import Float8CurrentScalingQuantizer, Quantizer
from ...utils import clear_tensor_data
from ..op import BasicOperation, OperationContext
from .._common import maybe_dequantize

__all__ = [
    "GELU",
    "GEGLU",
    "GLU",
    "QGELU",
    "QGEGLU",
    "ReLU",
    "ReGLU",
    "SReLU",
    "ScaledSReLU",
    "SReGLU",
    "SiLU",
]

TensorOrQuantized = Union[torch.Tensor, QuantizedTensorStorage]


@dataclass(slots=True)
class ActivationFwdArgs:
    """Flat, ``self``-free inputs to an activation forward."""

    input_: TensorOrQuantized
    dtype: torch.dtype
    output_quantizer: Optional[Quantizer]
    cache_quantized_input: bool
    requires_grad: bool
    prev_op_grad_output_quantizer: Optional[Quantizer]


@dataclass(slots=True)
class ActivationBwdArgs:
    """Flat inputs to an activation backward."""

    grad_output: Optional[torch.Tensor] = None
    saved_input: Optional[TensorOrQuantized] = None
    dtype: Optional[torch.dtype] = None
    grad_input_quantizer: Optional[Quantizer] = None


def _activation_output_shape(
    input_shape: Tuple[int, ...], halves_last_dim: bool
) -> Tuple[int, ...]:
    """Output shape of an activation: GLU variants consume pairs along the inner dim."""
    if not halves_last_dim:
        return input_shape
    return (*input_shape[:-1], input_shape[-1] // 2)


@dataclass(slots=True)
class _ActivationImpls:
    """One activation class's compute halves, plus their registered custom ops.

    Held in a container rather than as class attributes so the functions stay
    plain functions instead of being bound as methods on attribute lookup.
    """

    forward: Callable[[ActivationFwdArgs], Any]
    forward_fake: Callable[[ActivationFwdArgs], Any]
    backward: Callable[[ActivationBwdArgs], Any]
    backward_fake: Callable[[ActivationBwdArgs], Any]
    ops: Optional[Tuple[Callable[..., Any], Callable[..., Any]]]


def _make_activation_ops(
    op_name: str,
    forward_kernel: Callable[..., torch.Tensor],
    backward_kernel: Callable[..., torch.Tensor],
    halves_last_dim: bool,
) -> _ActivationImpls:
    """Build and register the compute halves for one activation class.

    The kernel pair is baked in here rather than passed as an argument: it is
    fixed by the class, and a callable has no place in an op schema.
    """

    def forward_impl(args: ActivationFwdArgs):
        x = maybe_dequantize(args.input_.contiguous(), args.dtype)
        y = forward_kernel(x, args.output_quantizer)
        if args.cache_quantized_input:
            input_quantizer = Float8CurrentScalingQuantizer(DType.kFloat8E4M3, x.device)
            input_quantizer.set_usage(rowwise=True, columnwise=False)
            x = input_quantizer(x)
        # Only the re-quantized input is handed back. Otherwise x is derived from
        # the input by dequantize + contiguous, both of which are no-ops for an
        # already-plain contiguous tensor, so x would *be* the input -- which a
        # custom op may not return. Whether those calls are no-ops depends on
        # strides, which the fake cannot see, so the rule has to be a static one:
        # the caller passes the input to the backward, which dequantizes it there.
        saved = (x,) if (args.requires_grad and args.cache_quantized_input) else ()
        ctx_attrs = {
            "dtype": args.dtype,
            "prev_op_grad_output_quantizer": args.prev_op_grad_output_quantizer,
        }
        return y, saved, ctx_attrs

    def forward_fake_impl(args: ActivationFwdArgs):
        x = args.input_
        shape = tuple(x.shape)
        y = TensorSpec(
            shape=_activation_output_shape(shape, halves_last_dim),
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
        ctx_attrs = {
            "dtype": args.dtype,
            "prev_op_grad_output_quantizer": args.prev_op_grad_output_quantizer,
        }
        return y, saved, ctx_attrs

    def backward_impl(args: ActivationBwdArgs):
        x = maybe_dequantize(args.saved_input.contiguous(), args.dtype)
        dy = maybe_dequantize(args.grad_output.contiguous(), x.dtype)
        dx = backward_kernel(dy, x, args.grad_input_quantizer)
        return (dx,)

    def backward_fake_impl(args: ActivationBwdArgs):
        shape = tuple(args.saved_input.shape)
        return (
            TensorSpec(
                shape=shape,
                dtype=args.dtype,
                quantizer=args.grad_input_quantizer,
                device=args.saved_input.device,
            ),
        )

    return _ActivationImpls(
        forward=forward_impl,
        forward_fake=forward_fake_impl,
        backward=backward_impl,
        backward_fake=backward_fake_impl,
        ops=register_op_halves(
            op_name=op_name,
            fwd_arg_type=ActivationFwdArgs,
            fwd_impl=forward_impl,
            fwd_fake_impl=forward_fake_impl,
            bwd_arg_type=ActivationBwdArgs,
            bwd_impl=backward_impl,
            bwd_fake_impl=backward_fake_impl,
            num_grad_inputs=1,
        ),
    )


class _ActivationOperation(BasicOperation, metaclass=abc.ABCMeta):
    r"""Apply activation function

    Activation functions are either element-wise unary functions or
    variants of the gated linear unit (GLU). Recall that GLU is
    computed by splitting the input tensor into chunks :math:`a` and
    :math:`b` along the last dimension and computing

    .. math::
       \text{GLU}(a,b) = \sigma(a) * b

    .. warning::

       Transformer Engine gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:`a` and
       :math:`b`. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    Parameters
    ----------
    cache_quantized_input : bool, default = False
        Quantize input tensor when caching for use in the backward
        pass. This will typically reduce memory usage but require
        extra compute and increase numerical error. This feature is
        highly experimental.

    """

    def __init__(self, *, cache_quantized_input: bool = False):
        super().__init__()
        self.cache_quantized_input: bool = cache_quantized_input

    @staticmethod
    @abc.abstractmethod
    def _activation_forward_impl(*args, **kwargs) -> torch.Tensor:
        """Forward implementation

        Implementation from transformer_engine.pytorch.cpp_extensions.

        """

    @staticmethod
    @abc.abstractmethod
    def _activation_backward_impl(*args, **kwargs) -> torch.Tensor:
        """Backward implementation

        Implementation from transformer_engine_torch.

        """

    # GLU variants consume pairs along the inner dimension; set per subclass.
    _output_halves_last_dim: bool = False
    _impls: Optional[_ActivationImpls] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # The kernel pair is fixed by the class, so each subclass gets its own
        # registration; the op registry stays bounded by the op zoo, not by the
        # model.
        if getattr(cls._activation_forward_impl, "__isabstractmethod__", False):
            return
        cls._impls = _make_activation_ops(
            op_name=f"activation_{cls.__name__.lower()}",
            forward_kernel=cls._activation_forward_impl,
            backward_kernel=cls._activation_backward_impl,
            halves_last_dim=cls._output_halves_last_dim,
        )

    def resolve_fwd_args(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: bool,
        prev_op_grad_output_quantizer: Optional[Quantizer] = None,
        next_op_input_quantizer: Optional[Quantizer] = None,
    ) -> ActivationFwdArgs:
        """Gather everything the forward needs into a flat, self-free container.

        Reads the autocast state, so it must run in the traced region (where
        Dynamo guards that read), never inside the custom op.
        """
        dtype: torch.dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = input_.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(f"Unsupported dtype ({dtype})")
        return ActivationFwdArgs(
            input_=input_,
            dtype=dtype,
            output_quantizer=next_op_input_quantizer,
            cache_quantized_input=self.cache_quantized_input,
            requires_grad=requires_grad,
            prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
        )

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        args = self.resolve_fwd_args(
            input_,
            requires_grad=ctx.requires_grad,
            prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
            next_op_input_quantizer=next_op_input_quantizer,
        )
        y, saved, ctx_attrs = self._impls.forward(args)
        if ctx.requires_grad:
            # Without ``cache_quantized_input`` the op keeps nothing: the backward
            # rebuilds its input from the operation's input tensor.
            saved = saved or (input_,)
            if is_cpu_offload_enabled():
                mark_activation_offload(*saved)
            ctx.save_for_backward(*saved)
            for name, value in ctx_attrs.items():
                setattr(ctx, name, value)
        return y

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        (x,) = ctx.saved_tensors
        (dx,) = self._impls.backward(
            ActivationBwdArgs(
                grad_output=grad_output,
                saved_input=x,
                dtype=ctx.dtype,
                grad_input_quantizer=ctx.prev_op_grad_output_quantizer,
            )
        )
        # Eager only: the compiled path leaves saved tensors to the graph's own
        # memory planning, and clearing an op input would lie to functionalization.
        clear_tensor_data(x)
        return dx, ()


class GELU(_ActivationOperation):
    r"""Gaussian Error Linear Unit

    This computes the "tanh" approximation to GELU:

    .. math::

       \text{GELU}(x) \approx \frac{x}{2} \left( 1 + \tanh\left( 0.797x+0.036 x^3 \right) \right)

    See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`__.

    """

    @staticmethod
    def _activation_forward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.gelu(*args, **kwargs)

    @staticmethod
    def _activation_backward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.dgelu(*args, **kwargs)


class GLU(_ActivationOperation):
    r"""Gated Linear Unit

    The input tensor is split into chunks :math:`a` and :math:`b`
    along the last dimension and the following is computed:

    .. math::

       \text{GLU}(a,b) = \sigma(a) * b

    where :math:`\sigma` is the sigmoid function.

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:`a` and
       :math:`b`. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    See `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`__
    and `GLU Variants Improve Transformer <https://arxiv.org/abs/2002.05202>`__.

    """

    _output_halves_last_dim: bool = True

    @staticmethod
    def _activation_forward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.glu(*args, **kwargs)

    @staticmethod
    def _activation_backward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.dglu(*args, **kwargs)


class GEGLU(_ActivationOperation):
    r"""Gaussian Error Gated Linear Unit

    The input tensor is split into chunks :math:`a` and :math:`b`
    along the last dimension and the following is computed:

    .. math::

       \text{GEGLU}(a,b) = \text{GELU}(a) * b

    where

    .. math::

       \text{GELU}(x) \approx \frac{x}{2} \left( 1 + \tanh\left( 0.797x+0.036 x^3 \right) \right)

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:`a` and
       :math:`b`. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    See `GLU Variants Improve Transformer <https://arxiv.org/abs/2002.05202>`__.

    """

    _output_halves_last_dim: bool = True

    @staticmethod
    def _activation_forward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.geglu(*args, **kwargs)

    @staticmethod
    def _activation_backward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.dgeglu(*args, **kwargs)


class QGELU(_ActivationOperation):
    r"""Quick Gaussian Error Linear Unit

    Quick GELU from `HuggingFace <https://github.com/huggingface/transformers/blob/3e93dd295b5343557a83bc07b0b2ea64c926f9b4/src/transformers/activations.py#L90>`__
    and `paper <https://github.com/hendrycks/GELUs>`__.

    .. math::

       \text{QGELU}(x) \approx x * \sigma(1.702 * x)

    """

    @staticmethod
    def _activation_forward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.qgelu(*args, **kwargs)

    @staticmethod
    def _activation_backward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.dqgelu(*args, **kwargs)


class QGEGLU(_ActivationOperation):
    r"""Quick Gaussian Error Gated Linear Unit

    The input tensor is split into chunks :math:`a` and :math:`b`
    along the last dimension and the following is computed:

    .. math::

       \text{QGEGLU}(a,b) = \text{QGELU}(a) * b

    where

    .. math::

       \text{QGELU}(x) \approx x * \sigma(1.702 * x)

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:`a` and
       :math:`b`. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    """

    _output_halves_last_dim: bool = True

    @staticmethod
    def _activation_forward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.qgeglu(*args, **kwargs)

    @staticmethod
    def _activation_backward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.dqgeglu(*args, **kwargs)


class ReLU(_ActivationOperation):
    r"""Rectified Linear Unit

    .. math::

       \text{ReLU}(x) = \max(x,0)

    """

    @staticmethod
    def _activation_forward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.relu(*args, **kwargs)

    @staticmethod
    def _activation_backward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.drelu(*args, **kwargs)


class ReGLU(_ActivationOperation):
    r"""Rectified Gated Linear Unit

    The input tensor is split into chunks :math:`a` and :math:`b`
    along the last dimension and the following is computed:

    .. math::

       \text{ReGLU}(a,b) = \max(a,0) * b

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:`a` and
       :math:`b`. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    See `GLU Variants Improve Transformer <https://arxiv.org/abs/2002.05202>`__.

    """

    _output_halves_last_dim: bool = True

    @staticmethod
    def _activation_forward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.reglu(*args, **kwargs)

    @staticmethod
    def _activation_backward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.dreglu(*args, **kwargs)


class SReLU(_ActivationOperation):
    r"""Squared Rectified Linear Unit

    .. math::

       \text{SReLU}(x) = \max(x^2,0)

    See `Primer: Searching for Efficient Transformers for Language Modeling <https://arxiv.org/abs/2109.08668v2>`__.

    """

    @staticmethod
    def _activation_forward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.srelu(*args, **kwargs)

    @staticmethod
    def _activation_backward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.dsrelu(*args, **kwargs)


class ScaledSReLU(BasicOperation):
    r"""Squared ReLU with per-row post-scaling.

    If the SReLU output has shape ``(d_1, ..., d_n)``, it is multiplied
    with an extra input tensor of shape ``(d_1, ..., d_{n-1})``.

    Parameters
    ----------
    activation_recompute_in_mlp : bool, default = ``False``
        Enable fused grouped MLP kernels to recompute activation outputs
        during backward when supported instead of saving them.
    """

    num_extra_inputs: int = 1

    def __init__(self, *, activation_recompute_in_mlp: bool = False) -> None:
        super().__init__()
        self.activation_recompute_in_mlp: bool = activation_recompute_in_mlp

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
        prev_op_grad_output_quantizer: Optional[Quantizer],  # pylint: disable=unused-argument
        next_op_input_quantizer: Optional[Quantizer],  # pylint: disable=unused-argument
        basic_op_kwargs: list[dict[str, Any]],  # pylint: disable=unused-argument
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:
        if self.activation_recompute_in_mlp:
            raise RuntimeError(
                f"{self.__class__.__name__}(activation_recompute_in_mlp=True) requires the "
                "fused grouped MLP path."
            )

        extra_input = basic_op_extra_inputs[0][0]

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        elif isinstance(input_, torch.Tensor):
            dtype = input_.dtype
        else:
            dtype = extra_input.dtype

        x = maybe_dequantize(input_.contiguous(), dtype)
        scales = maybe_dequantize(extra_input, dtype)
        y = tex.srelu(x, None) * scales.unsqueeze(-1)

        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(x)
            ctx.input_requires_grad = True
            ctx.extra_input_requires_grad = extra_input.requires_grad
            ctx.dtype = dtype
            ctx.save_for_backward(x, scales)

        return y, [()]

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
        del basic_op_grad_extra_outputs

        if self.activation_recompute_in_mlp:
            raise RuntimeError(
                f"{self.__class__.__name__}(activation_recompute_in_mlp=True) requires the "
                "fused grouped MLP path."
            )

        ctx = basic_op_ctxs[0]
        x, scales = ctx.saved_tensors
        x = maybe_dequantize(x.contiguous(), ctx.dtype)
        scales = maybe_dequantize(scales, ctx.dtype)
        grad_output = maybe_dequantize(grad_output.contiguous(), ctx.dtype)

        grad_input = None
        if ctx.input_requires_grad:
            grad_srelu_out = grad_output * scales.unsqueeze(-1)
            grad_input = tex.dsrelu(grad_srelu_out, x, None)

        grad_extra_input = None
        if ctx.extra_input_requires_grad:
            srelu_out = tex.srelu(x, None)
            grad_extra_input = torch.linalg.vecdot(srelu_out, grad_output)

        clear_tensor_data(ctx.saved_tensors[0])

        return grad_input, [()], [(grad_extra_input,)]


class SReGLU(_ActivationOperation):
    r"""Squared Rectified Gated Linear Unit

    The input tensor is split into chunks :math:`a` and :math:`b`
    along the last dimension and the following is computed:

    .. math::

       \text{SReGLU}(a,b) = \max(a^2,0) * b

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:`a` and
       :math:`b`. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    """

    _output_halves_last_dim: bool = True

    @staticmethod
    def _activation_forward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.sreglu(*args, **kwargs)

    @staticmethod
    def _activation_backward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.dsreglu(*args, **kwargs)


class SiLU(_ActivationOperation):
    r"""Sigmoid Linear Unit

    .. math::

       \text{SiLU}(x) = x \sigma(x) = \frac{x}{1+\exp(-x)}

    """

    @staticmethod
    def _activation_forward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.silu(*args, **kwargs)

    @staticmethod
    def _activation_backward_impl(*args, **kwargs) -> torch.Tensor:
        return tex.dsilu(*args, **kwargs)
