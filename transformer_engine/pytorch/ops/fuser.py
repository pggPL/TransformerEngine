# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Manager class for a pipeline of fusible operations."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
import copy
import itertools
from typing import Any, Optional, TypeAlias

import torch

from ..quantization import FP8GlobalStateManager, Recipe, DelayedScaling
from ..quantized_tensor import prepare_for_saving, restore_from_func_ctx
from ..dynamo.quantizer_opaque import warn_compile_unsupported
from .op import (
    BasicOperation,
    FusibleOperation,
    FusedOperation,
    OperationContext,
)


def _split_tuple(t: tuple, idx: int) -> tuple[tuple, tuple]:
    """Split tuple at index"""
    return t[:idx], t[idx:]


# Lazily imported function used in _is_graph_capturing
_is_graph_capturing_function: Optional[Callable[[], bool]] = None


def _is_graph_capturing() -> bool:
    """Whether function is called within ``make_graphed_callables``

    Avoid circular import with lazy import.

    """
    global _is_graph_capturing_function
    if _is_graph_capturing_function is None:
        from ..graph import is_graph_capturing

        _is_graph_capturing_function = is_graph_capturing
    return _is_graph_capturing_function()


# Type alias for a function that may perform operation fusion
OperationFusionFunction: TypeAlias = (
    "Callable[tuple[list[FusibleOperation], ...], list[FusibleOperation]]"
)


class _OperationFuserAutogradFunction(torch.autograd.Function):
    """Autograd function for a pipeline of operations

    Autograd must be done at the pipeline level since we may apply
    different fusions in the forward and backward passes.

    """

    # pylint: disable=unused-argument
    @staticmethod
    def forward(
        func_ctx: Optional[torch.autograd.function.FunctionCtx],
        input_: torch.Tensor,
        fuser: OperationFuser,
        basic_op_kwargs: list[dict[str, Any]],
        set_output_requires_grad: bool,
        use_compiled: bool,
        *params_and_extra_inputs: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass

        Parameters
        ----------
        func_ctx: torch.autograd.function.FunctionCtx
            Context for PyTorch autograd function
        input_: torch.Tensor
            Input to first operation in pipeline
        fuser: OperationFuser
            Container for the pipeline of operations to run
        basic_op_kwargs: list of dict
            Keyword arguments to BasicOperation
        set_output_requires_grad: bool
            Whether to set ``requires_grad`` flags on returned tensors
        use_compiled: bool
            Whether to call the operations' custom ops instead of their eager
            implementations. Decided once per group by ``OperationFuser``.
        *params_and_extra_inputs: torch.Tensor
            Other tensor inputs to include in autograd graph. Consists
            of parameter tensors, followed by extra operation inputs.

        Returns
        -------
        Output tensor(s). If none of the operations have any extra
        tensor outputs, then the pipeline's output tensor is returned.
        Otherwise, a tuple with the pipeline's output tensor and extra
        tensor outputs is returned.

        """

        # Operation autograd contexts
        basic_op_ctxs = [OperationContext() for _ in range(fuser._num_basic_ops)]

        # Mark input tensors as not deletable in backward. Skipped whenever this
        # is being traced -- not merely when the custom ops are used: these
        # tensors are created outside this function, and a higher-order op may
        # not mutate anything from an enclosing scope. Under fullgraph there is
        # no falling back out of the graph, so the constraint holds either way.
        if not torch.compiler.is_compiling():
            for tensor in (input_,) + params_and_extra_inputs:
                tensor._do_not_clear = True

        # Unflatten list of parameters and extra tensor inputs
        extra_inputs = params_and_extra_inputs[-fuser.num_extra_inputs :]
        basic_op_extra_inputs = []
        for op in fuser._basic_ops:
            xs, extra_inputs = _split_tuple(extra_inputs, op.num_extra_inputs)
            basic_op_extra_inputs.append(xs)

        # Apply forward ops
        x = input_
        extra_outputs = [None] * fuser._num_basic_ops
        for op, basic_op_idxs in fuser._forward_ops:

            # Set if backward op is required
            for idx in basic_op_idxs:
                basic_op_ctxs[idx].requires_grad = idx >= fuser.first_op_requiring_backward

            # Forward op
            extra_inputs = [basic_op_extra_inputs[idx] for idx in basic_op_idxs]
            prev_op_idx = basic_op_idxs[0] - 1
            prev_op = fuser._basic_ops[prev_op_idx] if prev_op_idx >= 0 else None
            prev_op_grad_output_quantizer = None
            if prev_op is not None:
                prev_op_grad_output_quantizer = prev_op.get_grad_output_quantizer()
            next_op_idx = basic_op_idxs[-1] + 1
            next_op = fuser._basic_ops[next_op_idx] if next_op_idx < fuser._num_basic_ops else None
            next_op_input_quantizer = None
            if next_op is not None:
                next_op_input_quantizer = next_op.get_input_quantizer()

            # An inline op's eager pass is pure PyTorch, so Dynamo traces it
            # directly; only ops with registered custom ops are routed there.
            if use_compiled and not op.compile_inline:
                if op.is_fused_op:
                    x = op.compiled_fuser_forward(
                        [basic_op_ctxs[idx] for idx in basic_op_idxs],
                        x,
                        prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
                        next_op_input_quantizer=next_op_input_quantizer,
                        basic_op_kwargs=[basic_op_kwargs[idx] for idx in basic_op_idxs],
                    )
                    fused_op_extra_outputs = [() for _ in basic_op_idxs]
                else:
                    x = op.compiled_op_forward(
                        basic_op_ctxs[basic_op_idxs[0]],
                        x,
                        prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
                        next_op_input_quantizer=next_op_input_quantizer,
                        **basic_op_kwargs[basic_op_idxs[0]],
                    )
                    fused_op_extra_outputs = [()]
            else:
                x, fused_op_extra_outputs = op.fuser_forward(
                    [basic_op_ctxs[idx] for idx in basic_op_idxs],
                    x,
                    basic_op_extra_inputs=extra_inputs,
                    prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
                    next_op_input_quantizer=next_op_input_quantizer,
                    basic_op_kwargs=[basic_op_kwargs[idx] for idx in basic_op_idxs],
                )
            for idx, ys in zip(basic_op_idxs, fused_op_extra_outputs):
                for y in ys:
                    if set_output_requires_grad:
                        y.requires_grad_(idx >= fuser.first_op_requiring_backward)
                extra_outputs[idx] = ys

        # Flatten list of extra outputs
        extra_outputs_flat = []
        for idx, ys in enumerate(extra_outputs):
            ys = list(ys)
            num_extra_outputs = fuser._basic_ops[idx].num_extra_outputs
            if len(ys) != num_extra_outputs:
                raise RuntimeError(
                    f"Expected op {idx} to generate "
                    "{num_extra_outputs} extra inputs, "
                    f"but got {len(ys)}"
                )
            extra_outputs_flat.extend(ys)

        # Save context for backward pass
        if func_ctx is not None:

            # Flatten list of saved tensors
            to_save = []
            for ctx in basic_op_ctxs:
                range_start = len(to_save)
                if ctx.to_save is not None:
                    to_save.extend(ctx.to_save)
                range_end = len(to_save)
                ctx.to_save = None
                ctx._saved_tensors_range = (range_start, range_end)

            # Save tensors for backward
            tensors_to_save, tensor_objects = prepare_for_saving(*to_save)
            func_ctx.save_for_backward(*tensors_to_save)
            func_ctx.tensor_objects = tensor_objects

            # Whether to perform recipe update in backward pass. Skipped under
            # compile: this reads and flips global FP8 state, and delayed
            # scaling -- the only recipe it serves -- is gated out anyway.
            is_first_module = False
            if not torch.compiler.is_compiling() and (
                fuser.first_op_requiring_backward < fuser._num_basic_ops
            ):
                is_first_module = FP8GlobalStateManager.is_first_fp8_module()

            # Other context
            func_ctx.backward_ops = fuser._backward_ops
            func_ctx.basic_ops = fuser._basic_ops
            func_ctx.basic_op_ctxs = basic_op_ctxs
            func_ctx.basic_op_num_params = fuser._basic_op_num_params
            func_ctx.num_extra_inputs = fuser.num_extra_inputs
            func_ctx.num_extra_outputs = len(extra_outputs_flat)
            func_ctx.is_first_module = is_first_module
            func_ctx.use_compiled = use_compiled

        # Mark output tensors as not deletable in backward (eager only; see above)
        if not torch.compiler.is_compiling():
            for tensor in [x] + extra_outputs_flat:
                tensor._do_not_clear = True

        # Autograd marks the outputs of an ``apply`` itself, so this is only
        # needed on the eager path -- and AOTAutograd's functionalization drops
        # a requires_grad_() applied to a graph output anyway.
        if set_output_requires_grad and not torch.compiler.is_compiling():
            x.requires_grad_(fuser.first_op_requiring_backward < fuser._num_basic_ops)

        if extra_outputs_flat:
            return x, *extra_outputs_flat

        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        func_ctx: Any,
        grad_output: torch.Tensor,
        *grad_extra_outputs: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Backward pass"""

        # Operations and autograd state
        backward_ops = func_ctx.backward_ops
        basic_ops = func_ctx.basic_ops
        basic_op_ctxs = func_ctx.basic_op_ctxs

        # Restore saved tensors
        saved_tensors = restore_from_func_ctx(func_ctx)

        # Unflatten list of saved tensors. Under compile the contexts were
        # created in the forward, which is a different subgraph, so writing to
        # them here would be a side effect on an enclosing scope; copy them into
        # this one instead. The copy carries the attributes the forward set.
        if torch.compiler.is_compiling():
            basic_op_ctxs = [copy.copy(ctx) for ctx in basic_op_ctxs]
        for ctx in basic_op_ctxs:
            ctx.saved_tensors = saved_tensors[slice(*ctx._saved_tensors_range)]
            ctx._saved_tensors_range = None

        # Unflatten list of extra tensor output grads
        if len(grad_extra_outputs) != func_ctx.num_extra_outputs:
            raise ValueError(
                f"Expected grads for {func_ctx.num_extra_outputs} extra tensor outputs, "
                f"but got {len(grad_extra_outputs)}"
            )
        basic_op_grad_extra_outputs = []
        for op in basic_ops:
            dys, grad_extra_outputs = _split_tuple(grad_extra_outputs, op.num_extra_outputs)
            basic_op_grad_extra_outputs.append(dys)

        # Apply backward ops
        dx = grad_output
        grad_params = [None for _ in range(len(basic_ops))]
        grad_extra_inputs = [None for _ in range(len(basic_ops))]
        for op, basic_op_idxs in reversed(backward_ops):

            # Stop if no more gradients are required
            if all(not basic_op_ctxs[idx].requires_grad for idx in basic_op_idxs):
                dx = None
                break

            # Backward op
            grad_extra_outputs = [basic_op_grad_extra_outputs[idx] for idx in basic_op_idxs]
            if func_ctx.use_compiled and not op.compile_inline:
                dx, grad_params_one = op.compiled_op_backward(basic_op_ctxs[basic_op_idxs[0]], dx)
                fused_op_grad_params = [grad_params_one]
                fused_op_grad_extra_inputs = [()]
            else:
                dx, fused_op_grad_params, fused_op_grad_extra_inputs = op.fuser_backward(
                    [basic_op_ctxs[idx] for idx in basic_op_idxs],
                    dx,
                    basic_op_grad_extra_outputs=grad_extra_outputs,
                )
            for idx, dparams in zip(basic_op_idxs, fused_op_grad_params):
                grad_params[idx] = dparams
                # Dropping the reference frees the activation early; on the
                # compiled path the graph owns that lifetime instead.
                if not torch.compiler.is_compiling():
                    basic_op_ctxs[idx].saved_tensors = None
            for idx, dxs in zip(basic_op_idxs, fused_op_grad_extra_inputs):
                grad_extra_inputs[idx] = dxs

        # Flatten list of parameter gradients
        grad_params_flat = []
        for idx, dparams in enumerate(grad_params):
            num_params = func_ctx.basic_op_num_params[idx]
            if dparams is None:
                dparams = [None for _ in range(num_params)]
            else:
                dparams = list(dparams)
            if len(dparams) != num_params:
                raise RuntimeError(
                    f"Expected op {idx} to generate {num_params} param grads, "
                    f"but got {len(dparams)}"
                )
            grad_params_flat.extend(dparams)

        # Flatten list of parameter gradients
        grad_extra_inputs_flat = []
        for idx, dxs in enumerate(grad_extra_inputs):
            num_extra_inputs = basic_ops[idx].num_extra_inputs
            if dxs is None:
                dxs = [None for _ in range(num_extra_inputs)]
            else:
                dxs = list(dxs)
            if len(dxs) != num_extra_inputs:
                raise RuntimeError(
                    f"Expected op {idx} to generate grads "
                    f"for {num_extra_inputs} extra inputs, "
                    f"but got {len(dxs)}"
                )
            grad_extra_inputs_flat.extend(dxs)

        # Update FP8 scaling factors
        if func_ctx.is_first_module and not _is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        return (
            dx,  # input_
            None,  # fuser
            None,  # basic_op_kwargs
            None,  # set_output_requires_grad
            None,  # use_compiled
            *grad_params_flat,
            *grad_extra_inputs_flat,
        )


class OperationFuser:
    """Manages forward and backward passes for a pipeline of operations

    Operations are fused with three passes (see ``register_*_fusion``):

    1. Joint forward-backward fusions.
    2. Forward-only fusions.
    3. Backward-only fusions.

    Parameters
    ----------
    ops : list of FusibleOperation
        Pipeline of operations

    """

    # Functions to perform operation fusion
    forward_backward_fusion_functions: list[OperationFusionFunction] = []
    forward_fusion_functions: list[OperationFusionFunction] = []
    backward_fusion_functions: list[OperationFusionFunction] = []

    def __init__(
        self,
        ops: list[FusibleOperation],
    ) -> None:

        # Get list of basic operations
        basic_ops = []
        for op in ops:
            if op.is_fused_op:
                basic_ops.extend(op.basic_ops)
            else:
                basic_ops.append(op)
        self._num_basic_ops: int = len(basic_ops)
        self._basic_ops: list[BasicOperation] = basic_ops

        # Number of extra tensor inputs
        self._basic_op_num_extra_inputs: list[int] = list(op.num_extra_inputs for op in basic_ops)
        self.num_extra_inputs: int = sum(self._basic_op_num_extra_inputs)

        # Ops for forward and backward pass, will be populated in maybe_fuse_ops
        self._forward_ops: list[tuple[FusibleOperation, list[int]]]
        self._backward_ops: list[tuple[FusibleOperation, list[int]]]

        # Cache and detect change of state relevant for fusing operations
        self.recipe_type = None
        self.first_op_requiring_backward = 0
        self.backward_override = None
        self._last_amax_history_len = 0

        # Flatten list of parameters
        self._basic_op_params = [list(op.parameters()) for op in self._basic_ops]
        self._basic_op_num_params = list(map(len, self._basic_op_params))
        self._flat_basic_op_params = sum(self._basic_op_params, [])

    @staticmethod
    def _apply_fusions(
        ops: Iterable[FusibleOperation],
        fusion_funcs: Iterable[OperationFusionFunction],
        recipe: Optional[Recipe],
    ) -> list[FusibleOperation]:
        """Apply a sequence of fusion functions to a list of ops"""
        fused_ops = list(ops)
        for func in fusion_funcs:
            fused_ops = func(fused_ops, recipe=recipe)
        return fused_ops

    @staticmethod
    def _map_to_basic_ops(
        fused_ops: Sequence[FusibleOperation],
        basic_ops: Sequence[BasicOperation],
    ) -> list[tuple[FusibleOperation, list[int]]]:
        """Map a fused op list back to basic op indices

        Verifies that the fused ops expand to exactly ``basic_ops`` in
        order, and annotates each (possibly fused) op with the indices
        of the basic ops it covers.

        """

        def raise_mismatch_error() -> None:
            """Throw error indicating invalid op fusion"""
            raise RuntimeError(
                "Found mismatch after fusing operations "
                f"(basic_ops={[o.__class__.__name__ for o in basic_ops]}, "
                f"fused_ops={[o.__class__.__name__ for o in fused_ops]})"
            )

        # Determine basic op indices corresponding to each op
        out = []
        idx = 0
        for op in fused_ops:
            if isinstance(op, FusedOperation):
                idxs = []
                for basic_op in op.basic_ops:
                    if idx >= len(basic_ops) or basic_op is not basic_ops[idx]:
                        raise_mismatch_error()
                    idxs.append(idx)
                    idx += 1
                out.append((op, idxs))
            else:
                if idx >= len(basic_ops) or op is not basic_ops[idx]:
                    raise_mismatch_error()
                out.append((op, [idx]))
                idx += 1
        if idx != len(basic_ops):
            raise_mismatch_error()

        return out

    def maybe_fuse_ops(
        self,
        is_grad_enabled: bool,
        recipe: Optional[Recipe],
        input_: torch.Tensor,
        extra_inputs: list[Iterable[torch.Tensor]],
    ):
        """Attempt to fuse operations if neccesary"""

        # Determine which basic ops require backward
        if not is_grad_enabled:
            first_op_requiring_backward = self._num_basic_ops
        elif input_.requires_grad:
            first_op_requiring_backward = 0
        else:
            first_op_requiring_backward = self._num_basic_ops
            for op_idx in range(self._num_basic_ops):
                op_inputs = itertools.chain(self._basic_op_params[op_idx], extra_inputs[op_idx])
                if any(tensor.requires_grad for tensor in op_inputs):
                    first_op_requiring_backward = op_idx
                    break

        # Early exit if fusion parameters haven't changed
        need_reset = False
        recipe_type = type(recipe)
        backward_override = recipe.backward_override if recipe is not None else None
        fusion_params = (recipe_type, first_op_requiring_backward, backward_override)
        if fusion_params != (
            self.recipe_type,
            self.first_op_requiring_backward,
            self.backward_override,
        ):
            # Recipe type, backward override, or grad requirements have changed
            need_reset = True
        elif (
            recipe is not None
            and recipe.delayed()
            and self._last_amax_history_len != recipe.amax_history_len
        ):
            # FP8 delayed scaling has changed amax history length
            need_reset = True
        if not need_reset:
            return

        # Reset recipe state
        for op in self._basic_ops:
            op.reset_recipe_state(recipe=recipe)

        # Check if this is the first iteration
        if self.recipe_type is None:
            for op in self._basic_ops:
                op.pre_first_fuser_forward()

        # Apply joint forward-backward fusions first
        joint_ops = OperationFuser._apply_fusions(
            self._basic_ops,
            OperationFuser.forward_backward_fusion_functions,
            recipe=recipe,
        )

        # Apply forward-only and backward-only fusions
        self._forward_ops = OperationFuser._map_to_basic_ops(
            OperationFuser._apply_fusions(
                joint_ops,
                OperationFuser.forward_fusion_functions,
                recipe=recipe,
            ),
            self._basic_ops,
        )
        self._backward_ops = OperationFuser._map_to_basic_ops(
            OperationFuser._apply_fusions(
                joint_ops,
                OperationFuser.backward_fusion_functions,
                recipe=recipe,
            ),
            self._basic_ops,
        )

        # Save current fusion params
        self.recipe_type, self.first_op_requiring_backward, self.backward_override = fusion_params

        # Save amax history length
        if isinstance(recipe, DelayedScaling):
            self._last_amax_history_len = recipe.amax_history_len
        else:
            self._last_amax_history_len = 0

    def _compile_unsupported_reason(self, basic_op_kwargs: list[dict[str, Any]]) -> Optional[str]:
        """Why this group may not run through its operations' custom ops."""
        for op, _ in self._forward_ops:
            # A forward fused op compiles like a basic one, through its own
            # registered custom op; one that has not declared the compute
            # halves sends the group to eager.
            if op.is_fused_op and op.compile_ops is None:
                return f"{type(op).__name__} does not implement the compute halves"
        for op, _ in self._backward_ops:
            if op.is_fused_op:
                return "backward fused operations are not supported yet"
        for op, kwargs in zip(self._basic_ops, basic_op_kwargs, strict=True):
            # A kwarg an operation declares is resolved into its args container
            # like any other config. Anything else -- notably the preallocated
            # buffers of the grouped operations -- is written to by the op, and a
            # custom op may not mutate a tensor from an enclosing scope.
            undeclared = sorted(name for name in kwargs if name not in op.fwd_kwarg_names)
            if undeclared:
                return f"{type(op).__name__} does not support keyword arguments {undeclared}"
            # Only tensors. The other fields of an args container are values read
            # off the module, constant across calls and baked into the graph; a
            # kwarg changes per call, and on the second value Dynamo hands over a
            # symbolic scalar, which cannot go into an opaque value bundle. Pass a
            # 0-d tensor instead -- it is a graph input, so it does not recompile.
            values = sorted(name for name, v in kwargs.items() if not isinstance(v, torch.Tensor))
            if values:
                return f"{type(op).__name__} keyword arguments {values} are not tensors"
        for op in self._basic_ops:
            reason = op.compile_unsupported_reason()
            if reason is not None:
                return reason
        return None

    def _use_compiled(self, basic_op_kwargs: list[dict[str, Any]]) -> bool:
        """Whether this group runs through its operations' custom ops.

        Decided once for the whole group: a pipeline compiles as a whole, so one
        unsupported operation sends all of them to eager.

        The reason is reported from the eager path only -- ``warnings.warn`` is
        not traceable, so warning from inside the traced region would itself
        break the graph. A configuration that is never run eagerly therefore
        falls back silently.
        """
        reason = self._compile_unsupported_reason(basic_op_kwargs)
        if reason is None:
            return torch.compiler.is_compiling()
        if not torch.compiler.is_compiling():
            warn_compile_unsupported(f"running {type(self).__name__} eagerly: {reason}")
        return False

    def __call__(
        self,
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        *extra_inputs: torch.Tensor,
        basic_op_kwargs: Optional[list[dict[str, Any]]] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # Verify extra input count
        if len(extra_inputs) != self.num_extra_inputs:
            raise ValueError(
                f"Expected {self.num_extra_inputs} extra inputs but got {len(extra_inputs)}"
            )

        # Canonicalize op kwargs
        if basic_op_kwargs is None:
            basic_op_kwargs = [{}] * self._num_basic_ops

        # Unflatten list of extra tensor inputs
        extra_inputs_copy = list(extra_inputs)
        basic_op_extra_inputs = []
        for op in self._basic_ops:
            xs, extra_inputs_copy = _split_tuple(extra_inputs_copy, op.num_extra_inputs)
            basic_op_extra_inputs.append(xs)

        # Get environment state
        recipe = None
        if FP8GlobalStateManager.is_fp8_enabled():
            recipe = FP8GlobalStateManager.get_fp8_recipe()
        is_grad_enabled = torch.is_grad_enabled()

        # Attempt to fuse operations if neccesary
        self.maybe_fuse_ops(is_grad_enabled, recipe, input, basic_op_extra_inputs)

        # Initialization before forward
        for idx, op in enumerate(self._basic_ops):
            op.pre_fuser_forward(requires_grad=idx >= self.first_op_requiring_backward)

        # Fuser forward pass
        # Note: We call forward directly when is_grad_enabled=False,
        # which can expose non-leaf tensors to the inner ops. Avoid
        # problems in this case by passing set_output_requires_grad=False.
        use_compiled = self._use_compiled(basic_op_kwargs)

        args = (
            input,
            self,
            basic_op_kwargs,
            is_grad_enabled,  # set_output_requires_grad
            use_compiled,
            *self._flat_basic_op_params,
            *extra_inputs,
        )

        if not is_grad_enabled:
            return _OperationFuserAutogradFunction.forward(None, *args)

        return _OperationFuserAutogradFunction.apply(*args)


def register_forward_backward_fusion(
    op_fusion_func: OperationFusionFunction,
    prepend: bool = False,
) -> None:
    """Register a joint forward-backward operation fusion.

    A joint fusion replaces a run of basic ops with a single fused op
    that implements *both* ``fuser_forward`` and ``fuser_backward``.
    Unlike forward-only or backward-only fusions (see
    ``register_forward_fusion`` / ``register_backward_fusion``), the two
    halves need not be individually interchangeable with the unfused
    ops; only the forward/backward pair must be jointly equivalent. This
    lets the forward pass cooperate with its own backward, e.g. saving
    state that only its backward knows how to handle.

    Joint fusions are applied before the forward-only and backward-only
    fusion passes, so a joint fused op is seen by both passes. The
    forward-only and backward-only passes then fuse the remaining ops
    independently.

    The fusion function should have the following signature:

    .. code-block:: python

        func(ops, *, recipe) -> updated ops

    Parameters
    ----------
    op_fusion_func: function
        Function that takes a list of operations and may substitute
        them with fused operations.
    prepend: bool, default = ``False``
        Whether the operation fuser should apply this fusion function
        first within the joint fusion pass. The default is to apply it
        last.

    """
    if prepend:
        OperationFuser.forward_backward_fusion_functions.insert(0, op_fusion_func)
    else:
        OperationFuser.forward_backward_fusion_functions.append(op_fusion_func)


def register_forward_fusion(
    op_fusion_func: OperationFusionFunction,
    prepend: bool = False,
) -> None:
    """Register a forward-only operation fusion.

    A forward-only fusion replaces a run of basic ops with a single
    fused op that implements ``fuser_forward``. Because the backward
    pass is fused independently (see ``register_backward_fusion``), the
    fused op's forward must be interchangeable with the corresponding
    basic ops' forward: it must produce the same output and save state in
    each basic op's context that the unfused backward can consume. If the
    forward and backward need to cooperate (e.g. the forward saving
    reduced state that only a matching backward can handle), use
    ``register_forward_backward_fusion`` instead.

    The fusion function should have the following signature:

    .. code-block:: python

        func(ops, *, recipe) -> updated ops

    Parameters
    ----------
    op_fusion_func: function
        Function that takes a list of operations and may substitute
        them with fused operations.
    prepend: bool, default = ``False``
        Whether the operation fuser should apply this fusion function
        first within the forward fusion pass. The default is to apply it
        last.

    """
    if prepend:
        OperationFuser.forward_fusion_functions.insert(0, op_fusion_func)
    else:
        OperationFuser.forward_fusion_functions.append(op_fusion_func)


def register_backward_fusion(
    op_fusion_func: OperationFusionFunction,
    prepend: bool = False,
) -> None:
    """Register a backward-only operation fusion.

    A backward-only fusion replaces a run of basic ops with a single
    fused op that implements ``fuser_backward``. Because the forward
    pass is fused independently (see ``register_forward_fusion``), the
    fused op's backward must be interchangeable with the corresponding
    basic ops' backward: it must consume the state saved in each basic
    op's context by the unfused forward and produce the same gradients.
    If the forward and backward need to cooperate (e.g. the forward
    saving reduced state that only a matching backward can handle), use
    ``register_forward_backward_fusion`` instead.

    The fusion function should have the following signature:

    .. code-block:: python

        func(ops, *, recipe) -> updated ops

    Parameters
    ----------
    op_fusion_func: function
        Function that takes a list of operations and may substitute
        them with fused operations.
    prepend: bool, default = ``False``
        Whether the operation fuser should apply this fusion function
        first within the backward fusion pass. The default is to apply it
        last.

    """
    if prepend:
        OperationFuser.backward_fusion_functions.insert(0, op_fusion_func)
    else:
        OperationFuser.backward_fusion_functions.append(op_fusion_func)
