# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNormLinear API"""
import os
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch.nn import init

from .. import cpp_extensions as pytex
import transformer_engine_torch as tex

from .base import (
    get_workspace,
    get_ub,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ..fp8 import get_fp8_te_dtype, FP8GlobalStateManager
from ..utils import (
    divide,
    get_default_init_method,
    init_method_constant,
    cast_if_needed,
    assert_dim_for_fp8_exec,
    clear_tensor_data,
    requires_grad,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
    _fsdp_scatter_tensors,
    _fsdp_gather_tensors,
)
from ..constants import GemmParallelModes, dist_group_type, TE_DType
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ._common import _apply_normalization, _noop_cat
from ..tensor import Float8Tensor, Float8Quantizer
from ..tensor.quantized_tensor import (
    QuantizedTensor,
    Quantizer,
    prepare_for_saving,
    restore_from_saved
)
from ..cpu_offload import is_cpu_offload_enabled

from ..cpp_extensions import (
    general_gemm,
)

__all__ = ["LayerNormLinear"]


class _LayerNormLinear(torch.autograd.Function):
    """LayerNormLinear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Union[torch.Tensor, None],
        weight: torch.Tensor,
        bias: torch.Tensor,
        use_bias: bool,
        eps: float,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        fuse_wgrad_accumulation: bool,
        input_quantizer: Optional[Quantizer],
        weight_quantizer: Optional[Quantizer],
        output_quantizer: Optional[Quantizer],
        grad_output_quantizer: Optional[Quantizer],
        grad_input_quantizer: Optional[Quantizer],
        cpu_offloading: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        return_layernorm_output: bool,
        return_layernorm_output_gathered: bool,
        is_grad_enabled: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        normalization: str,
        ub_bulk_wgrad: bool,
        ub_bulk_dgrad: bool,
        ub_overlap_rs_dgrad: bool,
        ub_overlap_ag: bool,
        ub_name: str,
        fp8_output: bool,
        fsdp_group: Union[dist_group_type, None],
        module: torch.nn.Module,
        skip_fp8_weight_update: bool,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # pylint: disable=missing-function-docstring
        # Make sure input dimensions are compatible
        out_features, in_features = weight.shape
        inp_shape = inp.shape
        assert inp_shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))
        if fp8:
            assert_dim_for_fp8_exec(inputmat)
            assert_dim_for_fp8_exec(weight)

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)

        if ub_overlap_ag:
            raise NotImplementedError
            tp_world_size = get_distributed_world_size(tp_group)
            if tp_world_size == 1 or (not is_grad_enabled):
                ub_overlap_ag = False
        if ub_overlap_ag:
            raise NotImplementedError
            dim_size = list(inputmat.size())
            dim_size[0] = dim_size[0] * tp_world_size
            ub_obj_lnout = get_ub(ub_name + "_fprop")
            if return_layernorm_output:
                # First prepare LN output in higher precision,
                # which will be later copied to a FP8 UB
                ln_out = torch.empty_like(inputmat, memory_format=torch.contiguous_format)
            else:
                ln_out = ub_obj_lnout.get_ubuf_output(0)

        weight_requires_grad = weight.requires_grad
        backward_needs_input = is_grad_enabled and weight_requires_grad
        with_input_all_gather = parallel_mode == "column" and sequence_parallel

        # Configure quantizer for normalization output
        if fp8 and input_quantizer is None:
            raise ValueError("Missing quantizer for input tensor")
        with_quantized_norm = fp8 and not return_layernorm_output
        if with_quantized_norm:
            if with_input_all_gather:
                if isinstance(input_quantizer, Float8Quantizer):
                    input_quantizer.set_usage(rowwise=True, columnwise=False)
                else:
                    with_quantized_norm = False
            else:
                input_quantizer.set_usage(
                    rowwise=True,
                    columnwise=(is_grad_enabled and weight_requires_grad),
                )

        # Apply normalization
        ln_out, mu, rsigma = _apply_normalization(
            inputmat,
            None,
            ln_weight,
            ln_bias,
            eps,
            input_quantizer if with_quantized_norm else None,
            inp.dtype,
            normalization,
            fwd_ln_sm_margin,
            zero_centered_gamma,
            is_grad_enabled,
        )
        ln_out_return = ln_out if return_layernorm_output else None

        # Prepare GEMM input
        # Note: Cast to expected dtype and perform tensor-parallel communication
        if with_input_all_gather:
            with_quantized_all_gather = fp8
            if return_layernorm_output and return_layernorm_output_gathered:
                with_quantized_all_gather = False
            if fp8:
                input_quantizer.set_usage(rowwise=True, columnwise=False)
            ln_out_total, _ = gather_along_first_dim(
                ln_out,
                tp_group,
                quantizer=(input_quantizer if with_quantized_all_gather else None),
            )
            if return_layernorm_output and return_layernorm_output_gathered:
                ln_out_return = ln_out_total
            if fp8 and not with_quantized_all_gather:
                ln_out_total = input_quantizer(ln_out_total)
        else:
            if fp8 and not with_quantized_norm:
                input_quantizer.set_usage(
                    rowwise=True,
                    columnwise=(is_grad_enabled and weight_requires_grad),
                )
                ln_out = input_quantizer(ln_out)
            ln_out_total = ln_out

        # Cast weight to expected dtype
        weightmat = weight
        if not fp8:
            weightmat = cast_if_needed(weightmat, activation_dtype)
        else:
            if not isinstance(weight, QuantizedTensor):

                # Configure quantizer
                if weight_quantizer is not None:
                    weight_quantizer.set_usage(rowwise=True, columnwise=True)

                # FP8 cast to workspace buffer
                update_workspace = is_first_microbatch is None or is_first_microbatch
                weightmat = module.get_weight_workspace(
                    tensor=weight,
                    quantizer=weight_quantizer,
                    cache_name=(None if is_first_microbatch is None else "weight"),
                    update_workspace=update_workspace,
                    skip_update_flag=skip_fp8_weight_update,
                    fsdp_group=fsdp_group,
                    is_grad_enabled=is_grad_enabled,
                )

        # Cast bias to expected dtype
        bias_dtype = activation_dtype
        if fp8 and activation_dtype == torch.float32:
            bias_dtype = torch.bfloat16
        bias = cast_if_needed(bias, bias_dtype) if bias is not None else bias

        # Configure output quantizer
        if output_quantizer is not None:
            output_quantizer.set_usage(rowwise=True, columnwise=False)

        # Calibrate quantizers if needed
        if not fp8 and fp8_calibration:
            if input_quantizer is not None:
                input_quantizer.calibrate(ln_out_total)
            if weight_quantizer is not None:
                weight_quantizer.calibrate(weight)

        out, _, _ = general_gemm(
            weightmat,
            ln_out_total,
            get_workspace(),
            quantization_params=output_quantizer,
            out_dtype=activation_dtype,
            bias=bias,
            use_split_accumulator=_2X_ACC_FPROP,
            ub_algo=tex.CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P if ub_overlap_ag else None,
            ub=ub_obj_lnout if ub_overlap_ag else None
        )

        if is_grad_enabled:
            if cpu_offloading:
                if fp8 and weightmat is not None:
                    weightmat.weight_offloading = True
                ln_weight.weight_offloading = True
                weight.weight_offloading = True

                inputmat.activation_offloading = True
                if normalization == "LayerNorm":
                    mu.activation_offloading = True
                rsigma.activation_offloading = True
                ln_out.activation_offloading = True

            # Scatter intermediate/activation tensors saved for the backward pass
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                mu,
                rsigma,
                weightmat if fp8 and not isinstance(weight, Float8Tensor) else None,
                ln_out if weight.requires_grad else None,
            )

            saved_input_tensors, saved_input = prepare_for_saving(inputmat)

            saved_weight_tensors, saved_weight = prepare_for_saving(weightmat)
            saved_bias_tensors, saved_bias = prepare_for_saving(bias)
            saved_ln_weight_tensors, saved_ln_weight = prepare_for_saving(ln_weight)
            saved_ln_bias_tensors, saved_ln_bias = prepare_for_saving(ln_bias)
            saved_ln_out_tensors, saved_ln_out = [None], None
            if weight.requires_grad:
                saved_ln_out_tensors, saved_ln_out = prepare_for_saving(ln_out)

            ctx.save_for_backward(
                *saved_input_tensors,
                *saved_weight_tensors,
                *saved_bias_tensors,
                *saved_ln_weight_tensors,
                *saved_ln_bias_tensors,
                *saved_ln_out_tensors,
                mu,
                rsigma,
                weight.main_grad if fuse_wgrad_accumulation and weight.requires_grad else None
            )

            ctx.saved_input = saved_input
            ctx.saved_weight = saved_weight
            ctx.saved_bias = saved_bias
            ctx.saved_ln_weight = saved_ln_weight
            ctx.saved_ln_bias = saved_ln_bias
            ctx.saved_ln_out = saved_ln_out

            ctx.requires_dgrad = inp.requires_grad
            ctx.requires_wgrad = weight.requires_grad

            ctx.grad_input_quantizer = grad_input_quantizer
            ctx.grad_output_quantizer = grad_output_quantizer
            ctx.input_quantizer = input_quantizer

            ctx.owns_input = inputmat is not inp

            ctx.weight = weight
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp_shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.return_layernorm_output = return_layernorm_output
            ctx.return_layernorm_output_gathered = return_layernorm_output_gathered
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
            ctx.ub_name = ub_name
            ctx.requires_dgrad = inp.requires_grad
            ctx.normalization = normalization
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if ctx.fp8 and requires_grad(inp, ln_weight, ln_bias, weight, bias):
                _first_fp8_module = FP8GlobalStateManager.IS_FIRST_FP8_MODULE
                ctx.reduce_and_update_bwd_fp8_tensors = FP8GlobalStateManager.is_first_fp8_module()
                if in_fp8_activation_recompute_phase():
                    FP8GlobalStateManager.IS_FIRST_FP8_MODULE = _first_fp8_module

        # Row Parallel Linear
        if parallel_mode == "row" and sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        out = out.view(-1, *inp_shape[1:-1], out_features)

        if return_layernorm_output:
            if return_layernorm_output_gathered:
                shape = list(inp.shape)
                shape[0] *= tp_size
                return out, ln_out_return.view(shape)
            return out, ln_out_return.view_as(inputmat)
        return out

    @staticmethod
    def backward(
        ctx, *grad_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring

        with torch.cuda.nvtx.range("_LayerNormLinear_backward"):
            saved_tensors = ctx.saved_tensors
            inputmat, saved_tensors = restore_from_saved(ctx.saved_input, saved_tensors)
            weight, saved_tensors = restore_from_saved(ctx.saved_weight, saved_tensors)
            bias, saved_tensors = restore_from_saved(ctx.saved_bias, saved_tensors)
            ln_weight, saved_tensors = restore_from_saved(ctx.saved_ln_weight, saved_tensors)
            ln_bias, saved_tensors = restore_from_saved(ctx.saved_ln_bias, saved_tensors)
            ln_out, saved_tensors = restore_from_saved(ctx.saved_ln_out, saved_tensors)
            mu, rsigma, main_grad = saved_tensors

            if ctx.grad_output_quantizer is not None:
                ctx.grad_output_quantizer.set_usage(
                    rowwise=True, # TODO: look at this, it may be optimied, but casting to transpose only is not implemented
                    columnwise=True, # TODO should not be implemented ctx.requires_wgrad 
                )
            if ctx.grad_input_quantizer is not None:
                ctx.grad_input_quantizer.set_usage(rowwise=True, columnwise=False)

            # Gather intermediate/activation tensors if needed
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves

            # TODO
            #_fsdp_gather_tensors(
            #    ctx.fsdp_group,
            #    ctx.fsdp_shapes,
            #    mu,
            #    rsigma,
            #    weight_fp8 if ctx.fp8 and not isinstance(weight, Float8Tensor) else None,
            #    ln_out,
            #)

            if ctx.cpu_offloading and ctx.fuse_wgrad_accumulation:
                raise NotImplementedError
                weight = torch.nn.Parameter(weight, weight.requires_grad)
                weight.main_grad = main_grad
             

            if ctx.ub_overlap_rs_dgrad:
                raise NotImplementedError
                ctx.ub_bulk_dgrad = False
                ctx.ub_bulk_wgrad = False
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1:
                    ctx.ub_overlap_rs_dgrad = False
            if ctx.ub_bulk_dgrad:
                raise NotImplementedError
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1 or not weight.requires_grad:
                    ctx.ub_bulk_dgrad = False
            if ctx.ub_bulk_dgrad:
                raise NotImplementedError
                dim_size = list(ln_out.size())
                dim_size[0] = dim_size[0] * tp_world_size
                ub_obj_lnout = get_ub(ctx.ub_name + "_dgrad")
                ub_obj_lnout.copy_input_to_ubuf(ln_out, 1)


            if ctx.ub_bulk_wgrad:
                raise NotImplementedError
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1 or not weight.requires_grad:
                    ctx.ub_bulk_wgrad = False

            (
                grad_output,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx,
                grad_outputs[0],
                ctx.parallel_mode == "row",
                ctx.grad_output_quantizer,
            )

            # Prepare GEMM input
            # Note: Perform tensor-parallel communication if needed
            ln_out_total = None
            ln_out_total_work = None
            if ctx.requires_wgrad and ctx.parallel_mode == "column" and ctx.sequence_parallel:
                quantizer = None
                if ctx.fp8:
                    quantizer = ctx.input_quantizer
                    quantizer.set_usage(rowwise=True, columnwise=True)
                ln_out_total, ln_out_total_async = gather_along_first_dim(
                    ln_out,
                    ctx.tp_group,
                    async_op=True,
                    quantizer=quantizer,
                )
            else:
                ln_out_total = ln_out

            # Check whether to output wgrad GEMM directly into main grad
            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            # dgrad GEMM
            if ctx.grad_input_quantizer is not None:
                ctx.grad_input_quantizer.set_usage(rowwise=True, columnwise=False)
            dgrad, _, _ = general_gemm(
                weight,
                grad_output,
                get_workspace(),
                layout="NN",
                grad=True,
                quantization_params=ctx.grad_input_quantizer,
                out_dtype=ctx.activation_dtype,
                use_split_accumulator=_2X_ACC_DGRAD,
            )

            # Launch tensor-parallel communication
            dgrad_work = None
            if ctx.parallel_mode == "column":
                if ctx.sequence_parallel:
                    if ctx.return_layernorm_output and ctx.return_layernorm_output_gathered:
                        dgrad = dgrad + grad_outputs[1].view_as(dgrad)
                    dgrad, dgrad_work = reduce_scatter_along_first_dim(
                        dgrad,
                        ctx.tp_group,
                        async_op=True,
                    )
                else:
                    dgrad, dgrad_work = allreduce(dgrad, ctx.tp_group, async_op=True)

            # Compute grad weight tensor
            wgrad = None
            if ctx.requires_wgrad:

                # Synchronize tensor-parallel communication
                if ln_out_total_work is not None:
                    ln_out_total_work.wait()
                    ln_out_total_work = None

                if hasattr(ln_out_total, "_create_transpose"):
                    ln_out_total._create_transpose() # TODO(pgadzinski) - temporary

                # wgrad GEMM
                # Note: Fuse with bgrad computation if needed
                wgrad, grad_bias_, _ = general_gemm(
                    ln_out_total,
                    grad_output,
                    get_workspace(),
                    layout="NT",
                    grad=True,
                    bias=(bias if grad_bias is None else None),
                    out_dtype=ctx.activation_dtype,  # TODO: not sure about that
                    out=main_grad if ctx.fuse_wgrad_accumulation else None,
                    use_split_accumulator=_2X_ACC_WGRAD,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                )
                if grad_bias is None:
                    grad_bias = grad_bias_
                del grad_bias_

                # Deallocate input tensor
                if ctx.owns_input:
                    clear_tensor_data(ln_out_total)

            # Don't return grad bias if not needed
            if not ctx.use_bias:
                grad_bias = None

            # Synchronize tensor parallel communication
            if ln_out_total_work is not None:
                ln_out_total_work.wait()
                ln_out_total_work = None
            if dgrad_work is not None:
                dgrad_work.wait()
                dgrad_work = None

            # Residual gradient
            dgrad = dgrad.view(inputmat.shape)
            if ctx.return_layernorm_output and not ctx.return_layernorm_output_gathered:
                dgrad = dgrad + grad_outputs[1].view_as(dgrad)

            # Norm gradient
            dgamma = None
            dbeta = None
            if ctx.normalization == "LayerNorm":
                dgrad, dgamma, dbeta = tex.layernorm_bwd(
                    dgrad,
                    inputmat,
                    mu,
                    rsigma,
                    ln_weight,
                    ctx.bwd_ln_sm_margin,
                    ctx.zero_centered_gamma,
                )
                dgrad = dgrad.reshape(inputmat.size())
            elif ctx.normalization == "RMSNorm":
                dgrad, dgamma = tex.rmsnorm_bwd(
                    dgrad,
                    inputmat,
                    rsigma,
                    ln_weight,
                    ctx.bwd_ln_sm_margin,
                    ctx.zero_centered_gamma,
                )
                dgrad = dgrad.reshape(inputmat.size())
                dbeta = None
            clear_tensor_data(mu)
            clear_tensor_data(rsigma)

        if ctx.requires_wgrad:
            # Handle custom DDP from mcore.
            if ctx.fuse_wgrad_accumulation and hasattr(weight, "grad_added_to_main_grad"):
                weight.grad_added_to_main_grad = True
                if getattr(weight, "zero_out_wgrad", False):
                    wgrad = torch.zeros(
                        weight.main_grad.shape,
                        dtype=weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    wgrad = torch.empty(
                        weight.main_grad.shape,
                        dtype=weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
            elif ctx.fuse_wgrad_accumulation:
                wgrad = None
        else:
            wgrad = None

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        # Scatter fp8 weight buffers
        #if ctx.fp8 and not isinstance(weight, Float8Tensor):
        #    _fsdp_scatter_tensors(ctx.fsdp_group, weight_fp8)

        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            dgamma,
            dbeta,
            wgrad,
            grad_bias,
            None,  # use_bias
            None,  # use_bias
            None,  # use_bias
            None,  # use_bias
            None,  # use_bias
            None,  # use_bias
            None,  # use_bias
            None,  # eps
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # fp8_meta
            None,  # fuse_wgrad_accumulation
            None,  # cpu_offloading
            None,  # tp_group
            None,  # tp_size
            None,  # sequence_parallel
            None,  # tensor_parallel
            None,  # activation_dtype
            None,  # parallel_mode
            None,  # return_layernorm_output
            None,  # return_layernorm_output_gathered
            None,  # is_grad_enabled
            None,  # fwd_ln_sm_margin
            None,  # bwd_ln_sm_margin
            None,  # zero_centered_gamma
            None,  # normalization
            None,  # ub_bulk_wgrad
            None,  # ub_bulk_dgrad
            None,  # ub_overlap_rs_dgrad
            None,  # ub_overlap_ag
            None,  # ub_name
            None,  # fp8_output
            None,  # fsdp_group
        )


class LayerNormLinear(TransformerEngineBaseModule):
    r"""
    Applies layer normalization followed by linear transformation to the incoming data.

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    eps : float, default = 1e-5
         a value added to the denominator of layer normalization for numerical stability.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module is
                             taken post layernorm.
    return_layernorm_output_gathered : bool, default = `False`
                             if set to `True`, output of layernorm is returned after the all
                             gather operation. Ignored if return_layernorm_output is False.
                             Example use case: with sequence parallel, input to residual connection
                             for transformer module (e.g. LoRA) will need to be gathered.
                             Returning layernorm output gathered will prevent a redundant gather.
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      Configuration for splitting the weight and bias tensors along dim 0 into
                      multiple PyTorch parameters. If a list or tuple of strings is provided,
                      they are used to make the names of equally-sized parameters. If a dict
                      (preferably an OrderedDict) is provided, the keys are used as names and
                      values as split sizes along dim 0. The resulting parameters will have
                      names that end in `_weight` or `_bias`, so trailing underscores are
                      stripped from any provided names.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.
    parallel_mode : {None, 'column', 'row'}, default = `None`
                   used to decide whether this Linear layer is Column Parallel Linear or Row
                   Parallel Linear as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
                   When set to `None`, no communication is performed.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        normalization: str = "LayerNorm",
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        return_layernorm_output: bool = False,
        return_layernorm_output_gathered: bool = False,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "cuda",
        ub_bulk_wgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_overlap_ag: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.normalization = normalization
        assert normalization in ["LayerNorm", "RMSNorm"], "Unsupported normalization type!"
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = self.use_bias and not return_bias
        self.return_layernorm_output = return_layernorm_output
        self.return_layernorm_output_gathered = return_layernorm_output_gathered
        self.zero_centered_gamma = zero_centered_gamma
        self.ub_bulk_wgrad = ub_bulk_wgrad
        self.ub_bulk_dgrad = ub_bulk_dgrad
        self.ub_overlap_ag = ub_overlap_ag
        self.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
        if any([ub_bulk_wgrad, ub_bulk_dgrad, ub_overlap_ag, ub_overlap_rs_dgrad]):
            assert ub_name is not None, "Userbuffer name [string] is not set."
        self.ub_name = ub_name

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        self.parallel_mode = parallel_mode
        assert (
            self.parallel_mode in GemmParallelModes
        ), f"parallel_mode {parallel_mode} not supported"

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        if init_method is None:
            init_method = get_default_init_method()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        self.eps = eps
        layer_norm_weight = torch.nn.Parameter(
            torch.empty(in_features, device=device, dtype=params_dtype)
        )
        self.register_parameter(
            "layer_norm_weight",
            layer_norm_weight,
            init_fn=init_method_constant(float(not self.zero_centered_gamma)),
        )
        if self.normalization != "RMSNorm":
            layer_norm_bias = torch.nn.Parameter(
                torch.empty(in_features, device=device, dtype=params_dtype)
            )
            self.register_parameter(
                "layer_norm_bias", layer_norm_bias, init_fn=init_method_constant(0.0)
            )
        else:
            self.layer_norm_bias = None

        # Initialize params in FP8
        with_fp8_params = FP8GlobalStateManager.with_fp8_parameters()

        # Contiguous buffers for params
        weight_tensor = torch.empty(
            self.out_features,
            self.in_features,
            device=device,
            dtype=params_dtype,
        )
        bias_tensor = None
        if self.use_bias:
            bias_tensor = torch.empty(
                self.out_features,
                device=device,
                dtype=params_dtype,
            )

        # Configure parameter splits
        self.weight_names = []
        self.bias_names = []
        self.parameter_split_sizes = []
        if parameters_split is None:
            # Split into a single parameter by default
            self.weight_names = ["weight"]
            self.bias_names = ["bias"]
            self.parameter_split_sizes = [out_features]
        elif not parameters_split:
            raise ValueError("Cannot split weight buffer into 0 parameters")
        elif isinstance(parameters_split, dict):
            # Split parameters with provided sizes
            for name, split_size in parameters_split.items():
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        elif all(isinstance(name, str) for name in parameters_split):
            # Split parameters evenly
            split_size = out_features // len(parameters_split)
            for name in parameters_split:
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        else:
            raise TypeError("Invalid configuration for parameters split")

        # Make sure parameter splits are valid
        if sum(self.parameter_split_sizes) != out_features:
            raise ValueError(
                f"Trying to split weight buffer ({out_features=}) "
                f"with split sizes {self.parameter_split_sizes}"
            )

        # Adjust parameter splits for tensor-parallel distribution
        if self.parallel_mode == "column":
            for i, size in enumerate(self.parameter_split_sizes):
                if size % self.tp_size != 0:
                    raise RuntimeError(
                        f"Attempting to distribute a parameter with out_features={size} "
                        f"between {self.tp_size} tensor-parallel processes"
                    )
                self.parameter_split_sizes[i] = size // self.tp_size

        # Construct weight parameters
        # Note: Register weights together so that they are adjacent to
        # each other in LayerNormLinear.parameters(). This makes it
        # more likely that they will stay contiguous if the weights
        # are manipulated externally, e.g. by FSDP.
        offset = 0
        for i, split_size in enumerate(self.parameter_split_sizes):
            split_start = offset
            offset += split_size
            split_end = offset

            # Check if parameters are subviews of buffers
            is_subview = (split_start, split_end) != (0, self.out_features)
            if is_subview and with_fp8_params:
                raise RuntimeError("Splitting Float8Tensor into multiple params is not supported")

            # Construct weight parameter
            self.register_parameter(
                self.weight_names[i],
                torch.nn.Parameter(weight_tensor[split_start:split_end]),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
                fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
            )

        # Construct bias parameters if needed
        if self.use_bias:
            offset = 0
            for i, split_size in enumerate(self.parameter_split_sizes):
                split_start = offset
                offset += split_size
                split_end = offset
                self.register_parameter(
                    self.bias_names[i],
                    torch.nn.Parameter(bias_tensor[split_start:split_end]),
                    init_fn=init_method_constant(0.0),
                )
        else:
            for name in self.bias_names:
                bias = torch.Tensor().to(dtype=params_dtype, device=device)
                setattr(self, name, bias)

        if with_fp8_params:
            self.init_fp8_metadata()

        self.reset_parameters(defer_init=device == "meta")

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward LayerNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with LN.
        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))
        self.inf_ln_sm_margin = int(os.getenv("NVTE_INF_LAYERNORM_SM_MARGIN", "0"))

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        warnings.warn(
            "This method will be deprecated in an upcoming release. "
            "Update your code to use LayerNormLinear.reset_parameters() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.zero_centered_gamma:
            init.ones_(self.layer_norm_weight)
        else:
            init.zeros_(self.layer_norm_weight)
        if self.layer_norm_bias is not None:
            init.zeros_(self.layer_norm_bias)

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallelism attributes for layer norm parameters
            setattr(self.layer_norm_weight, "sequence_parallel", self.sequence_parallel)
            if self.normalization != "RMSNorm":
                setattr(self.layer_norm_bias, "sequence_parallel", self.sequence_parallel)

            # Set parallelism attributes for linear weights
            for weight in self.weight_names:
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, weight),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

            # Set parallelism attributes for linear biases
            if self.use_bias:
                for bias in self.bias_names:
                    if self.parallel_mode == "row":
                        setattr(getattr(self, bias), "sequence_parallel", self.sequence_parallel)
                    elif self.parallel_mode == "column":
                        set_tensor_model_parallel_attributes(getattr(self, bias), True, 0, 1)

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        fp8_output: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply layer normalization to the input followed by a linear transformation.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        """

        if FP8GlobalStateManager.fp8_graph_capturing():
            skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        else:
            skip_fp8_weight_update = None
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False

        with self.prepare_forward(
            inp,
            allow_non_contiguous=False # removed .contiguous from inside the layer
        ) as inp:

            # Get concatenated weight and bias tensors
            unfused_weights = [getattr(self, name) for name in self.weight_names]
            if any(isinstance(w, QuantizedTensor) for w in unfused_weights):
                if self.fp8:
                    if len(unfused_weights) != 1:
                        raise RuntimeError(
                            "Splitting QuantizedTensor into multiple params is not supported"
                        )
                else:
                    unfused_weights = [w.dequantize() for w in unfused_weights]
            weight_tensor = _noop_cat(unfused_weights)
            if self.use_bias:
                bias_tensor = _noop_cat([getattr(self, name) for name in self.bias_names])
            else:
                bias_tensor = getattr(self, self.bias_names[0])  # Unused
            
            # Get quantizers
            input_quantizer, weight_quantizer, output_quantizer = None, None, None
            grad_output_quantizer, grad_input_quantizer = None, None
            if self.fp8:
                input_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
                input_quantizer.internal = False
                weight_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
                weight_quantizer.internal = True
                if fp8_output:
                    output_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_OUTPUT]
                if torch.is_grad_enabled():
                    grad_output_quantizer = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT1]
                    grad_output_quantizer.internal = True


            if torch.is_grad_enabled():
                fwd_fn = _LayerNormLinear.apply
                args = []
            else:
                fwd_fn = _LayerNormLinear.forward
                args = [None]
            args += (
                inp,
                self.layer_norm_weight,
                self.layer_norm_bias,
                weight_tensor,
                bias_tensor,
                self.apply_bias and not self.gemm_bias_unfused_add,
                self.eps,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.fuse_wgrad_accumulation,
                input_quantizer,
                weight_quantizer,
                output_quantizer,
                grad_output_quantizer,
                grad_input_quantizer,
                is_cpu_offload_enabled(),
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.parallel_mode,
                self.return_layernorm_output,
                self.return_layernorm_output_gathered,
                torch.is_grad_enabled(),
                self.fwd_ln_sm_margin if torch.is_grad_enabled() else self.inf_ln_sm_margin,
                self.bwd_ln_sm_margin,
                self.zero_centered_gamma,
                self.normalization,
                self.ub_bulk_wgrad,
                self.ub_bulk_dgrad,
                self.ub_overlap_rs_dgrad,
                self.ub_overlap_ag,
                self.ub_name,
                fp8_output,
                self.fsdp_group,
                self,
                skip_fp8_weight_update
            )
            out = fwd_fn(*args)

        if self.return_layernorm_output:
            out, ln_out = out

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            if self.return_layernorm_output:
                return out, cast_if_needed(bias_tensor, self.activation_dtype), ln_out
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
        return out
