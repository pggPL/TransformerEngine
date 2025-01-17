# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
import logging
import os
from typing import Union, Optional, Callable, Tuple, List, Dict, Any

import torch

import transformer_engine_torch as tex

from .base import (
    get_workspace,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ....pytorch.module._common import _noop_cat
from ....pytorch.fp8 import get_fp8_te_dtype, FP8GlobalStateManager
from ....pytorch.utils import (
    divide,
    cast_if_needed,
    assert_dim_for_fp8_exec,
    clear_tensor_data,
    init_method_constant,
    requires_grad
)
from ....pytorch.distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
)
from ....pytorch.cpp_extensions import (
    fp8_gemm,
    gemm,
    cast_to_fp8,
)
from ....pytorch.constants import GemmParallelModes, dist_group_type
from ....pytorch.jit import no_torch_dynamo
from ....pytorch.float8_tensor import Float8Tensor
from ....pytorch.graph import is_graph_capturing

from ...debug_state import DebugLayerState, set_weight_tensor_tp_group_reduce
try:
    import nvtorch_inspect.api as nvinspect_api
except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError("ERROR: Could not locate nvtorch_inspect package. Make sure it is installed correctly.")


__all__ = ["Linear"]

def maybe_gather_tensor(tensor, parallel_mode, sequence_parallel, tp_group, async_op=False):
    if parallel_mode == "column" and sequence_parallel:
        tensor_total, handle = gather_along_first_dim(tensor, tp_group, async_op)
        return tensor_total, handle
    else:
        return tensor, None


def maybe_gather_grad_tensor(tensor, parallel_mode, sequence_parallel, tp_group, async_op=False):
    if parallel_mode == "row" and sequence_parallel:
        tensor_total, handle = gather_along_first_dim(tensor, tp_group, async_op)
        return tensor_total, handle
    else:
        tensor_total = tensor

    return tensor_total, None


class _Linear(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: Union[Float8Tensor, torch.Tensor],
        weight_fp8_fprop: Union[Float8Tensor, None],
        weight_fp8_t_dgrad: Union[Float8Tensor, None],
        inp: torch.Tensor,
        bias: torch.Tensor,
        use_bias: bool,
        is_first_microbatch: Union[bool, None],
        skip_fp8_weight_update: Union[torch.Tensor, None],
        fp8: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        is_grad_enabled: bool,
        primary_weights_in_fp8: bool,
        ub_overlap_rs: bool,
        ub_overlap_ag: bool,
        ub_name: str,
        is_first_module_in_mha: bool,
        name: str,
        step_counter: int,
    ) -> torch.Tensor:
        # Make sure input dimensions are compatible
        in_features = weight.shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view(-1, in_features)
        if fp8:
            assert_dim_for_fp8_exec(inputmat)
            assert_dim_for_fp8_exec(weight)

        update_fp8_weights = (
            is_first_microbatch is None
            or is_first_microbatch
            or skip_fp8_weight_update is not None
        )

        # Cast input to expected dtype
        inputmat = cast_if_needed(inputmat, activation_dtype)
        inputmat_no_fp8 = inputmat

        if fp8:
            bias_dtype = (
                torch.bfloat16
                if activation_dtype == torch.float32
                else activation_dtype
            )
            bias = cast_if_needed(bias, bias_dtype) if use_bias else bias
            fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

            if DebugLayerState.get(name).DelayedScaling.FPROP_ACTIVATION or DebugLayerState.get(name).DelayedScaling.WGRAD_ACTIVATION:
                inputmat_fp8_ds = cast_to_fp8(
                                    inputmat,
                                    fp8_meta["scaling_fwd"],
                                    tex.FP8FwdTensors.GEMM1_INPUT,
                                    fp8_dtype_forward)
            if update_fp8_weights:
                if DebugLayerState.get(name).DelayedScaling.FPROP_WEIGHT:
                    weight_fp8_fprop = Float8Tensor(
                        data=weight_fp8_fprop._data,
                        fp8_meta=fp8_meta,
                        fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
                    )
                    cast_to_fp8(
                            weight,
                            fp8_meta["scaling_fwd"],
                            tex.FP8FwdTensors.GEMM1_WEIGHT,
                            fp8_dtype_forward,
                            out=weight_fp8_fprop._data)
                
                elif DebugLayerState.get(name).FP8Gemm.FPROP:
                    # current scaling
                    ret = nvinspect_api.transformer_engine.process_tensor(
                        name, gemm="fprop", tensor_name="weight", tensor=weight,  
                        fp8_enabled=fp8, fp8_dtype=fp8_dtype_forward, out=weight_fp8_fprop._data)
                    weight_fp8_fprop._scale_inv = ret["scale_inv"]
                    
                if DebugLayerState.get(name).DelayedScaling.DGRAD_WEIGHT:
                    weight_fp8_t_dgrad = Float8Tensor(
                        data=weight_fp8_t_dgrad._data,
                        fp8_meta=fp8_meta,
                        fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
                    )
                    weight_fp8 = cast_to_fp8(
                            weight,
                            fp8_meta["scaling_fwd"],
                            tex.FP8FwdTensors.GEMM1_WEIGHT,
                            fp8_dtype_forward)
                    tex.fp8_transpose_noalloc(weight_fp8, weight_fp8_t_dgrad._data, fp8_dtype_forward)

                elif DebugLayerState.get(name).FP8Gemm.DGRAD:
                    # current scaling
                    ret = nvinspect_api.transformer_engine.process_tensor(
                        name, gemm="dgrad", tensor_name="weight", tensor=weight.t().contiguous(),  
                        fp8_enabled=fp8, fp8_dtype=fp8_dtype_forward, out=weight_fp8_t_dgrad._data)
                    weight_fp8_t_dgrad._scale_inv = ret["scale_inv"]
                    
            
        # always gather for statistics
        inputmat_no_fp8_total, _ = maybe_gather_tensor(inputmat_no_fp8, parallel_mode, sequence_parallel, tp_group)
        nvinspect_api.transformer_engine.save_stats_for_logging(name, tensor=inputmat_no_fp8_total, tensor_name="activation", iteration=step_counter)

        # weight tensor has different reduciton group - tp_group. If DebugLayerState.weight_tensor_tp_group_reduce is set, it is not reduced.
        red_group = nvinspect_api.get_tensor_reduction_group()
        skip_reduction = not DebugLayerState.weight_tensor_tp_group_reduce
        nvinspect_api.set_tensor_reduction_group(tp_group)
        nvinspect_api.transformer_engine.save_stats_for_logging(name, tensor=weight, tensor_name="weight", iteration=step_counter, skip_reduction=skip_reduction)
        nvinspect_api.set_tensor_reduction_group(red_group)
                
        if DebugLayerState.get(name).FP8Gemm.FPROP:
            # Activation cast
            qinputmat_dict = {}
            if DebugLayerState.get(name).DelayedScaling.FPROP_ACTIVATION:
                nvinspect_api.log_message("Fprop Activation: Delayed Scaling", name)
                qinputmat_dict["scale_inv"] = fp8_meta["scaling_fwd"].scale_inv
                qinputmat_dict["index"] = tex.FP8FwdTensors.GEMM1_INPUT
                qinputmat_dict["tensor"], _ = maybe_gather_tensor(inputmat_fp8_ds, parallel_mode, sequence_parallel, tp_group)
            else:
                nvinspect_api.log_message("Fprop Activation: Current Scaling", name)
                qinputmat_dict.update(nvinspect_api.transformer_engine.process_tensor(name, gemm="fprop", tensor_name="activation", tensor=inputmat_no_fp8_total, fp8_enabled=fp8, fp8_dtype=fp8_dtype_forward))
                qinputmat_dict["index"] = 0

            # Weight cast
            qweight_dict = {}
            if DebugLayerState.get(name).DelayedScaling.FPROP_WEIGHT:
                nvinspect_api.log_message("Fprop Weight: Delayed Scaling", name)
                qweight_dict["tensor"] = weight_fp8_fprop._data
                qweight_dict["scale_inv"] = fp8_meta["scaling_fwd"].scale_inv
                qweight_dict["index"] = tex.FP8FwdTensors.GEMM1_WEIGHT
            else:
                nvinspect_api.log_message("Fprop Weight: Current Scaling", name)
                qweight_dict["tensor"] = weight_fp8_fprop._data
                qweight_dict["scale_inv"] = weight_fp8_fprop._scale_inv
                qweight_dict["index"] = 0
            

            nvinspect_api.transformer_engine.save_fp8_stats_for_logging(name, tensor=qinputmat_dict["tensor"], tensor_name="activation", iteration=step_counter)

            # weight tensor has different reduciton group - tp_group. If DebugLayerState.weight_tensor_tp_group_reduce is set, it is not reduced.
            skip_reduction = not DebugLayerState.weight_tensor_tp_group_reduce
            nvinspect_api.set_tensor_reduction_group(tp_group)
            nvinspect_api.transformer_engine.save_fp8_stats_for_logging(name, tensor=qweight_dict["tensor"], tensor_name="weight", iteration=step_counter, skip_reduction=skip_reduction)
            nvinspect_api.set_tensor_reduction_group(red_group)

            dim_size = list(qinputmat_dict["tensor"].size())
            dim_size[1] = weight.size(0)
            out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat.device)
            
            proj_out_index, meta_tensor, proj_out_tetype, proj_out_pttype = (
                None, None, None, activation_dtype)
            _ = fp8_gemm(
                qweight_dict["tensor"],
                qweight_dict["scale_inv"],
                qweight_dict["index"],
                fp8_dtype_forward,
                qinputmat_dict["tensor"],
                qinputmat_dict["scale_inv"],
                qinputmat_dict["index"],
                fp8_dtype_forward,
                proj_out_pttype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                use_split_accumulator=_2X_ACC_FPROP,
                out=out,
                ub_algo=None,
                ub=None,
                extra_output_tensor=None,
                out_index=proj_out_index,
                fp8_meta_tensor = meta_tensor,
                D_dtype = proj_out_tetype,
            )
        else:
            nvinspect_api.log_message("Fprop: High Precision", name)
            # Cast for native AMP
            weight = cast_if_needed(weight, activation_dtype)
            bias = cast_if_needed(bias, activation_dtype) if use_bias else bias

            inputmat_dict = nvinspect_api.transformer_engine.process_tensor(name, gemm="fprop", tensor_name="activation", tensor=inputmat_no_fp8_total, fp8_enabled=fp8)
            weight_dict = nvinspect_api.transformer_engine.process_tensor(name, gemm="fprop", tensor_name="weight", tensor=weight, fp8_enabled=fp8)

            dim_size = list(inputmat_no_fp8_total.size())
            dim_size[1] = weight.size(0)
            out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_no_fp8_total.device)

            _ = gemm(
                weight_dict["tensor"],
                inputmat_dict["tensor"],
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                out=out,
                ub_algo=None,
                ub=None,
                extra_output_tensor=None,
            )


        if is_grad_enabled:
            saved_inputmat = inputmat_no_fp8
            saved_weight = weight
            saved_fp8_inputmat = None
            saved_fp8_t_weight = None
            saved_fp8_t_weight_scale_inv = None
            weight_requires_grad = weight.requires_grad
            if fp8:
                if weight_requires_grad and DebugLayerState.get(name).DelayedScaling.WGRAD_ACTIVATION:
                    saved_fp8_inputmat = inputmat_fp8_ds
            
                if DebugLayerState.get(name).FP8Gemm.DGRAD:
                    saved_fp8_t_weight = weight_fp8_t_dgrad._data
                    saved_fp8_t_weight_scale_inv = weight_fp8_t_dgrad._scale_inv

            ctx.save_for_backward(
                saved_inputmat,
                saved_weight,
                saved_fp8_inputmat,
                saved_fp8_t_weight,
                saved_fp8_t_weight_scale_inv,
                fp8_meta["scaling_fwd"].scale_inv.clone() if fp8 else None
            )
            
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_meta = fp8_meta
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp.shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.requires_dgrad = inp.requires_grad
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if ctx.fp8 and requires_grad(inp, weight, bias):
                ctx.reduce_and_update_bwd_fp8_tensors = (
                    ctx.reduce_and_update_bwd_fp8_tensors or
                    FP8GlobalStateManager.is_first_fp8_module())
            
            ctx.name = name
            ctx.step_counter = step_counter
            ctx.weight_requires_grad = weight_requires_grad
        # Row Parallel Linear
        if parallel_mode == "row" and sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])


    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        with torch.cuda.nvtx.range("_Linear_backward"):
            (
                inputmat,
                weight,
                inputmat_fp8,
                weight_fp8_t,
                weight_fp8_t_scale_inv,
                fwd_scale_inverses
            ) = ctx.saved_tensors

            grad_output = grad_output.contiguous()
            grad_output = grad_output.view((-1, grad_output.shape[-1]))
            grad_bias = torch.sum(grad_output, dim=0) if ctx.use_bias else None

            if ctx.fp8:
                fp8_dtype_backward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=False)
                fp8_dtype_forward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=True)
                
                if DebugLayerState.get(ctx.name).DelayedScaling.DGRAD_GRADIENT or DebugLayerState.get(ctx.name).DelayedScaling.WGRAD_GRADIENT:
                    grad_output_c = cast_to_fp8(
                            grad_output,
                            ctx.fp8_meta["scaling_bwd"],
                            tex.FP8BwdTensors.GRAD_OUTPUT1,
                            fp8_dtype_backward)
                    grad_output_c, _ = maybe_gather_grad_tensor(grad_output_c, ctx.parallel_mode, ctx.sequence_parallel, ctx.tp_group)
            
            grad_output, _ = maybe_gather_grad_tensor(grad_output, ctx.parallel_mode, ctx.sequence_parallel, ctx.tp_group)
            nvinspect_api.transformer_engine.save_stats_for_logging(ctx.name, tensor=grad_output, tensor_name="gradient", iteration=ctx.step_counter)

            if DebugLayerState.get(ctx.name).DelayedScaling.WGRAD_ACTIVATION:
                inputmat_total, handle = maybe_gather_tensor(inputmat_fp8, ctx.parallel_mode, ctx.sequence_parallel, ctx.tp_group, async_op=ctx.requires_dgrad)
            else:
                inputmat_total, handle = maybe_gather_tensor(inputmat, ctx.parallel_mode, ctx.sequence_parallel, ctx.tp_group, async_op=ctx.requires_dgrad)

            if ctx.requires_dgrad:
                if DebugLayerState.get(ctx.name).FP8Gemm.DGRAD:
                    out_index, meta_tensor, output_te_dtype, output_dtype = (
                        None, None, None, ctx.activation_dtype)
                    
                    dgrad_qgrad_dict = {}
                    dgrad_qweight_dict = {}

                    if DebugLayerState.get(ctx.name).DelayedScaling.DGRAD_GRADIENT:
                        nvinspect_api.log_message("Dgrad Gradient: Delayed Scaling", ctx.name)
                        dgrad_qgrad_dict["tensor"] = grad_output_c
                        dgrad_qgrad_dict["scale_inv"] = ctx.fp8_meta["scaling_bwd"].scale_inv
                        dgrad_qgrad_dict["index"] = tex.FP8BwdTensors.GRAD_OUTPUT1
                    else:
                        nvinspect_api.log_message("Dgrad Gradient: Current Scaling", ctx.name)
                        dgrad_qgrad_dict = nvinspect_api.transformer_engine.process_tensor(ctx.name, gemm="dgrad", tensor_name="gradient", tensor=grad_output, fp8_enabled=ctx.fp8, fp8_dtype=fp8_dtype_backward)
                        dgrad_qgrad_dict["index"] = 0
                    
                    if DebugLayerState.get(ctx.name).DelayedScaling.DGRAD_WEIGHT:
                        nvinspect_api.log_message("Dgrad Weight: Delayed Scaling", ctx.name)
                        assert  weight_fp8_t != None
                        dgrad_qweight_dict["tensor"] = weight_fp8_t
                        dgrad_qweight_dict["scale_inv"] = fwd_scale_inverses
                        dgrad_qweight_dict["index"] = tex.FP8FwdTensors.GEMM1_WEIGHT
                    else:
                        nvinspect_api.log_message("Dgrad Weight: Current Scaling", ctx.name)
                        assert  weight_fp8_t != None
                        dgrad_qweight_dict["tensor"] = weight_fp8_t
                        dgrad_qweight_dict["scale_inv"] = weight_fp8_t_scale_inv
                        dgrad_qweight_dict["index"] = 0

                    nvinspect_api.transformer_engine.save_fp8_stats_for_logging(ctx.name, tensor=dgrad_qgrad_dict["tensor"], tensor_name="gradient", iteration=ctx.step_counter)

                    dgrad, _ = fp8_gemm(
                        dgrad_qweight_dict["tensor"],
                        dgrad_qweight_dict["scale_inv"],
                        dgrad_qweight_dict["index"],
                        fp8_dtype_forward,
                        dgrad_qgrad_dict["tensor"],
                        dgrad_qgrad_dict["scale_inv"],
                        dgrad_qgrad_dict["index"],
                        fp8_dtype_backward,
                        output_dtype,
                        get_workspace(),
                        use_split_accumulator=_2X_ACC_DGRAD,
                        ub_algo=None,
                        ub=None,
                        out_index=out_index,
                        fp8_meta_tensor=meta_tensor,
                        D_dtype=output_te_dtype,
                    )
                else:
                    nvinspect_api.log_message("Dgrad: High Precision", ctx.name)
                    dgrad_grad_dict = nvinspect_api.transformer_engine.process_tensor(ctx.name, tensor=grad_output, tensor_name="gradient",
                                                                         gemm="dgrad", fp8_enabled=ctx.fp8)
                    dgrad_weight_dict = nvinspect_api.transformer_engine.process_tensor(ctx.name, tensor=weight, tensor_name="weight",
                                                                gemm="dgrad", fp8_enabled=ctx.fp8)
                    dgrad, _, _ = gemm(
                        dgrad_weight_dict["tensor"],
                        dgrad_grad_dict["tensor"],
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NN",
                        grad=True,
                        ub_algo=None,
                        ub=None,
                    )

                # Overlap dgrad-RS/AR with wgrad
                if ctx.parallel_mode == "column" and ctx.sequence_parallel:
                    if handle is not None:
                        handle.wait()
                    dgrad, handle = reduce_scatter_along_first_dim(
                        dgrad, ctx.tp_group, async_op=True
                    )
                elif ctx.parallel_mode == "column" and ctx.tensor_parallel:
                    dgrad, handle = allreduce(dgrad, ctx.tp_group, async_op=True)

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation


            if ctx.weight_requires_grad:
                if DebugLayerState.get(ctx.name).FP8Gemm.WGRAD:
                    wgrad_qinput_dict = {}
                    wgrad_qgrad_dict = {}

                    if DebugLayerState.get(ctx.name).DelayedScaling.WGRAD_GRADIENT:
                        nvinspect_api.log_message("Wgrad Gradient: Delayed Scaling", ctx.name)
                        wgrad_qgrad_dict["tensor"] = tex.fp8_transpose(grad_output_c, fp8_dtype_backward)
                        wgrad_qgrad_dict["scale_inv"] = ctx.fp8_meta["scaling_bwd"].scale_inv
                        wgrad_qgrad_dict["index"] = tex.FP8BwdTensors.GRAD_OUTPUT1
                    else:
                        nvinspect_api.log_message("Wgrad Gradient: Current Scaling", ctx.name)
                        wgrad_qgrad_dict = nvinspect_api.transformer_engine.process_tensor(ctx.name, gemm="wgrad", tensor_name="gradient", tensor=grad_output.t().contiguous(), fp8_enabled=ctx.fp8, fp8_dtype=fp8_dtype_backward)
                        wgrad_qgrad_dict["index"] = 0
                    
                    if DebugLayerState.get(ctx.name).DelayedScaling.WGRAD_ACTIVATION:
                        nvinspect_api.log_message("Wgrad Activation: Delayed Scaling", ctx.name)
                        assert  inputmat_fp8 != None and inputmat_total != None
                        wgrad_qinput_dict["tensor"] = tex.fp8_transpose(inputmat_total, fp8_dtype_forward)
                        wgrad_qinput_dict["scale_inv"] = fwd_scale_inverses
                        wgrad_qinput_dict["index"] = tex.FP8FwdTensors.GEMM1_INPUT
                    else:
                        nvinspect_api.log_message("Wgrad Activation: Current Scaling", ctx.name)
                        wgrad_qinput_dict = nvinspect_api.transformer_engine.process_tensor(ctx.name, gemm="wgrad", tensor_name="activation", tensor=inputmat_total.t().contiguous(), fp8_enabled=ctx.fp8, fp8_dtype=fp8_dtype_forward)
                        wgrad_qinput_dict["index"] = 0

                    wgrad, _ = fp8_gemm(
                        wgrad_qinput_dict["tensor"],
                        wgrad_qinput_dict["scale_inv"],
                        wgrad_qinput_dict["index"],
                        fp8_dtype_forward,
                        wgrad_qgrad_dict["tensor"],
                        wgrad_qgrad_dict["scale_inv"],
                        wgrad_qgrad_dict["index"],
                        fp8_dtype_backward,
                        ctx.activation_dtype,
                        get_workspace(),
                        accumulate=accumulate_wgrad_into_param_main_grad,
                        out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                        use_split_accumulator=_2X_ACC_WGRAD,
                    )
                else:
                    nvinspect_api.log_message("Wgrad: High Precision", ctx.name)
                    wgrad_input_dict = nvinspect_api.transformer_engine.process_tensor(ctx.name, tensor=inputmat_total, tensor_name="activation",
                                                                    gemm="wgrad", fp8_enabled=ctx.fp8)
                    wgrad_grad_dict = nvinspect_api.transformer_engine.process_tensor(ctx.name, tensor=grad_output, tensor_name="gradient",
                                                                         gemm="wgrad", fp8_enabled=ctx.fp8)
                    wgrad, grad_bias, _ = gemm(
                        wgrad_input_dict["tensor"],
                        wgrad_grad_dict["tensor"],
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NT",
                        grad=True,
                        use_bias=ctx.use_bias,
                        accumulate=accumulate_wgrad_into_param_main_grad,
                        out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    )

                # Deallocate input tensor
                clear_tensor_data(inputmat_total)

            # Column Parallel Linear
            if ctx.parallel_mode == "column" and ctx.tensor_parallel and handle is not None:
                handle.wait()

        if ctx.weight_requires_grad:
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

        if not ctx.use_bias:
            grad_bias = None
        
        return (
            wgrad,
            None,
            None,
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class Linear(TransformerEngineBaseModule):
    """Applies a linear transformation to the incoming data :math:`y = xA^T + b`

    On NVIDIA GPUs it is a drop-in replacement for `torch.nn.Linear`.

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      Configuration for splitting the weight and bias tensors along dim 0 into
                      multiple PyTorch parameters. If a list or tuple of strings is provided,
                      they are used to make the names of equally-sized parameters. If a dict
                      (preferably an OrderedDict) is provided, the keys are used as names and
                      values as split sizes along dim 0. The resulting parameters will have
                      names that end in `_weight` or `_bias`, so trailing underscores are
                      stripped from any provided names.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will allocated. It is the user's
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
    parallel_mode : {None, 'Column', 'Row'}, default = `None`
                   used to decide whether this Linear layer is Column Parallel Linear or Row
                   Parallel Linear as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
                   When set to `None`, no communication is performed.
    pp_group : ProcessGroup, default = `None`
                pipeline parallel process group.

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

                  
    Debug: Optimization parameters disabled
    ---------------------------------------
    - userbuffers
    - Primary weights in FP8
    - CPU offloading
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        rng_tracker_name: Optional[str] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_rs: bool = False,
        ub_overlap_ag: bool = False,
        ub_name: Optional[str] = None,
        name: str = None,
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.primary_weights_in_fp8 = False
        self.ub_overlap_rs = ub_overlap_rs
        self.ub_overlap_ag = ub_overlap_ag
        self.ub_name = ub_name
        self.name = name
        self.step_counter = 0
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name

        if device == 'meta':
            assert parameters_split is None, ("Cannot split module parameters "
                                              "on 'meta' device.")
        if (ub_overlap_ag or ub_overlap_rs):
            nvinspect_api.log_message("[DEBUG-WARNING] UserBuffers are not supported in debug module. "
                                    "Using UB optimization will not affect the debug module. ",
                                    level=logging.WARNING)

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

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

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
        # each other in Linear.parameters(). This makes it more likely
        # that they will stay contiguous if the weights are
        # manipulated externally, e.g. by FSDP.
        offset = 0
        for i, split_size in enumerate(self.parameter_split_sizes):
            split_start = offset
            offset += split_size
            split_end = offset

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

        self.reset_parameters(defer_init=(device == 'meta'))

        self.fp8_weight_shapes.append(torch.Size((self.out_features, self.in_features)))

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
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

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[Float8Tensor]:
        """
        Fetch the fp8 weight tensor placeholders if they exist (when
        `is_first_microbatch` is not `None`) or return empty fp8 weight
        tensors (if `is_first_microbatch is None`)
        """
        if not self.fp8:
            return [None, None]

        if is_first_microbatch is None:
            # Return empty weight placeholders for each fwd/bwd pass
            fp8_weight_tensors = self.get_fp8_weights_empty_tensors(
                is_first_microbatch
            )
        else:
            # These persistent weight placeholders should've been created in
            # `set_fp8_weights` method
            fp8_weight_tensors = [self.weight1_fp8, self.weight1_fp8_t]

        return fp8_weight_tensors

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        is_first_module_in_mha: Optional[bool] = False,
        overwrite_name: str = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        
        """
        Apply the linear transformation to the input.

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
        self._validate_name(overwrite_name)

        skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False
        
        if is_first_microbatch in [True, None]:
            self.step_counter += 1

        # This call initializes many of the required variables at every iteration.
        # Will this call, also allow calling self objects in other inherited modules?
        with self.prepare_forward(inp, 
                                  is_first_microbatch, 
                                  allow_non_contiguous=isinstance(inp,Float8Tensor)) as inp:
            
            DebugLayerState.initialize_state(self.name, fp8_enabled=self.fp8)

            # Get concatenated weight and bias tensors
            weight_tensor = _noop_cat(
                [getattr(self, name) for name in self.weight_names],
            )
            if self.use_bias:
                bias_tensor = _noop_cat(
                    [getattr(self, name) for name in self.bias_names],
                )
            else:
                bias_tensor = getattr(self, self.bias_names[0])  # Unused
            weight1_fp8, weight1_fp8_cs = (None, None)
            
            # Fetch the fp8 weights placeholders (for linear/gemm)
            weight1_fp8, weight1_fp8_cs = self.get_fp8_weights_scratchpad(
                is_first_microbatch
            )
                
            if torch.is_grad_enabled():
                linear_fn = _Linear.apply
                args = []
            else:
                linear_fn = _Linear.forward
                args = [None]
            args += (
                weight_tensor,
                weight1_fp8, # holds delayed scaling weights
                weight1_fp8_cs, # holds current scaling weights
                inp,
                bias_tensor,
                self.apply_bias and not self.gemm_bias_unfused_add,
                is_first_microbatch,
                skip_fp8_weight_update,
                self.fp8,
                self.fp8_calibration,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                False, # Disabled CPUOffloadEnabled
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.parallel_mode,
                torch.is_grad_enabled(),
                self.primary_weights_in_fp8,
                self.ub_overlap_rs,
                self.ub_overlap_ag,
                self.ub_name,
                is_first_module_in_mha,
                self.name,
                self.step_counter,
            )
            out = linear_fn(*args)

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        return out
