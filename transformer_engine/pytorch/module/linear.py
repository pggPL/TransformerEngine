# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

import transformer_engine_torch as tex

from .base import (
    get_workspace,
    get_ub,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ._common import _noop_cat
from ..fp8 import FP8GlobalStateManager
from ..utils import (
    divide,
    cast_if_needed,
    clear_tensor_data,
    init_method_constant,
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
from ..cpp_extensions import (
    general_gemm,
)
from ..constants import GemmParallelModes, dist_group_type
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ..tensor.quantized_tensor import (
    QuantizedTensor,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)

from ..cpu_offload import is_cpu_offload_enabled

__all__ = ["Linear"]


class _Linear(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        inp: torch.Tensor,
        bias: Optional[torch.Tensor],
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        input_quantizer: Optional[Quantizer],
        weight_quantizer: Optional[Quantizer],
        output_quantizer: Optional[Quantizer],
        grad_output_quantizer: Optional[Quantizer],
        grad_input_quantizer: Optional[Quantizer],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        is_grad_enabled: bool,
        ub_overlap_rs: bool,
        ub_overlap_ag: bool,
        ub_name: str,
        fp8_output: bool,
        fsdp_group: Union[dist_group_type, None],
        module: torch.nn.Module,
        skip_fp8_weight_update: bool,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Make sure input dimensions are compatible
        _, in_features = weight.shape
        inp_shape = inp.shape
        assert inp_shape[-1] == in_features, "GEMM not possible"

        tp_world_size = get_distributed_world_size(tp_group)
        ub_overlap_rs = False if tp_world_size == 1 else ub_overlap_rs

        weight_requires_grad = weight.requires_grad
        backward_needs_input = is_grad_enabled and weight_requires_grad

        # Prepare input tensor
        # Note: Cast to expected dtype and perform tensor-parallel communication
        inputmat = inp
        inputmat_total = None
        with_input_all_gather = parallel_mode == "column" and sequence_parallel
        own_quantized_input = False
        if fp8:
            if input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            if with_input_all_gather:
                input_quantizer.set_usage(rowwise=True, columnwise=False)
                inputmat_total, _ = gather_along_first_dim(
                    inputmat,
                    tp_group,
                    quantizer=input_quantizer,
                )
            else:
                input_quantizer.set_usage(
                    rowwise=True,
                    columnwise=(is_grad_enabled and weight_requires_grad),
                )
                inputmat = input_quantizer(inputmat)
                inputmat_total = inputmat
        else:
            inputmat = cast_if_needed(inp, activation_dtype)
            if with_input_all_gather:
                inputmat_total, _ = gather_along_first_dim(inputmat, tp_group)
            else:
                inputmat_total = inputmat

        # Cast weight to expected dtype
        weightmat = weight
        if not fp8:
            weightmat = cast_if_needed(weightmat, activation_dtype)
        else:
            if not isinstance(weight, QuantizedTensor):

                # Configure quantizer
                if weight_quantizer is not None:
                    columnwise_usage = is_grad_enabled and inp.requires_grad
                    if not columnwise_usage:
                        columnwise_usage = (
                            is_fp8_activation_recompute_enabled()
                            and not in_fp8_activation_recompute_phase()
                        )
                    weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)

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
                input_quantizer.calibrate(inputmat_total)
            if weight_quantizer is not None:
                weight_quantizer.calibrate(weight)

        if ub_overlap_rs:
            # I think this should be inside the gemm call rather than linear
            ub_obj_projout = get_ub(ub_name + "_fprop")
            ub_buffer = ub_obj_projout.get_ubuf_output(1)
            if ub_obj_projout.is_p2p_overlap():
                if ub_obj_projout.is_atomic_gemm():
                    ub_algo = tex.UbufOverlapAlgo.ATOMIC_GEMM_RS_P2P
                else:
                    ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS_P2P
            else:
                if ub_obj_projout.is_atomic_gemm():
                    ub_algo = tex.UbufOverlapAlgo.ATOMIC_GEMM_RS
                else:
                    ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS
            if fp8 and ub_obj_projout.is_fp8_ubuf():
                assert fp8_output
                ub_obj_projout.set_ubuf_scale_inv(
                    torch.reciprocal(output_quantizer.scale)
                )

        out, _, _ = general_gemm(
            weightmat,
            inputmat_total,
            get_workspace(),
            quantization_params=output_quantizer,
            out_dtype=activation_dtype,
            bias=bias,
            use_split_accumulator=_2X_ACC_FPROP,
            ub_algo=ub_algo if ub_overlap_rs else None,
            ub=ub_obj_projout if ub_overlap_rs else None,
            ub_buffer=ub_buffer if ub_overlap_rs else None,
        )

        if is_grad_enabled:
            saved_inputmat = None
            if backward_needs_input:
                if own_quantized_input and isinstance(inputmat, QuantizedTensor):
                    inputmat.update_usage(rowwise_usage=False)
                saved_inputmat = inputmat

            if cpu_offloading:
                weightmat.weight_offloading = True
                weight.weight_offloading = True
                if saved_inputmat is not None:
                    saved_inputmat.activation_offloading = True

            # Scatter intermediate/activation tensors saved for the backward pass
            # NOTE: FSDP sharding is not valid for models initialized with primary Fp8 weights
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                saved_inputmat,
                weightmat if fp8 and not isinstance(weight, QuantizedTensor) else None,
            )

            saved_input_tensors, saved_input = prepare_for_saving(saved_inputmat)
            saved_weight_tensors, saved_weight = prepare_for_saving(weightmat)
            ctx.save_for_backward(
                *saved_input_tensors,
                *saved_weight_tensors,
                bias,
                weight if (fuse_wgrad_accumulation and
                           hasattr(weight, "grad_added_to_main_grad")) else None
            )

            ctx.saved_input = saved_input
            ctx.saved_weight = saved_weight
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.input_quantizer = input_quantizer
            ctx.grad_output_quantizer = grad_output_quantizer
            ctx.grad_input_quantizer = grad_input_quantizer
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = bias is not None
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp_shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.ub_overlap_ag = ub_overlap_ag
            ctx.ub_name = ub_name
            ctx.tp_size = tp_size
            ctx.requires_dgrad = inp.requires_grad
            ctx.requires_wgrad = weight.requires_grad
            ctx.reduce_and_update_bwd_fp8_tensors = False
            ctx.owns_input = saved_inputmat is not inp
            ctx.is_input_fp8 = not own_quantized_input
            if ctx.fp8 and requires_grad(inp, weight, bias):
                _first_fp8_module = FP8GlobalStateManager.IS_FIRST_FP8_MODULE
                ctx.reduce_and_update_bwd_fp8_tensors = FP8GlobalStateManager.is_first_fp8_module()
                if in_fp8_activation_recompute_phase():
                    FP8GlobalStateManager.IS_FIRST_FP8_MODULE = _first_fp8_module

        # Row Parallel Linear
        if not ub_overlap_rs:
            if parallel_mode == "row" and sequence_parallel:
                out, _ = reduce_scatter_along_first_dim(out, tp_group)
            elif parallel_mode == "row" and tensor_parallel:
                out, _ = allreduce(out, tp_group)

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring

        with torch.cuda.nvtx.range("_Linear_backward"):
            saved_tensors = ctx.saved_tensors
            inputmat, saved_tensors = restore_from_saved(ctx.saved_input, saved_tensors)
            weight_fp8, saved_tensors = restore_from_saved(ctx.saved_weight, saved_tensors)
            (
                bias,
                weight
            ) = saved_tensors

            # Since main_grad can be modified inplace, it should not be a part of saved_tensors
            main_grad = weight.main_grad if weight is not None and ctx.fuse_wgrad_accumulation and weight.requires_grad else None

            # Gather intermediate/activation tensors if needed
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            _fsdp_gather_tensors(
                ctx.fsdp_group,
                ctx.fsdp_shapes,
                inputmat,
                weight_fp8,
            )

            #TODO: understand and fix
            # if ctx.cpu_offloading and ctx.fuse_wgrad_accumulation:
            #     weight = torch.nn.Parameter(weight, weight.requires_grad)
            #     weight.main_grad = main_grad

            tp_world_size = get_distributed_world_size(ctx.tp_group)
            ctx.ub_overlap_ag = False if tp_world_size == 1 else ctx.ub_overlap_ag
            ub_algo = None
            if ctx.ub_overlap_ag:
                dim_size = list(grad_output.size())
                dim_size[0] = dim_size[0] * tp_world_size
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                if ctx.ub_obj_gradout.is_atomic_gemm():
                    ub_algo = tex.CommOverlapAlgo.ATOMIC_GEMM_AG_P2P
                else:
                    ub_algo = tex.CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P

            # Prepare grad output tensor
            # Note: Cast to expected dtype and perform tensor-parallel communication
            if ctx.grad_output_quantizer is not None:
                ctx.grad_output_quantizer.set_usage(
                    rowwise=True,
                    columnwise=True, # TODO(pgadzinski) - remove
                )
            (
                grad_output,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx,
                grad_output,
                ctx.parallel_mode == "row",
                ctx.grad_output_quantizer,
            )

            # Prepare input tensor
            # Note: Perform tensor-parallel communication if needed
            inputmat_total = None
            inputmat_total_work = None
            with_input_all_gather = (
                ctx.requires_wgrad
                and ctx.parallel_mode == "column"
                and ctx.sequence_parallel
            )
            if ctx.requires_wgrad and ctx.parallel_mode == "column" and ctx.sequence_parallel:
                quantizer = None
                if ctx.fp8:
                    quantizer = ctx.input_quantizer
                    quantizer.set_usage(rowwise=True, columnwise=True)
                inputmat_total, inputmat_total_async = gather_along_first_dim(
                    inputmat,
                    ctx.tp_group,
                    async_op=True,
                    quantizer=quantizer,
                )
            else:
                inputmat_total = inputmat

            # Check whether to output wgrad GEMM directly into main grad
            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            # Compute grad input tensor
            dgrad = None
            dgrad_work = None
            if ctx.requires_dgrad:

                # Update quantizer
                if ctx.grad_input_quantizer is not None:
                    ctx.grad_input_quantizer.set_usage(rowwise=True, columnwise=False)

                # dgrad GEMM
                dgrad, _, _ = general_gemm(
                        weight_fp8,
                        grad_output,
                        get_workspace(),
                        layout="NN",
                        grad=True,
                        quantization_params=ctx.grad_input_quantizer,
                        out_dtype=ctx.activation_dtype,
                        use_split_accumulator=_2X_ACC_DGRAD,
                        ub_algo=ub_algo if ctx.ub_overlap_ag else None,
                        ub=ctx.ub_obj_gradout if ctx.ub_overlap_ag else None
                )

                # Launch tensor-parallel communication
                if ctx.parallel_mode == "column":
                    if ctx.sequence_parallel:
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
                if inputmat_total_work is not None:
                    inputmat_total_work.wait()
                    inputmat_total_work = None

                if ctx.fp8:
                    # TODO: deal with this
                    if ctx.ub_overlap_ag:
                        raise NotImplementedError
                        if isinstance(grad_output_c, QuantizedTensor):
                            grad_output_t = grad_output_c.transpose_2d()
                        else:
                            grad_output_t = tex.fp8_transpose(grad_output_c, fp8_dtype_backward)

                # wgrad GEMM
                # Note: Fuse with bgrad computation if needed
                wgrad, grad_bias_, _ = general_gemm(
                    inputmat_total,
                    grad_output,
                    get_workspace(),
                    layout="NT",
                    grad=True,
                    out_dtype=main_grad.dtype if ctx.fuse_wgrad_accumulation else ctx.activation_dtype,
                    out=main_grad if ctx.fuse_wgrad_accumulation else None,
                    use_split_accumulator=_2X_ACC_WGRAD,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                )
                if grad_bias is None:
                    grad_bias = grad_bias_
                del grad_bias_

                # Deallocate input tensor
                if ctx.owns_input:
                    clear_tensor_data(inputmat_total)

            # Don't return grad bias if not needed
            if not ctx.use_bias:
                grad_bias = None

            # Synchronize tensor parallel communication
            if inputmat_total_work is not None:
                inputmat_total_work.wait()
                inputmat_total_work = None
            if dgrad_work is not None:
                dgrad_work.wait()
                dgrad_work = None

        if ctx.requires_wgrad:
            # Handle custom DDP from mcore.
            if (ctx.fuse_wgrad_accumulation and
                weight is not None and
                hasattr(weight, "grad_added_to_main_grad")
            ):
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
        if ctx.fp8 and not isinstance(weight, QuantizedTensor):
            _fsdp_scatter_tensors(ctx.fsdp_group, weight_fp8)

        return (
            wgrad,
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            grad_bias,
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # input_quantizer
            None,  # weight_quantizer
            None,  # output_quantizer
            None,  # grad_output_quantizer
            None,  # grad_input_quantizer
            None,  # fuse_wgrad_accumulation
            None,  # cpu_offloading
            None,  # tp_group
            None,  # tp_size
            None,  # sequence_parallel
            None,  # tensor_parallel
            None,  # activation_dtype
            None,  # parallel_mode
            None,  # is_grad_enabled
            None,  # ub_overlap_rs
            None,  # ub_overlap_ag
            None,  # ub_name
            None,  # fp8_output
            None,  # fsdp_group
            None,  # module
            None,  # skip_fp8_weight_update
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
    get_rng_state_tracker : Callable, default = `None`
                 used to get the random number generator state tracker for initializing weights.
    rng_tracker_name : str, default = `None`
                 the param passed to get_rng_state_tracker to get the specific rng tracker.
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      Configuration for splitting the weight and bias tensors along dim 0 into
                      multiple PyTorch parameters. If a list or tuple of strings is provided,
                      they are used to make the names of equally-sized parameters. If a dict
                      (preferably an OrderedDict) is provided, the keys are used as names and
                      values as split sizes along dim 0. The resulting parameters will have
                      names that end in `_weight` or `_bias`, so trailing underscores are
                      stripped from any provided names.
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
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.ub_overlap_rs = ub_overlap_rs
        self.ub_overlap_ag = ub_overlap_ag
        if ub_overlap_rs or ub_overlap_ag:
            assert ub_name is not None, "Userbuffer name [string] is not set."
        self.ub_name = ub_name
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name

        if device == "meta":
            assert parameters_split is None, "Cannot split module parameters on 'meta' device."
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
        # each other in Linear.parameters(). This makes it more likely
        # that they will stay contiguous if the weights are
        # manipulated externally, e.g. by FSDP.
        offset = 0
        for i, split_size in enumerate(self.parameter_split_sizes):
            split_start = offset
            offset += split_size
            split_end = offset

            # Check if parameters are subviews of buffers
            is_subview = (split_start, split_end) != (0, self.out_features)
            if is_subview and with_fp8_params:
                raise RuntimeError("Splitting QuantizedTensor into multiple params is not supported")

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

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        fp8_output: Optional[bool] = False,
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
        if FP8GlobalStateManager.fp8_graph_capturing():
            skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        else:
            skip_fp8_weight_update = None
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False

        with self.prepare_forward(
            inp,
            allow_non_contiguous=isinstance(inp, QuantizedTensor),
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
                bias_tensor = None

            # Get quantizers
            input_quantizer, weight_quantizer, output_quantizer = None, None, None
            grad_output_quantizer, grad_input_quantizer = None, None
            if self.fp8:
                input_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
                input_quantizer.internal = True
                weight_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
                weight_quantizer.internal = True
                if fp8_output:
                    output_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_OUTPUT]
                if torch.is_grad_enabled():
                    grad_output_quantizer = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT1]
                    grad_output_quantizer.internal = True

            # Make sure weight tensor has correct quantizer
            # Note: Quantizer might have changed if quantization
            # recipe changed
            if weight_quantizer is not None and isinstance(weight_tensor, QuantizedTensor):
                weight_tensor._quantizer = weight_quantizer

            if torch.is_grad_enabled():
                linear_fn = _Linear.apply
                args = []
            else:
                linear_fn = _Linear.forward
                args = [None]
            args += (
                weight_tensor,
                inp,
                bias_tensor if (self.apply_bias and not self.gemm_bias_unfused_add) else None,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                input_quantizer,
                weight_quantizer,
                output_quantizer,
                grad_output_quantizer,
                grad_input_quantizer,
                self.fuse_wgrad_accumulation,
                is_cpu_offload_enabled(),
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.parallel_mode,
                torch.is_grad_enabled(),
                self.ub_overlap_rs,
                self.ub_overlap_ag,
                self.ub_name,
                fp8_output,
                self.fsdp_group,
                self,
                skip_fp8_weight_update,
            )
            out = linear_fn(*args)
        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        return out
