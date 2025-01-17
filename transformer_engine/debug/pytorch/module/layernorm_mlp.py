# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNormMLP API"""
import logging
from typing import Union, Optional, Callable, Tuple, List, Dict, Any
import pickle
from ....pytorch.export import is_in_onnx_export_mode

from ....pytorch.module import LayerNorm, RMSNorm
from .linear import Linear
import io

import torch

from .base import (
    TransformerEngineBaseModule,
)


from ....pytorch.constants import dist_group_type
from ....pytorch.jit import no_torch_dynamo

from ....pytorch.constants import dist_group_type, TE_DType

from ....pytorch import cpp_extensions as tex

from ....pytorch.distributed import (
    set_tensor_model_parallel_attributes,
    gather_along_first_dim
)

try:
    import nvtorch_inspect.api as nvinspect_api
except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError("ERROR: Could not locate nvtorch_inspect package. Make sure it is installed correctly.")


from ....pytorch.utils import get_default_init_method

__all__ = ["LayerNormMLP"]

def _create_torch_function_from_activation(name, forward, backward):
    obj = type(
        name, 
        (torch.autograd.Function,), 
        {'forward': staticmethod(forward), 'backward': staticmethod(backward)}
    )
    return obj

def _act_func(activation: str):
    funcs = {
        "gelu": (tex.gelu, tex.dgelu),
        "relu": (tex.relu, tex.drelu),
        "geglu": (tex.geglu, tex.dgeglu),
        "reglu": (tex.reglu, tex.dreglu),
        "swiglu": (tex.swiglu, tex.dswiglu),
        "qgelu": (tex.qgelu, tex.dqgelu),
        "srelu": (tex.srelu, tex.dsrelu),
    }
    if activation not in funcs:
        raise NotImplementedError("Activation type " + activation + " is not supported!")

    def forward_func(ctx, x, funcs, activation):
        ctx.fc1_out = x
        return funcs[activation][0](x, None, tex.FP8FwdTensors.GEMM2_INPUT, TE_DType[x.dtype])
    
    return _create_torch_function_from_activation(
        activation,
        lambda ctx, x: forward_func(ctx, x, funcs, activation),
        lambda ctx, x: funcs[activation][1](x, ctx.fc1_out, TE_DType[x.dtype]).view(*ctx.fc1_out.shape)
    )


class LayerNormMLP(TransformerEngineBaseModule):
    r"""
    Applies layer normalization on the input followed by the MLP module, consisting of
    2 successive linear transformations, separated by the GeLU activation.

    This debug module uses te.pytorch.LayerNorm or te.pytorch.RMSNorm and te.debug.pytorch.Linear and activations functions. 
    Logging and GEMMs are selected in Linear.
    Normalization is performed in higher precision and all casts occur in te.debug.pytorch.Linear.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    ffn_hidden_size : int
                     intermediate size to which input samples are projected.
    eps : float, default = 1e-5
         a value added to the denominator of layer normalization for numerical stability.
    bias : bool, default = `True`
          if set to `False`, the FC1 and FC2 layers will not learn an additive bias.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    activation : str, default = 'gelu'
          activation function used.
          Options: 'gelu', 'geglu', 'relu', 'reglu', 'squared_relu', 'swiglu', 'qgelu', 'srelu'.
    init_method : Callable, default = `None`
                 used for initializing FC1 weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.torch.nninit.normal_(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
                              used for initializing FC2 weights in the following way:
                              `output_layer_init_method(weight)`. When set to `None`, defaults to
                              `torch.torch.nninit.normal_(mean=0.0, std=0.023)`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module
                             is taken post layernorm.
    return_layernorm_output_gathered : bool, default = `False`
                             if set to `True`, output of layernorm is returned after the all
                             gather operation. Ignored if return_layernorm_output is False.
                             Example use case: with sequence parallel, input to residual connection
                             for transformer module (e.g. LoRA) will need to be gathered.
                             Returning layernorm output gathered will prevent a redundant gather.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.

    Parallelism parameters
    ----------------------
    set_parallel_mode : bool, default = `False`
                      if set to `True`, FC1 is used as Column Parallel and FC2 is used as Row
                      Parallel as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
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

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias for FC2, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    seq_length: int
               sequence length of input samples. Needed for JIT Warmup, a technique where jit fused
               functions are warmed up before training to ensure same kernels are used for forward
               propogation and activation recompute phase.
    micro_batch_size: int
                     batch size per training step. Needed for JIT Warmup, a technique where jit
                     fused functions are warmed up before training to ensure same kernels are
                     used for forward propogation and activation recompute phase.

    
    Debug
    -----------------------

    Debug: Optimization parameters disabled
    ---------------------------------------
    - userbuffers
    - Primary weights in FP8
    - CPU offloading
    - GEMM gelu fusion
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        return_bias: bool = False,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        normalization: str = "LayerNorm",
        activation: str = "gelu",
        output_layer_init_method: Optional[Callable] = None,
        fuse_wgrad_accumulation: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_layernorm_output: bool = False,
        return_layernorm_output_gathered: bool = False,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        set_parallel_mode: bool = False,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "cuda",
        ub_bulk_wgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_overlap_rs: bool = False,
        ub_overlap_ag: bool = False,
        name: str = None
    ) -> None:
        super().__init__()
        
        if (ub_bulk_wgrad or ub_bulk_dgrad or ub_overlap_ag or ub_overlap_rs or ub_overlap_rs_dgrad):
            nvinspect_api.log_message("> UserBuffers are not supported in debug module. "
                                    "Using UB optimization will not affect the debug module. ",
                                    level=logging.WARNING)
        
        if (seq_length or micro_batch_size):
            nvinspect_api.log_message("> JIT warmup is not supported in this module. ",
                                    level=logging.WARNING)

        assert normalization in ["LayerNorm", "RMSNorm"], "Unsupported normalization type!"

        self.name = name
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.return_layernorm_output = return_layernorm_output
        self.return_layernorm_output_gathered = return_layernorm_output_gathered

        if normalization == 'LayerNorm':
            normalization_cls = LayerNorm
        elif normalization == 'RMSNorm':
            normalization_cls = RMSNorm

        fc1_parallel_mode, fc2_parallel_mode = ("column", "row") \
            if set_parallel_mode else (None, None)

        if activation in ["reglu", "geglu", "swiglu"]:
            fc1_output_features = 2 * ffn_hidden_size
        else:
            fc1_output_features = ffn_hidden_size
        
        def _init_linear_layer(in_features, out_features, init_method, parallel_mode, name_suffix, return_bias=False):
            if init_method is None:
                init_method = get_default_init_method()
            return Linear(
                in_features=in_features,
                out_features=out_features,
                sequence_parallel=sequence_parallel,
                fuse_wgrad_accumulation=fuse_wgrad_accumulation,
                tp_group=tp_group,
                tp_size=tp_size,
                get_rng_state_tracker=get_rng_state_tracker,
                init_method=init_method,
                bias=bias,
                return_bias=return_bias,
                params_dtype=params_dtype,
                parallel_mode=parallel_mode,
                parameters_split=None,
                device=device,
                ub_overlap_rs=False,
                ub_overlap_ag=False,
                ub_name=None,
                name=(name + name_suffix if name is not None else None)
            )

        ln = normalization_cls(hidden_size, eps=eps, sequence_parallel=sequence_parallel,
                            params_dtype=params_dtype, zero_centered_gamma=zero_centered_gamma, device=device)
        fc1 = _init_linear_layer(hidden_size, fc1_output_features, init_method, fc1_parallel_mode, ".fc1")
        self.activation = _act_func(activation)
        
        fc2 = _init_linear_layer(ffn_hidden_size, hidden_size, output_layer_init_method, fc2_parallel_mode, ".fc2", return_bias=return_bias)

        # Move parameters from sublayers into LayerNormMLP as in the vanilia TE.
        self.layer_norm_weight = torch.nn.Parameter(ln.weight.data)
        ln.weight = self.layer_norm_weight

        if hasattr(ln, "bias"):
            self.layer_norm_bias = torch.nn.Parameter(ln.bias.data)
            ln.bias = self.layer_norm_bias

        self.fc1_weight = torch.nn.Parameter(fc1.weight.data)
        fc1.weight = self.fc1_weight

        if self.use_bias:
            self.fc1_bias = torch.nn.Parameter(fc1.bias.data)
            fc1.bias = self.fc1_bias

        self.fc2_weight = torch.nn.Parameter(fc2.weight.data)
        fc2.weight = self.fc2_weight

        if self.use_bias:
            self.fc2_bias = torch.nn.Parameter(fc2.bias.data)
            fc2.bias = self.fc2_bias

        # trick to not have any submodules like in the vanilia TE
        self.ln = [ln]
        self.fc1 = [fc1]
        self.fc2 = [fc2]

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        self.norm.reset_parameters()

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallel attributes for layer norm parameters
            setattr(self.layer_norm_weight, "sequence_parallel", self.sequence_parallel)
            if self.normalization != "RMSNorm":
                setattr(self.layer_norm_bias, "sequence_parallel", self.sequence_parallel)

            # Set parallel attributes for linear parameters
            set_tensor_model_parallel_attributes(self.fc1_weight, True, 0, 1)
            set_tensor_model_parallel_attributes(self.fc2_weight, True, 1, 1)
            if self.use_bias:
                set_tensor_model_parallel_attributes(self.fc1_bias, True, 0, 1)
                if self.set_parallel_mode:
                    setattr(self.fc2_bias, "sequence_parallel", self.sequence_parallel)

    @no_torch_dynamo()
    def forward(
        self, inp: torch.Tensor, is_first_microbatch: Optional[bool] = None, overwrite_name=None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        self._validate_name(overwrite_name)
        ln_out = self.ln[0](inp)

        ln_out_total = None
        if self.return_layernorm_output_gathered:
            ln_out_total, _ = gather_along_first_dim(ln_out, self.tp_group)

        fc1_out = self.fc1[0](ln_out,
            is_first_microbatch=is_first_microbatch, 
            is_first_module_in_mha=False,
            overwrite_name=self.name + ".fc1"
        )

        act_out = self.activation.apply(fc1_out)

        out = self.fc2[0](act_out,
            is_first_microbatch=is_first_microbatch, 
            is_first_module_in_mha=False,
            overwrite_name=self.name + ".fc2"
        )

        if self.return_bias:
            out = (out[0].view(-1, *inp.shape[1:-1], out[0].shape[-1]), out[1])
        else:
            out = out.view(-1, *inp.shape[1:-1], out.shape[-1])
        
        if self.return_layernorm_output_gathered:
            ln_out = ln_out_total

        if self.return_bias:
            if self.return_layernorm_output:
                return *out, ln_out
            return out

        if self.return_layernorm_output:
            return out, ln_out
        return out

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """Abstract method. Not used."""
        return

    def get_extra_state(self):
        """Save before checkpointing."""
        state = {}
        state_fc1 = self.fc1[0].get_extra_state()
        state_fc2 = self.fc2[0].get_extra_state()
        if isinstance(state_fc1, torch.Tensor):
            state_fc1 = pickle.loads(state_fc1.detach().cpu().numpy().tobytes())
            state_fc2 = pickle.loads(state_fc2.detach().cpu().numpy().tobytes())
        elif isinstance(state_fc1, io.BytesIO):
            state_fc1.seek(0)
            state_fc2.seek(0)
            state_fc1 = torch.load(state_fc1, map_location="cuda")
            state_fc2 = torch.load(state_fc2, map_location="cuda")

        fp8_checkpoint = self.fc1[0].fp8_meta["fp8_checkpoint"] or self.fc1[0].fp8 or self.fc1[0].fp8_calibration
        state["extra_fp8_variables"] = {
            "num_gemms": 2
        }

        if fp8_checkpoint:
            for key in state_fc1.keys():
                if key != "extra_fp8_variables":
                    # backward keys are in reverse order
                    if "fwd" in key:
                        state[key] = torch.cat((state_fc1[key], state_fc2[key]), dim=-1)
                    if "bwd" in key:
                        state[key] = torch.cat((state_fc2[key], state_fc1[key]), dim=-1)


        if is_in_onnx_export_mode():
            state_serialized = torch.frombuffer(pickle.dumps(state), dtype=torch.uint8)
        else:
            state_serialized = io.BytesIO()
            torch.save(state, state_serialized)

        return state_serialized

    def set_extra_state(self, state: torch.Tensor) -> None:
        """Load previous state."""
        if state is None:
            return

        if isinstance(state, torch.Tensor):
            state = pickle.loads(state.detach().cpu().numpy().tobytes())
        elif isinstance(state, io.BytesIO):
            state.seek(0)
            state = torch.load(state, map_location="cuda")
        else:
            raise RuntimeError("--Unsupported checkpoint format.")

        if state is None:
            return
        
        state_fc1, state_fc2 = {}, {}
        dim = -1
        for key in state.keys():
            if key != "extra_fp8_variables":
                tensor = state[key]
                split_point = tensor.size(dim) // 2
                # backward keys are in reverse order
                if "fwd" in key:
                    state_fc1[key] = tensor.narrow(dim, 0, split_point)
                    state_fc2[key] = tensor.narrow(dim, split_point, tensor.size(dim) - split_point)
                if "bwd" in key:
                    state_fc1[key] = tensor.narrow(dim, split_point, tensor.size(dim) - split_point)
                    state_fc2[key] = tensor.narrow(dim, 0, split_point)

        state_fc1['extra_fp8_variables'] = {
            'num_gemms': 1
        }
        state_fc1_serialized = io.BytesIO()
        torch.save(state_fc1, state_fc1_serialized)
        self.fc1[0].set_extra_state(state_fc1_serialized)

        state_fc2['extra_fp8_variables'] = {
            'num_gemms': 1
        }
        state_fc2_serialized = io.BytesIO()
        torch.save(state_fc2, state_fc2_serialized)
        self.fc2[0].set_extra_state(state_fc2_serialized)
        
    