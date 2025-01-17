# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNormLinear API"""
import logging
from typing import Union, Optional, Callable, Tuple, List, Dict, Any

import torch

from .base import (
    TransformerEngineBaseModule,
)

from ....pytorch.module import LayerNorm, RMSNorm
from ....pytorch.constants import dist_group_type
from ....pytorch.jit import no_torch_dynamo
from .linear import Linear


try:
    import nvtorch_inspect.api as nvinspect_api
except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError("ERROR: Could not locate nvtorch_inspect package. Make sure it is installed correctly.")


__all__ = ["LayerNormLinear"]

class LayerNormLinear(TransformerEngineBaseModule):
    r"""
    Applies layer normalization followed by linear transformation to the incoming data.

    This debug module uses te.pytorch.LayerNorm or te.pytorch.RMSNorm and te.debug.pytorch.Linear. Logging and GEMMs are selected in Linear.
    Normalization is performed in higher precision and all casts occur in te.debug.pytorch.Linear.

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
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        normalization: str = 'LayerNorm',
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
        name: str = None,
    ) -> None:
        
        super().__init__()
        
        if (ub_bulk_wgrad or ub_bulk_dgrad or ub_overlap_ag or ub_overlap_rs_dgrad):
            nvinspect_api.log_message("> UserBuffers are not supported in debug module. "
                                    "Using UB optimization will not affect the debug module. ",
                                    level=logging.WARNING)
        self.name = name
        self.return_bias = return_bias
        self.return_layernorm_output = return_layernorm_output
        assert normalization in ['LayerNorm', 'RMSNorm'], "Unsupported normalization type!"

        if normalization == 'LayerNorm':
            normalization_cls = LayerNorm
        elif normalization == 'RMSNorm':
            normalization_cls = RMSNorm

        norm = normalization_cls(in_features,
                                    eps=eps,
                                    sequence_parallel=sequence_parallel,
                                    params_dtype=params_dtype,
                                    zero_centered_gamma=zero_centered_gamma,
                                    device=device)
        
        linear = Linear(in_features=in_features,
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
                            parameters_split=parameters_split,
                            device=device,
                            ub_overlap_rs=False,
                            ub_overlap_ag=False,
                            ub_name=ub_name,
                            name=name)
        
        self.layer_norm_weight = torch.nn.Parameter(norm.weight.data)
        norm.weight = self.layer_norm_weight

        if hasattr(norm, "bias"):
            self.layer_norm_bias = torch.nn.Parameter(norm.bias.data)
            norm.bias = self.layer_norm_bias

        # linear can have different parameter names like query_weight, key_weight etc.
        for name, param in linear.named_parameters():
            new_param = torch.nn.Parameter(param.data)
            setattr(self, name, new_param)
            setattr(linear, name, new_param)

        # fp8 metadata should be loaded into linear submodule
        self.get_extra_state = linear.get_extra_state
        self.set_extra_state = linear.set_extra_state

        # trick to not have any submodules like in the vanilia TE
        self.norm = [norm]
        self.linear = [linear]


    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """Abstract method. Not used."""
        return

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        overwrite_name: str = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        self._validate_name(overwrite_name)
        # (High Precision) -> (High Precision)
        ln_out = self.norm[0](inp)

        if self.return_bias:
            # (High Precision) -> (FP8/High Precision)
            out, bias_tensor = self.linear[0](ln_out,
                                            is_first_microbatch=is_first_microbatch, 
                                            is_first_module_in_mha=False,
                                            overwrite_name=self.name)
            if self.return_layernorm_output:
                return out, bias_tensor, ln_out
            return out, bias_tensor

        # (High Precision) -> (FP8/High Precision)
        out = self.linear[0](ln_out,
                            is_first_microbatch=is_first_microbatch, 
                            is_first_module_in_mha=False,
                            overwrite_name=self.name)
        if self.return_layernorm_output:
            return out, ln_out
        return out