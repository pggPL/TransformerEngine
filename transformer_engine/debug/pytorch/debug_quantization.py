# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
    This file contains DebugQuantizer and DebugQuantizedTensor objects, 
    which are wrappers over Quantizer and QuantizedTensor. 
    These wrapper add logic related to the debugging, using the nvdlfw_inspect package.
"""

from __future__ import annotations
from typing import Optional, Tuple, Iterable, Any, Dict, List
import torch

aten = torch.ops.aten

from ...pytorch.tensor.quantized_tensor import (
    QuantizedTensor,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from transformer_engine.debug.debug_state import TEDebugState
import transformer_engine_torch as tex


import nvdlfw_inspect.api as nvinspect_api

_gemm_map = {
    "weight": ["fprop", "dgrad"],
    "activation": ["fprop", "wgrad"],
    "output": ["fprop", None],
    "gradient": ["dgrad", "wgrad"],
    "wgrad": ["wgrad", None],
    "dgrad": ["dgrad", None],
}


class DebugQuantizer(Quantizer):
    """
        aa
        bb
        cc
    """
    def __init__(self, layer_name, tensor_name, parent_quantizer, tp_group):
        super().__init__(rowwise=True, columnwise=True)
        self.layer_name = layer_name
        self.tensor_name = tensor_name
        self.parent_quantizer = parent_quantizer
        self.tp_group = tp_group  # used in look_at_tensor calls
        iteration = nvinspect_api.DEBUG_MANAGER._trainer_iteration_count

        self.first_gemm_name, self.second_gemm_name = _gemm_map[tensor_name]

        self.output_tensor = tensor_name in ["output", "wgrad", "dgrad"]
        if self.output_tensor:
            self.use_look_at_tensor_before_process = (
                nvinspect_api.transformer_engine.use_look_at_tensor_before_process(
                    layer_name=self.layer_name, tensor_name=self.tensor_name, iteration=iteration
                )
            )
            self.output_process_tensor = nvinspect_api.transformer_engine.use_process_tensor(
                layer_name=self.layer_name,
                gemm=self.first_gemm_name,
                tensor_name=self.tensor_name,
                iteration=iteration,
            )
            return  # logic for tensors which are output of the gemms is much simpler

        # If gemm in conducted in the high precision,
        # then it is checked whether
        # process_tensor() hook is defined. If it is the case,
        # tensor for that gemm will be computed using process_tensor().
        #
        # Then it is checked whether
        # gemms, that use the tensor, will be conducted in
        # fp8 or in high precision. If in fp8, then
        # standard quantizer is used.
        #
        # If none of the above is True,
        # then high-precision tensor without any modifications will be used.
        self.process_tensor_first_gemm = False
        self.process_tensor_second_gemm = False
        self.fp8_quantize_second_gemm = False
        self.fp8_quantize_first_gemm = False
        self.use_look_at_tensor_before_process = (
            nvinspect_api.transformer_engine.use_look_at_tensor_before_process(
                layer_name=self.layer_name, tensor_name=self.tensor_name, iteration=iteration
            )
        )
        self.use_look_at_tensor_after_process = (
            nvinspect_api.transformer_engine.use_look_at_tensor_after_process(
                layer_name=self.layer_name, tensor_name=self.tensor_name, iteration=iteration
            )
        )

        self.process_tensor_first_gemm = nvinspect_api.transformer_engine.use_process_tensor(
            layer_name=self.layer_name,
            gemm=self.first_gemm_name,
            tensor_name=self.tensor_name,
            iteration=iteration,
        )
        if not self.process_tensor_first_gemm:
            if self.parent_quantizer is not None:
                self.fp8_quantize_first_gemm = nvinspect_api.transformer_engine.fp8_gemm(
                    layer_name=self.layer_name, gemm=self.first_gemm_name, iteration=iteration
                )

        if self.second_gemm_name is not None:
            self.process_tensor_second_gemm = nvinspect_api.transformer_engine.use_process_tensor(
                layer_name=self.layer_name,
                gemm=self.second_gemm_name,
                tensor_name=self.tensor_name,
                iteration=iteration,
            )
            if not self.process_tensor_second_gemm:
                if self.parent_quantizer is not None:
                    self.fp8_quantize_second_gemm = nvinspect_api.transformer_engine.fp8_gemm(
                        layer_name=self.layer_name, gemm=self.second_gemm_name, iteration=iteration
                    )
        self.log_messages()


    def log_messages(self):
        # Information of the invoked API will be posted here
        if self.process_tensor_first_gemm:
            nvinspect_api.log_message(
                f"Tensor: {self.tensor_name}, gemm {self.first_gemm_name} - process_tensor",
                layer_name=self.layer_name,
                extra_cachable_args=(self.first_gemm_name, self.tensor_name),
            )
        elif self.fp8_quantize_first_gemm:
            nvinspect_api.log_message(
                f"Tensor: {self.tensor_name}, gemm {self.first_gemm_name} - FP8 quanitation",
                layer_name=self.layer_name,
                extra_cachable_args=(self.first_gemm_name, self.tensor_name),
            )
        else:
            nvinspect_api.log_message(
                f"Tensor: {self.tensor_name}, gemm {self.first_gemm_name} - High precision",
                layer_name=self.layer_name,
                extra_cachable_args=(self.first_gemm_name, self.tensor_name),
            )

        if self.process_tensor_second_gemm:
            nvinspect_api.log_message(
                f"Tensor: {self.tensor_name}, gemm {self.second_gemm_name} - process_tensor",
                layer_name=self.layer_name,
                extra_cachable_args=(self.second_gemm_name, self.tensor_name),
            )
        elif self.fp8_quantize_second_gemm:
            nvinspect_api.log_message(
                f"Tensor: {self.tensor_name}, gemm {self.second_gemm_name} - FP8 quanitation",
                layer_name=self.layer_name,
                extra_cachable_args=(self.first_gemm_name, self.tensor_name),
            )
        else:
            nvinspect_api.log_message(
                f"Tensor: {self.tensor_name}, gemm {self.second_gemm_name} - High precision",
                layer_name=self.layer_name,
                extra_cachable_args=(self.second_gemm_name, self.tensor_name),
            )

    def _call_look_at_tensor_api(self, tensor, first_gemm_tensor=None, second_gemm_tensor=None):
        args = {
            "layer_name": self.layer_name,
            "tensor": tensor,
            "tensor_name": self.tensor_name,
            "iteration": nvinspect_api.DEBUG_MANAGER._trainer_iteration_count,
            "tp_group": self.tp_group
        }
        if tensor is not None:
            nvinspect_api.transformer_engine.look_at_tensor_before_process(**args)

        if self.output_tensor:
            return

        if self.fp8_quantize_first_gemm or self.process_tensor_first_gemm:
            args["tensor"] = first_gemm_tensor
            args["rowwise"] = True
            nvinspect_api.transformer_engine.look_at_tensor_after_process(**args)
        elif self.fp8_quantize_second_gemm or self.process_tensor_second_gemm:
            args["tensor"] = second_gemm_tensor
            args["rowwise"] = False
            nvinspect_api.transformer_engine.look_at_tensor_after_process(**args)

    def quantize(self, tensor, *, out=None, dtype=None):
        assert not self.output_tensor
        if out is not None:
            return self.update_quantized(tensor, self)

        iteration = nvinspect_api.DEBUG_MANAGER._trainer_iteration_count

        # 1. If there is fp8 quantization in at least one of the gemms,
        #    the quantization using the self.parent_quantizer is performed.

        # first gemm corresponds to the rowwise_usage in fp8, similarly with columnwise
        first_gemm_quantize = self.rowwise_usage and self.fp8_quantize_first_gemm
        second_gemm_quantize = self.columnwise_usage and self.fp8_quantize_second_gemm
        if second_gemm_quantize and not first_gemm_quantize:
            first_gemm_quantize = True  # only second_gemm not implemented

        first_gemm_tensor = None
        second_gemm_tensor = None

        if self.fp8_quantize_second_gemm or self.fp8_quantize_first_gemm:
            self.parent_quantizer.set_usage(
                rowwise=first_gemm_quantize, columnwise=second_gemm_quantize
            )
            quantized_tensor = self.parent_quantizer(tensor)
            # if both first_gemm_tensor and second_gemm_tensor need to be in fp8,
            # one tensor with columnwise=True and rowwise=True is computed
            # and both first_gemm_tensor and second_gemm_tensor point to it.
            if self.rowwise_usage and self.fp8_quantize_first_gemm:
                first_gemm_tensor = quantized_tensor
            if self.columnwise_usage and self.fp8_quantize_second_gemm:
                second_gemm_tensor = quantized_tensor

        # 2. Process_tensor() is called, if it is used.
        if self.process_tensor_second_gemm:
            second_gemm_tensor = nvinspect_api.transformer_engine.process_tensor(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                gemm=self.second_gemm_name,
                tensor=tensor,
                default_quantizer=self.parent_quantizer,
                iteration=iteration,
                dtype=dtype,
            )
        if self.process_tensor_first_gemm:
            first_gemm_tensor = nvinspect_api.transformer_engine.process_tensor(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                gemm=self.first_gemm_name,
                tensor=tensor,
                default_quantizer=self.parent_quantizer,
                iteration=iteration,
                dtype=dtype,
            )

        # 3. If some tensors still are not defined we use input tensor.
        if first_gemm_tensor is None:
            first_gemm_tensor = tensor.to(dtype)
        if second_gemm_tensor is None:
            second_gemm_tensor = tensor.to(dtype)

        self._call_look_at_tensor_api(tensor, first_gemm_tensor, second_gemm_tensor)

        # sometimes we may want to return simple tensor with only first_gemm
        if self.tensor_name in ["wgrad", "dgrad", "output"]:
            return first_gemm_tensor

        return DebugQuantizedTensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            first_gemm_tensor=first_gemm_tensor,
            second_gemm_tensor=second_gemm_tensor,
            quantizer=self,
            layer_name=self.layer_name,
            tensor_name=self.tensor_name,
        )

    def process_gemm_output(self, tensor):
        # This call is invoked after the gemm to process output tensor and save the stats.
        assert self.parent_quantizer is None, "FP8 output is not supported for debug=True."
        assert self.output_tensor
        tensor_to_gemm = {"output": "fprop", "wgrad": "wgrad", "dgrad": "dgrad"}
        if self.output_process_tensor:
            tensor = nvinspect_api.transformer_engine.process_tensor(
                layer_name=self.layer_name,
                gemm=tensor_to_gemm[self.tensor_name],
                tensor_name=self.tensor_name,
                tensor=tensor,
                iteration=nvinspect_api.DEBUG_MANAGER._trainer_iteration_count,
            )
        self._call_look_at_tensor_api(tensor)
        return tensor

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> QuantizedTensor:
        if self.parent_quantizer is not None:
            return self.parent_quantizer(shape, dtype=dtype, device=device)
        else:
            return torch.empty(shape, dtype=dtype, device=device)

    def calibrate(self, tensor: torch.Tensor):
        raise RuntimeError("Calibration with debug=True is not supported")

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        assert noop_flag is None, "CUDA Graphs are not supported with debug=True!"
        iteration = nvinspect_api.DEBUG_MANAGER._trainer_iteration_count
        updated_second_gemm = False
        updated_first_gemm = False
        if self.parent_quantizer is not None:
            if dst.first_gemm_tensor is not None and self.fp8_quantize_first_gemm:
                if hasattr(dst.first_gemm_tensor, "quantize_"):
                    dst.first_gemm_tensor.quantize_(src, noop_flag=None)
                else:
                    tex.quantize(src, self.parent_quantizer, dst.first_gemm_tensor, None)
                updated_first_gemm = True
            if dst.second_gemm_tensor is not None and self.fp8_quantize_second_gemm:
                if hasattr(dst.second_gemm_tensor, "quantize_"):
                    dst.second_gemm_tensor.quantize_(src, noop_flag=None)
                else:
                    tex.quantize(src, self.parent_quantizer, dst.second_gemm_tensor, None)
                updated_second_gemm = True

        if self.process_tensor_second_gemm:
            out = nvinspect_api.transformer_engine.process_tensor(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                gemm=self.second_gemm_name,
                tensor=src,
                default_quantizer=self.parent_quantizer,
                out=dst.second_gemm_tensor,
                iteration=iteration,
            )
            assert out is None, (
                "API call nvinspect_api.transformer_engine.process_tensor with out != None should"
                " return None"
            )
            updated_second_gemm = True
        if self.process_tensor_first_gemm:
            nvinspect_api.transformer_engine.process_tensor(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                gemm=self.first_gemm_name,
                tensor=src,
                default_quantizer=self.parent_quantizer,
                out=dst.first_gemm_tensor,
                iteration=iteration,
            )
            updated_first_gemm = True
        if not updated_second_gemm:
            dst.second_gemm_tensor.copy_(src)
        if updated_second_gemm and not updated_first_gemm:
            dst.first_gemm_tensor.copy_(src)
            # if updated_first_gemm and updated_second_gemm, then
            # dst.second_gemm and dst.first_gemm. is the same tensor,
            # and it is already updated.

    def is_fp8(self):
        return self.parent_quantizer is not None

    def use_any_feature(self):
        if self.output_tensor:
            return self.use_look_at_tensor_before_process or self.output_process_tensor
        if (
            self.use_look_at_tensor_before_process
            or self.use_look_at_tensor_after_process
            or self.process_tensor_first_gemm
            or self.process_tensor_second_gemm
        ):
            return True
        if self.parent_quantizer is not None:
            if not self.fp8_quantize_first_gemm:
                return True
            if not self.fp8_quantize_second_gemm:
                return True
        return False


class DebugQuantizedTensor(QuantizedTensor):
    def __new__(
        cls,
        shape,
        dtype,
        first_gemm_tensor,
        second_gemm_tensor,
        quantizer,
        requires_grad=False,
        layer_name=None,
        tensor_name=None,
    ):
        instance = super().__new__(cls, shape, dtype, requires_grad=requires_grad)

        instance.first_gemm_tensor = first_gemm_tensor
        instance.second_gemm_tensor = second_gemm_tensor
        instance.quantizer = quantizer
        instance._layer_name = layer_name
        instance._tensor_name = tensor_name

        return instance

    def prepare_for_saving(self):
        tensor_list, tensor_objects_list = prepare_for_saving(
            self.first_gemm_tensor, self.second_gemm_tensor
        )
        self.first_gemm_tensor, self.second_gemm_tensor = tensor_objects_list
        return tensor_list, self

    def restore_from_saved(self, tensors):
        (self.first_gemm_tensor, self.second_gemm_tensor), saved_tensors = restore_from_saved(
            [self.first_gemm_tensor, self.second_gemm_tensor], tensors, return_saved_tensors=True
        )
        return saved_tensors

    def quantize_(self, tensor, *, noop_flag=None):
        assert noop_flag is None, "CUDA Graphs are not supported with debug=True!"
        self.quantizer.update_quantized(tensor, self)

    def dequantize(self, *, dtype=None):
        if dtype is None:
            dtype = self.first_gemm_tensor.dtype
        return self.first_gemm_tensor.dequantize().to(dtype)

    def get_tensor(self, transpose: bool):
        # Is used in the python gemm() to get tensor or transpose of the tensor.
        return self.first_gemm_tensor if not transpose else self.second_gemm_tensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func in [aten.slice.Tensor]:
            tensor = args[0]
            first_gemm_tensor = tensor.first_gemm_tensor.__torch_dispatch__(
                func,
                types,
                [first_gemm_tensor] + list(args[1:]),
                kwargs,
            )
            return DebugQuantizedTensor(
                shape=first_gemm_tensor.shape,
                dtype=tensor.dtype,
                first_gemm_tensor=first_gemm_tensor,
                second_gemm_tensor=None,
                quantizer=tensor.quantizer,
                requires_grad=tensor.requires_grad,
                layer_name=tensor._layer_name,
                tensor_name=tensor._tensor_name,
            )
        
        return super().__torch_dispatch__(func, types, args, kwargs)


def use_any_feature(quantizers: List[DebugQuantizer]):
    for q in quantizers:
        if q.use_any_feature():
            return True
    return False
