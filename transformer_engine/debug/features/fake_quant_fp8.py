# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

from transformer_engine.debug.features.api import TEConfigAPIMapper
import nvdlfw_inspect.api as nvinspect_api
from nvdlfw_inspect.registry import Registry, api_method
from nvdlfw_inspect.utils import append_parent_docstring


import transformer_engine_torch as tex
from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.fp8 import _default_sf_compute
from transformer_engine.pytorch.constants import TE_DType


def fake_quantize_fp8(tensor: torch.Tensor, fp8_format, margin=0, out=None):
    assert tensor.dtype in (
        torch.float,
        torch.float16,
        torch.bfloat16,
    ), "[NVTORCH INSPECT ERROR] Unsupported tensor type."
    assert tensor.is_cuda, "[NVTORCH INSPECT ERROR] Must be a GPU tensor."
    assert fp8_format in {
        "E4M3",
        "E5M2",
        "MXE4M3",
        "MXE5M2",
    }, "[NVTORCH INSPECT ERROR] Only 4 FP8 types: E4M3, E5M2, MXE4M3, MXE5M2 are supported in TE."
    if fp8_format in ["E4M3", "E5M2"]:
        if fp8_format == "E4M3":
            fp8_max = Format.E4M3.value.max_fwd
            fp8_dtype = tex.DType.kFloat8E4M3
        else:
            fp8_max = Format.E5M2.value.max_fwd
            fp8_dtype = tex.DType.kFloat8E5M2
        amax = tensor.abs().max().float()
        one = torch.ones(1, device=tensor.device)
        scale = _default_sf_compute(amax, one, fp8_max, margin)

        quantizer = Float8Quantizer(scale, amax, fp8_dtype)
    else:
        MXFP8Quantizer(fp8_dtype=fp8_format)
    if out is None:
        return quantizer(tensor).dequantize()
    else:
        out.copy_(quantizer(tensor).dequantize())


@Registry.register_feature(namespace="transformer_engine")
@append_parent_docstring(parent=TEConfigAPIMapper)
class FakeQuantFp8(TEConfigAPIMapper):
    """
    Fake Quantization feature in Transformer engine.

    Fake quantization in this case refers to casting a tensor to FP8 and back to original dtype.

    Config:
    To enable the feature in yaml config:
    transformer_engine:
      fake_quant_fp8:
        enabled: True
        ...

    Config fields:
    This feature works at a tensor level, you can set the following properties for each tensor:
    - quant_format: Dictionary containing tensor names to FP8 formats. Options: {'E4M3', 'E5M2', 'MXE4M3', 'MXE5M2'}
    - margin: int, default is 0
    - tensors/tensors_struct: tensors list or tensors_struct - please look into the Transformer Engine Precision Debug Tools documentation for more information.

    """

    def _supported_formats(self):
        return ["E4M3", "E5M2", "MXE4M3", "MXE5M2"]

    def _get_margin_default(self):
        return 0

    @api_method
    def fp8_gemm(self, config, layer_name, gemm, iteration):
        return False

    @api_method
    def use_process_tensor(self, config, layer_name, tensor_name, gemm, iteration):
        return True

    @api_method
    def process_tensor(
        self,
        config,
        layer_name,
        gemm,
        tensor_name,
        tensor,
        iteration,
        default_quantizer=None,
        out=None,
        dtype=None,
    ):
        for key in config.keys():
            if key not in ["gemm", "tensor", "quant_format", "margin"]:
                raise ValueError(f'[NVTORCH INSPECT ERROR] Unexpected key in config: "{key}".')

        if "quant_format" not in config:
            raise ValueError(
                f"[NVTORCH INSPECT ERROR] Feature={self.__class__.__name__}, API=process_tensor:"
                f" quant_format missing for Tensor: {tensor_name} in the config yaml for"
                " FakeQuantFp8 feature which is a required field"
            )
        if config["quant_format"] not in self._supported_formats():
            raise ValueError(
                f"[NVTORCH INSPECT ERROR] Feature={self.__class__.__name__}, API=process_tensor:"
                f" quant_format: {config['quant_format']} for Tensor: {tensor_name} in the config"
                " yaml for FakeQuantFp8 feature is not supported"
            )
        nvinspect_api.log_message(
            f"Feature={self.__class__.__name__}, API=process_tensor: {gemm}, {tensor_name}",
            layer_name,
            extra_cachable_args=(gemm, tensor_name),
        )

        quant_format = config["quant_format"]
        margin = config.get("margin", self._get_margin_default())
        q_tensor = fake_quantize_fp8(tensor, quant_format, margin=margin, out=out)
        if dtype is not None:
            q_tensor = q_tensor.to(dtype)
        return q_tensor
