# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""
import functools
from typing import Iterable, Optional, Tuple, Union, List
import os
import torch
from ..tensor.quantized_tensor import Quantizer
import transformer_engine_torch as tex
from ..constants import TE_DType
from ..utils import assert_dim_for_fp8_exec, get_sm_count
from ...debug.debug_quantization import DebugQuantizer
from ..tensor.quantized_tensor import QuantizedTensor
from ..tensor.float8_tensor import Float8Tensor, Float8TensorBase
from ..tensor.mxfp8_tensor import MXFP8Tensor, MXFP8TensorBase




from ..tensor.quantized_tensor import (
    QuantizedTensor,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)

from ..tensor._internal.mxfp8_tensor_base import MXFP8TensorBase

__all__ = [
    "general_gemm",
    "general_grouped_gemm",
]


@functools.lru_cache(maxsize=None)
def _empty_tensor() -> torch.Tensor:
    """Get tensor with no entries and no data"""
    return torch.Tensor().cuda()


def swizzle(tensor, scale_inv, rowwise):
    if scale_inv is None:
        return None

    swizzle_func = tex.rowwise_swizzle if rowwise else tex.columnwise_swizzle
    return swizzle_func(tensor, scale_inv)


def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    workspace: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
    ub_algo: tex.CommOverlapAlgo = None,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_buffer: Optional[torch.Tensor] = None,
) -> Iterable[Optional[torch.Tensor]]:
    """GEMM supporting fp8 inputs."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    # assert quantization_params is None, "FP8 output not supported yet"
    if out is not None:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    
    if isinstance(quantization_params, DebugQuantizer):
        if type(A) in [Float8TensorBase, Float8Tensor, MXFP8Tensor, MXFP8TensorBase] and out_dtype == torch.float32:
            if bias is not None:
                bias = bias.to(torch.bfloat16)
        else:
            if bias is not None:
                bias = bias.to(out_dtype)

    # Use bfloat16 as default bias_dtype
    bias_dtype = torch.bfloat16 if bias is None else bias.dtype
    bias_dtype = TE_DType[bias_dtype]
    if bias is None and not grad:
        bias = _empty_tensor()
    
    quantization_params_final = quantization_params
    if isinstance(quantization_params, DebugQuantizer):
        quantization_params_final = quantization_params._parent_quantizer

    args = (
        A,
        transa,  # transa
        B,
        transb,  # transb
        out,
        quantization_params_final,
        TE_DType[out_dtype] if out_dtype is not None else None,
        bias,
        bias_dtype,
        gelu,
        gelu_in,
        grad,  # grad
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )

    fn = tex.generic_gemm
    if ub_algo is not None:
        raise ValueError("Not implemented yet!")
        if ub_algo == tex.CommOverlapAlgo.BULK_OVERLAP_AG:
            fn = ub.bulk_overlap
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(
                args
                + (
                    tex.CommOverlapType.AG,
                    extra_output_tensor,
                )
            )
        elif ub_algo == tex.CommOverlapAlgo.BULK_OVERLAP_RS:
            fn = ub.bulk_overlap
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(
                args
                + (
                    tex.CommOverlapType.RS,
                    extra_output_tensor,
                )
            )
        elif ub_algo == tex.CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P:
            fn = ub.split_overlap_ag_p2p
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == tex.CommOverlapAlgo.ATOMIC_GEMM_AG_P2P:
            assert A_scaling_mode == [-1, -1, 1] and B_scaling_mode == [
                -1,
                -1,
                1,
            ], "Block scaling unsupported for atomic GEMM."
            fn = ub.atomic_gemm_overlap_ag_p2p
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == tex.CommOverlapAlgo.SPLIT_PIPELINED_RS:
            fn = ub.split_overlap_rs
            assert (
                extra_output_tensor is not None
            ), "SPLIT_PIPELINED_RS requires extra output tensor"
            args = tuple(
                args
                + (
                    True,
                    extra_output_tensor,
                )
            )
        elif ub_algo == tex.CommOverlapAlgo.SPLIT_PIPELINED_RS_P2P:
            fn = ub.split_overlap_rs_p2p
            assert (
                extra_output_tensor is not None
            ), "SPLIT_PIPELINED_RS_P2P requires extra output tensor"
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == tex.CommOverlapAlgo.ATOMIC_GEMM_RS:
            assert A_scaling_mode == [-1, -1, 1] and B_scaling_mode == [
                -1,
                -1,
                1,
            ], "Block scaling unsupported for atomic GEMM."
            fn = ub.atomic_gemm_overlap_rs
            assert extra_output_tensor is not None, "ATOMIC_GEMM_RS requires extra output tensor"
            args = tuple(
                args
                + (
                    True,
                    extra_output_tensor,
                )
            )
        elif ub_algo == tex.CommOverlapAlgo.ATOMIC_GEMM_RS_P2P:
            assert A_scaling_mode == [-1, -1, 1] and B_scaling_mode == [
                -1,
                -1,
                1,
            ], "Block scaling unsupported for atomic GEMM."
            fn = ub.atomic_gemm_overlap_rs_p2p
            assert (
                extra_output_tensor is not None
            ), "ATOMIC_GEMM_RS_P2P requires extra output tensor"
            args = tuple(args + (extra_output_tensor,))
    if ub_algo is not None and ub_algo == tex.CommOverlapAlgo.ATOMIC_GEMM_AG_P2P:
        out = fn(*args)
        gelu_input = None
        bias_grad = None
    else:
        if isinstance(A, MXFP8TensorBase) or isinstance(B, MXFP8TensorBase):
            tmp_scale_inverses = (
                A._rowwise_scale_inv,
                A._columnwise_scale_inv,
                B._rowwise_scale_inv,
                B._columnwise_scale_inv,
            )
            (
                A._rowwise_scale_inv,
                A._columnwise_scale_inv,
                B._rowwise_scale_inv,
                B._columnwise_scale_inv,
            ) = (
                swizzle(A._rowwise_data, A._rowwise_scale_inv, True),
                swizzle(A._columnwise_data, A._columnwise_scale_inv, False),
                swizzle(B._rowwise_data, B._rowwise_scale_inv, True),
                swizzle(B._columnwise_data, B._columnwise_scale_inv, False),
            )
        out, bias_grad, gelu_input = fn(*args)
        if isinstance(A, MXFP8TensorBase) or isinstance(B, MXFP8TensorBase):
            (
                A._rowwise_scale_inv,
                A._columnwise_scale_inv,
                B._rowwise_scale_inv,
                B._columnwise_scale_inv,
            ) = tmp_scale_inverses
    
    if quantization_params is not None:
        # used by debug quantizers for the hooks
        out = quantization_params.process_after_quantization(out)


    return out, bias_grad, gelu_input


def general_grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    out_dtype: torch.dtype,
    workspaces: List[torch.Tensor],
    layout: str = "TN",
    m_splits: Optional[List[int]] = None,
    gelu: bool = False,
    grad=False,
    accumulate: bool = False,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
    single_output=False,
) -> Tuple[List[torch.Tensor], ...]:
    """
    TN layout Grouped GEMM with fp8 inputs.
    """
    num_gemms = len(A)

    transa = layout[0] == "T"
    transb = layout[1] == "T"

    # assert [a.is_contiguous() for a in A]
    # assert [b.is_contiguous() for b in B]

    if isinstance(A[0], QuantizedTensor):
        for a, b in zip(A, B):
            assert_dim_for_fp8_exec(a._data)
            assert_dim_for_fp8_exec(b._data)

    empty_tensor = _empty_tensor()
    empty_tensors = [empty_tensor] * num_gemms

    # Use bfloat16 as default bias_dtype
    gelu_input = empty_tensors
    out_dtype = TE_DType[out[0].dtype] if D_dtype is None else D_dtype

    sm_count = get_sm_count()
    if grad and use_bias:
        grad_bias = [
            torch.empty(B[i].shape[1], dtype=out[0].dtype, device="cuda") for i in range(num_gemms)
        ]
    else:
        grad_bias = empty_tensors
    bias = bias if use_bias else empty_tensors
    if use_bias:
        bias_dtype = TE_DType[grad_bias[0].dtype] if grad else TE_DType[bias[0].dtype]
    else:
        bias_dtype = TE_DType[torch.bfloat16]

    if gelu:
        gelu_input = [
            torch.empty_like(o, dtype=bias_dtype, memory_format=torch.contiguous_format)
            for o in out
        ]  # this should differ with respect to single output

    bias = tex.te_general_grouped_gemm(
        A,
        transa,
        B,
        transb,
        out,
        out_dtype,
        m_splits,
        grad_bias if grad else bias,
        bias_dtype,
        single_output,
        gelu_input,  # this is pre_gelu_out
        grad,  # grad
        workspaces,
        workspaces[0].shape[0],
        accumulate,
        use_split_accumulator,
        sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count))),
    )

    return out, bias, gelu_input
