# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
ONNX symbolic functions for Transformer Engine

Warnings of the type pasted below are a known Pytorch issue
(https://github.com/pytorch/pytorch/issues/81693):

tests/test_onnx_export.py::test_export_cast_ops[112]
  /opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py:649:
  UserWarning: The shape inference of trt::TRT_FP8DequantizeLinear type is missing,
  so it may result in wrong shape inference for the exported graph.
  Please consider adding it in symbolic function. (Triggered internally at
  /opt/pytorch/pytorch/torch/csrc/jit/passes/onnx/shape_type_inference.cpp:1880.)
    _C._jit_pass_onnx_graph_shape_type_inference(


Scale tensors are treated as lists ("fs") instead of tensors ("v") because we need to access
specific entries using the index passes as `fp8_tensor`. If you fail to do this you will get
the following error when accessing a sepcific scale element (e.g. `scale_inv[fp8_tensor]`):
    TypeError: 'torch._C.Value' object is not subscriptable
"""

import torch
from torch.onnx import symbolic_helper, register_custom_op_symbolic, _type_utils
import torch._C._onnx as _C_onnx

# Monkey-patch graph manipulation methods on Graph, used for the ONNX symbolics
from torch.onnx._internal import jit_utils

import transformer_engine
import transformer_engine_torch as tex

from .constants import TE_DType


# This file registers custom op symbolic ONNX functions and does not export any symbols.
__all__ = []


# Custom ops spec version
VER = 1
UNSPECIFIED_TYPE = -1



def _torch_dtype_to_int(dtype):
    te_dtype = TE_DType[dtype]
    return int(te_dtype)

def _quantizer(q):
    if q is None:
        return 0, None, None, 0, True, False
    if type(q) ==  transformer_engine.pytorch.tensor.float8_tensor.Float8Quantizer:
        q_type_int = 1
    return q_type_int, q.scale, q.amax, q.dtype, q.rowwise_usage, q.columnwise_usage

def _tensor(t):
    if t is None:
        return 0, None, None, None, 3, tex.DType.kFloat32,  *_quantizer(None)
    if type(t) ==  transformer_engine.pytorch.tensor.float8_tensor.Float8Tensor:
        return 1, t._data, t._transpose, t._scale_inv, t._fp8_dtype, _torch_dtype_to_int(t.dtype), *_quantizer(t._quantizer)
    # standard tensor
    return 0, t, None, None, 3, tex.DType.kFloat32,  *_quantizer(None)


def get_TensorProtoDataType(t):
    """Return the _C_onnx.TensorProtoDataType of the input tensor"""
    try:
        return {
            "Float": _C_onnx.TensorProtoDataType.FLOAT,
            "Half": _C_onnx.TensorProtoDataType.FLOAT16,
            "BFloat16": _C_onnx.TensorProtoDataType.BFLOAT16,
        }[t.type().scalarType()]
    except KeyError as e:
        raise TypeError(f"Onnx export for dtype {t.type().scalarType()} not supported.") from e


def is_dtype_fp32(t):
    """Check fp32 dtype"""
    return t.type().scalarType() == "Float"


def is_dtype_fp16(t):
    """Check fp16 dtype"""
    return t.type().scalarType() == "Half"


def is_dtype_bf16(t):
    """Check bf16 dtype"""
    return t.type().scalarType() == "BFloat16"

@torch.library.custom_op("tex_ts::crop", mutates_args=())
def crop(pic: torch.Tensor) -> torch.Tensor:
    return pic + 1

@crop.register_fake
def _(pic):
    return pic.new_empty(pic.shape)

@symbolic_helper.parse_args("v")
def crop_symbolic(g: jit_utils.GraphContext, input_tensor):
    """ONNX symbolic function for crop"""
    return g.op("Identity", input_tensor)

register_custom_op_symbolic("tex_ts::crop", crop_symbolic, VER)


# gets c args and returns te args
def te_parse_args(*expected_types):
    quantizer_types = ["i", "v", "v", "i", "b", "b"]
    type_to_onnx = {
        torch.Tensor: "v",
        "quantizer": quantizer_types,
        "tensor": ["i", "v", "v", "v", "i", "i", *quantizer_types],
        bool: "b",
        int: "i",
        float: "f",
    }
    onnx_c_args = [item for sublist in [type_to_onnx[t] for t in expected_types] for item in sublist]
    
    def create_object(arg_type, onnx_objects):
        if arg_type == torch.Tensor:
            return onnx_objects[0], onnx_objects[1:]
        elif arg_type == bool:
            return onnx_objects[0], onnx_objects[1:]
        elif arg_type == int:
            return onnx_objects[0], onnx_objects[1:]
        elif arg_type == float:
            return onnx_objects[0], onnx_objects[1:]
        elif arg_type == "quantizer":
            quantizer_id = onnx_objects[0]
            scale = onnx_objects[1]
            amax = onnx_objects[2]
            dtype = onnx_objects[3]
            rowwise = onnx_objects[4]
            columnwise = onnx_objects[5]
            if quantizer_id == 0:
                q =  None
            else:
                q = transformer_engine.pytorch.tensor.float8_tensor.Float8Quantizer(scale, amax, dtype, rowwise, columnwise)
            return q, onnx_objects[6:]
        elif arg_type == "tensor":
            tensor_id = onnx_objects[0]
            data = onnx_objects[1]
            transpose = onnx_objects[2]
            scale_inv = onnx_objects[3]
            dtype = onnx_objects[4]
            fake_dtype = onnx_objects[5]
            quantizer = onnx_objects[6]
            if tensor_id == 0:
                t = None
            else:
                t = transformer_engine.pytorch.tensor.float8_tensor.Float8Tensor(data, transpose, scale_inv, dtype, fake_dtype, quantizer)
            return t, onnx_objects[7:]


    def decorator(func):
        @symbolic_helper.parse_args(*onnx_c_args)
        def wrapper(*args):
            args = [create_object(expected_types[i], args) for i in range(len(expected_types))]
            return func(*args)
        return wrapper
    return decorator

def quantize(g, input_tensor, quantizer, out_tensor):
    """Helper Function for Quantization"""

    if quantizer is None:
        # return empty graph
        return g.op("Identity", input_tensor)
    else:
        return quantizer.onnx_graph(g, input_tensor, out_tensor)


def dequantize(g, inputs, scale_inv, fp8_tensor, otype):
    """Helper Function for Dequantization"""
    output_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inputs)

    scale = g.op("Constant", value_t=torch.tensor(scale_inv[fp8_tensor]))
    out = g.op("DequantizeLinear"), inputs, scale.setType(
        inputs.type().with_dtype(torch.float32).with_sizes(output_shape)
    )

    # DQ outputs are currently constrained to FP32 due to a similar limitation in ORT
    # custom ops, so cast the output if needed.
    if otype == int(tex.DType.kFloat16):
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.FLOAT16)
    elif otype == int(tex.DType.kBFloat16):
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.BFLOAT16)
    return out

@te_parse_args("tensor", "quantizer", "tensor")
def onnx_quantize(g, input_tensor, quantizer, out_tensor):
    return quantize(g, input_tensor, quantizer, out_tensor)

@te_parse_args("tensor", "quantizer")
def onnx_dequantize(g, input_tensor, quantizer):
    return dequantize(g, input_tensor, quantizer)


def compute_in_fp32(g, inp, subgraph, *args, **kwargs):
    """Wrap subgraph with casts to/from FP32 so that its precision is FP32.

    If `inp` data type is not FP32, add a cast of `inp` to FP32 and feed that into `subgraph`;
    then cast subgraphs's output back to `inp` data type.
    """
    inp_dtype = get_TensorProtoDataType(inp)
    is_fp32 = inp_dtype == _type_utils.JitScalarType.FLOAT
    if not is_fp32:
        inp = g.op("Cast", inp, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    sg_out = subgraph(g, inp, *args, **kwargs)
    if not is_fp32:
        sg_out = g.op("Cast", sg_out, to_i=inp_dtype)
    return sg_out


@te_parse_args("tensor", "quantizer")
def onnx_gelu(g, inp, quantizer):
    """ONNX graph for fp8_gelu"""
    # pylint: disable=unused-argument
    # TE computes GELU using float32 precision so wrap the GELU subgraph with
    # conversion to/from float32.
    dtype = get_TensorProtoDataType(inp)
    if dtype != _type_utils.JitScalarType.FLOAT:
        inp = g.op("Cast", inp, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    out = torch.onnx.symbolic_opset9.gelu(g, inp, "tanh")
    if quantizer is not None:
        out = quantize(g, out, quantizer, None)
    elif dtype != _type_utils.JitScalarType.FLOAT:
        out = g.op("Cast", out, to_i=dtype)
    return out


@te_parse_args("tensor", "quantizer")
def onnx_relu(g, inp, quantizer):
    """ONNX graph for fp8_relu"""
    # pylint: disable=unused-argument
    out = torch.onnx.symbolic_opset9.relu(g, inp)
    if quantizer is not None:
        out = quantize(g, out, quantizer, None)
    return out


@te_parse_args("tensor", "quantizer")
def onnx_swiglu(g: jit_utils.GraphContext, inp, quantizer):
    """ONNX graph for swiglu"""
    if quantizer is not None:
        dtype = get_TensorProtoDataType(inp)
        if dtype != _type_utils.JitScalarType.FLOAT:
            inp = g.op("Cast", inp, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    dim = 1

    # Check dimensions
    dim_size = symbolic_helper._get_tensor_dim_size(inp, dim)
    if dim_size is not None:
        assert dim_size % 2 == 0

    # Perform compute in FP32
    dtype = get_TensorProtoDataType(inp)
    if dtype != _type_utils.JitScalarType.FLOAT:
        inp = g.op("Cast", inp, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    first, second = g.op("Split", inp, axis_i=dim, outputs=2)
    out = g.op("Mul", g.op("Sigmoid", first), second)
    if dtype != _type_utils.JitScalarType.FLOAT:
        out = g.op("Cast", out, to_i=dtype)
    
    if quantizer is not None:
        out = quantize(g, out, quantizer, None)
    elif dtype != _type_utils.JitScalarType.FLOAT:
        out = g.op("Cast", out, to_i=dtype)
    return out


@te_parse_args("tensor", "quantizer") 
def onnx_reglu(g: jit_utils.GraphContext, inp, quantizer):
    """ONNX graph for reglu"""
    if quantizer is not None:
        dtype = get_TensorProtoDataType(inp)
        if dtype != _type_utils.JitScalarType.FLOAT:
            inp = g.op("Cast", inp, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    dim = 1

    # Check dimensions
    dim_size = symbolic_helper._get_tensor_dim_size(inp, dim)
    if dim_size is not None:
        assert dim_size % 2 == 0

    # Perform compute in FP32
    dtype = get_TensorProtoDataType(inp)
    if dtype != _type_utils.JitScalarType.FLOAT:
        inp = g.op("Cast", inp, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    first, second = g.op("Split", inp, axis_i=dim, outputs=2)
    out = g.op("Mul", g.op("Relu", first), second)
    if dtype != _type_utils.JitScalarType.FLOAT:
        out = g.op("Cast", out, to_i=dtype)
    
    if quantizer is not None:
        out = quantize(g, out, quantizer, None)
    elif dtype != _type_utils.JitScalarType.FLOAT:
        out = g.op("Cast", out, to_i=dtype)
    return out


@te_parse_args("tensor", "quantizer")
def onnx_geglu(g: jit_utils.GraphContext, inp, quantizer):
    """ONNX graph for geglu"""
    if quantizer is not None:
        dtype = get_TensorProtoDataType(inp)
        if dtype != _type_utils.JitScalarType.FLOAT:
            inp = g.op("Cast", inp, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    dim = 1

    # Check dimensions
    dim_size = symbolic_helper._get_tensor_dim_size(inp, dim)
    if dim_size is not None:
        assert dim_size % 2 == 0

    # Perform compute in FP32
    dtype = get_TensorProtoDataType(inp)
    if dtype != _type_utils.JitScalarType.FLOAT:
        inp = g.op("Cast", inp, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    first, second = g.op("Split", inp, axis_i=dim, outputs=2)
    first = torch.onnx.symbolic_opset9.gelu(g, first, "tanh")
    out = g.op("Mul", first, second)
    if dtype != _type_utils.JitScalarType.FLOAT:
        out = g.op("Cast", out, to_i=dtype)
    
    if quantizer is not None:
        out = quantize(g, out, quantizer, None)
    elif dtype != _type_utils.JitScalarType.FLOAT:
        out = g.op("Cast", out, to_i=dtype)
    return out


@te_parse_args("tensor", bool, "tensor", bool, "tensor", "quantizer", torch.Tensor,
                 int, torch.Tensor, bool, torch.Tensor, int, bool, bool)
def onnx_te_gemm(
    g,
    weight,
    trans_weight,
    inputs,
    trans_input,
    out,
    quantization_params,
    bias,
    bias_type,
    pre_gelu_out,
    grad,
    workspace,
    workspaceSize,
    accumulate,
    use_split_accumulator,
):
    """ONNX graph for te_gemm"""
    # pylint: disable=unused-argument
    is_fp16 = is_dtype_fp16(inputs)
    is_bf16 = is_dtype_bf16(inputs)

    # tutaj trzeba bedzie dodac gemm w fp8 niestety
    if isinstance(inputs, transformer_engine.pytorch.tensor.QuantizedTensor):
        inputs = dequantize(g, inputs, inputs.get_quantizer())

    if isinstance(weight, transformer_engine.pytorch.tensor.QuantizedTensor):
        weight = dequantize(g, weight, weight.get_quantizer())

    empty_tensor_size = [0]
    bias_empty = torch.onnx.symbolic_helper._get_tensor_sizes(bias) == empty_tensor_size
    pre_gelu_out_empty = (
        torch.onnx.symbolic_helper._get_tensor_sizes(pre_gelu_out) == empty_tensor_size
    )

    if not bias_empty:
        output = g.op("Gemm", inputs, weight, bias, transA_i=trans_input, transB_i=trans_weight)
    else:
        output = g.op("Gemm", inputs, weight, transA_i=trans_input, transB_i=trans_weight)
    if not bias_empty:
        if not pre_gelu_out_empty:
            # TE computes GELU using float32 precision so wrap the GELU subgraph with
            # conversion to/from float32.
            output = compute_in_fp32(g, output, torch.onnx.symbolic_opset9.gelu, "tanh")
    else:
        if is_fp16:
            output = g.op("Cast", output, to_i=_C_onnx.TensorProtoDataType.FLOAT16)
        elif is_bf16:
            output = g.op("Cast", output, to_i=_C_onnx.TensorProtoDataType.BFLOAT16)
    return output


def _ones_like(g, inp, dtype):
    """Returns a tensor filled with the scalar value 1, with the same size as input and
    with dtype data-type"""
    shape = g.op("Shape", inp)
    # WAR ONNX spec: ConstantOfShape accepts all data types except for BF16. To WAR
    # create a ConstantOfShape with type FP32 and then add a Cast to BF16.
    is_bf16 = dtype == torch.bfloat16
    one = g.op(
        "ConstantOfShape",
        shape,
        value_t=torch.tensor([1], dtype=torch.float32 if is_bf16 else dtype),
    )
    if is_bf16:
        one = g.op("Cast", one, to_i=_C_onnx.TensorProtoDataType.BFLOAT16)
    return one



@te_parse_args(torch.Tensor, torch.Tensor, torch.Tensor, float, "quantizer", "tensor", int, int)
def onnx_layernorm_fwd(g, inputs, weight, bias, eps, quantizer, out, sm_margin, zero_centered_gamma):
    """ONNX graph for layernorm_fwd"""
    # pylint: disable=unused-argument
    inp_dtype = get_TensorProtoDataType(inputs)

    if inp_dtype != get_TensorProtoDataType(weight):
        weight = g.op("Cast", weight, to_i=inp_dtype)
    if inp_dtype != get_TensorProtoDataType(bias):
        bias = g.op("Cast", bias, to_i=inp_dtype)

    normalized_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inputs)
    if normalized_shape is None:
        ndim = torch.onnx.symbolic_helper._get_tensor_rank(inputs)
        assert ndim is not None
        normalized_shape = list(range(0, ndim))
    # Normalization axis = 0, so normalized_shape uses all dims except dim = 0
    normalized_shape = normalized_shape[1:]

    if zero_centered_gamma:
        inputs_dtype = inputs.type().dtype()
        one = _ones_like(g, weight, inputs_dtype)
        weight = g.op("Add", weight, one)

    axis = -len(normalized_shape)
    ln = g.op(
        "LayerNormalization",
        inputs,
        weight,
        bias,
        epsilon_f=eps,
        axis_i=axis,
        # This sets the LN compute precision - use FP32 always as does TE.
        stash_type_i=_C_onnx.TensorProtoDataType.FLOAT,
    )
    if quantizer is not None:
        ln = quantize(g, ln, quantizer, out)
    return ln

@te_parse_args(torch.Tensor, torch.Tensor, float, "quantizer", "tensor", int, int)
def onnx_rmsnorm_fwd(g, inp, weight, eps, quantizer, out, sm_margin, zero_centered_gamma):
    """ONNX graph for rmsnorm_fwd"""
    # pylint: disable=unused-argument
    dtype = get_TensorProtoDataType(inp)
    if dtype != _type_utils.JitScalarType.FLOAT:
        inp = g.op("Cast", inp, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    # Check dimensions
    normalized_shape = torch.onnx.symbolic_helper._get_tensor_sizes(inp)
    if normalized_shape is None:
        ndim = torch.onnx.symbolic_helper._get_tensor_rank(inp)
        assert ndim is not None
        normalized_shape = list(range(0, ndim))
    # Normalization axis = 0, so normalized_shape uses all dims except dim = 0
    normalized_shape = normalized_shape[1:]
    axis = -len(normalized_shape)

    # Cast input tensors to FP32 if needed
    dtype = get_TensorProtoDataType(inp)
    if dtype != _type_utils.JitScalarType.FLOAT:
        inp = g.op("Cast", inp, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    if get_TensorProtoDataType(weight) != _type_utils.JitScalarType.FLOAT:
        weight = g.op("Cast", weight, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    # Adjust zero-centered weights
    if zero_centered_gamma:
        one = _ones_like(g, weight, torch.float32)
        weight = g.op("Add", weight, one)

    # Perform compute in FP32
    sum_square = g.op("ReduceSumSquare", inp, axes_i=[axis])
    shape = g.op("Shape", inp, start_i=-1)
    shape_f = g.op("Cast", shape, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    mean_squared = g.op("Div", sum_square, shape_f)
    eps_tensor = g.op("ConstantOfShape", shape, value_t=torch.tensor([eps], dtype=torch.float32))
    rms_squared = g.op("Add", mean_squared, eps_tensor)
    rms_eps = g.op("Sqrt", rms_squared)
    normalized_input = g.op("Div", inp, rms_eps)
    out = g.op("Mul", weight, normalized_input)
    if quantizer is not None:
        out = quantize(g, out, quantizer, out)
    elif dtype != _type_utils.JitScalarType.FLOAT:
        out = g.op("Cast", out, to_i=dtype)
    return out

register_custom_op_symbolic("tex_ts::quantize", quantize, VER)
register_custom_op_symbolic("tex_ts::dequantize", dequantize, VER)
register_custom_op_symbolic("tex_ts::gelu_ts", onnx_gelu, VER)
register_custom_op_symbolic("tex_ts::relu_ts", onnx_relu, VER)
register_custom_op_symbolic("tex_ts::reglu_ts", onnx_reglu, VER)
register_custom_op_symbolic("tex_ts::geglu_ts", onnx_geglu, VER)
register_custom_op_symbolic("tex_ts::swiglu_ts", onnx_swiglu, VER)
register_custom_op_symbolic("tex_ts::te_gemm_ts", onnx_te_gemm, VER)
register_custom_op_symbolic("tex_ts::layernorm_fwd_ts", onnx_layernorm_fwd, VER)
register_custom_op_symbolic("tex_ts::rmsnorm_ts", onnx_rmsnorm_fwd, VER)
