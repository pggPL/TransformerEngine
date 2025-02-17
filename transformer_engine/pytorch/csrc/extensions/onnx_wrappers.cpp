/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "pybind.h"
#include "extensions.h"
#include <string>

#define PYBIND11_DETAILED_ERROR_MESSAGE
#include <torch/script.h>
#include <pybind11/embed.h>
#define TE_ONNX_ARG_QUANTIZER(quantizer) \
    int64_t quantizer##_id, \
    at::Tensor quantizer##_scale, \
    at::Tensor quantizer##_amax, \
    int64_t quantizer##_fp8_dtype, \
    bool quantizer##_rowwise, \
    bool quantizer##_columnwise

#define TE_ONNX_ARG_TENSOR(tensor) \
    int64_t tensor##_id, \
    at::Tensor tensor##_data, \
    at::Tensor tensor##_transpose, \
    at::Tensor tensor##_scaling_factors, \
    int64_t tensor##_fp8_dtype, \
    int64_t tensor##_fake_dtype, \
    TE_ONNX_ARG_QUANTIZER(tensor##_quantizer)

#define TE_ONNX_QUANTIZER(quantizer) \
    get_quantizer(quantizer##_id, \
                 quantizer##_scale, \
                 quantizer##_amax, \
                 quantizer##_fp8_dtype, \
                 quantizer##_rowwise, \
                 quantizer##_columnwise)

#define TE_ONNX_QUANTIZER_1(quantizer) \
    quantizer##_id, \
    quantizer##_scale, \
    quantizer##_amax, \
    quantizer##_fp8_dtype, \
    quantizer##_rowwise, \
    quantizer##_columnwise

#define TE_ONNX_INPUT_TENSOR(tensor) \
    get_tensor(tensor##_id, \
              tensor##_data, \
              tensor##_transpose, \
              tensor##_scaling_factors, \
              tensor##_fp8_dtype, \
              tensor##_fake_dtype, \
              TE_ONNX_QUANTIZER_1(tensor##_quantizer))

#define TE_ONNX_GET_DATA_TENSORS(output) get_data_tensors(output)

namespace {
py::module float8tensor_module;
py::module quantization_module;
bool modules_initialized = false;

void init_modules() {
    if (modules_initialized) return;
    try {
        // Ensure Python is initialized
        if (!Py_IsInitialized()) {
            pybind11::initialize_interpreter();
        }
        
        // Import modules within a valid Python context
        py::gil_scoped_acquire gil;
        float8tensor_module = py::module::import("transformer_engine.pytorch.tensor.float8_tensor");
        quantization_module = py::module::import("transformer_engine.pytorch.tensor");
    } catch (const py::error_already_set& e) {
        NVTE_ERROR("Failed to import required Python modules: %s", e.what());
    }
    modules_initialized = true;
}

transformer_engine::DType reverse_map_dtype(int64_t dtype) {
  if (dtype >= 0 && dtype < static_cast<int64_t>(transformer_engine::DType::kNumTypes)) {
    return static_cast<transformer_engine::DType>(dtype);
  } else {
    NVTE_ERROR("Type not supported.");
  }
}
}

py::object get_quantizer(int64_t id, at::Tensor scale, at::Tensor amax, int64_t fp8_dtype, bool rowwise, bool columnwise) {
    if(id == 0) {
        return py::none();
    }
    if(id == 1) {
        py::gil_scoped_acquire gil;  // Acquire GIL before creating Python objects
        // dtype
        auto fp8_dtype_arg = reverse_map_dtype(fp8_dtype);
        return py::object(float8tensor_module.attr("Float8Quantizer")(
            scale,
            amax,
            fp8_dtype_arg,
            py::arg("rowwise") = rowwise,
            py::arg("columnwise") = columnwise
        ));
    }
    throw std::invalid_argument("Invalid quantizer id");
}

py::object get_tensor(int64_t id, std::optional<at::Tensor> data, 
                      std::optional<at::Tensor> data_transpose, at::Tensor fp8_scale_inv, 
                      int64_t fp8_dtype, int64_t fake_dtype, TE_ONNX_ARG_QUANTIZER(quantizer)) {
    // if data is none
    if (data == std::nullopt && data_transpose == std::nullopt) {
        return py::none();
    }
    py::object quantizer_obj = TE_ONNX_QUANTIZER(quantizer);
    if(id == 0) {
        return py::cast(data);
    }
    if(id == 1) {
        py::gil_scoped_acquire gil;
        py::dict kwargs;
        kwargs["shape"] = data->sizes();
        kwargs["dtype"] = transformer_engine::pytorch::GetATenDType(reverse_map_dtype(fake_dtype));
        kwargs["data"] = data.value();
        kwargs["fp8_scale_inv"] = fp8_scale_inv;
        kwargs["fp8_dtype"] = reverse_map_dtype(fp8_dtype);
        kwargs["requires_grad"] = false;
        kwargs["quantizer"] = quantizer_obj;
        
        // Only add data_transpose if it exists
        if (data_transpose.has_value()) {
            kwargs["data_transpose"] = data_transpose.value();
        }

        auto x = float8tensor_module.attr("Float8Tensor")(**kwargs);
        return x;
    }
    throw std::invalid_argument("Invalid tensor id");
}
std::vector<at::Tensor> get_data_tensors(py::object output) {
    py::gil_scoped_acquire gil;
    if (!py::hasattr(output, "get_data_tensors")) {
        if (output.is_none()) {
            return std::vector<at::Tensor>{at::Tensor()}; // Return tensor with None
        }
        return std::vector<at::Tensor>{output.cast<at::Tensor>()};
    }
    py::object result = output.attr("get_data_tensors")();
    
    // Handle both tuple and list returns
    if (py::isinstance<py::tuple>(result) || py::isinstance<py::list>(result)) {
        std::vector<at::Tensor> tensors;
        for (const auto& item : result) {
            if (item.is_none()) {
                tensors.push_back(at::Tensor()); // Add tensor with None
            } else {
                tensors.push_back(item.cast<at::Tensor>());
            }
        }
        return tensors;
    }
    
    // Single tensor case
    if (result.is_none()) {
        return std::vector<at::Tensor>{at::Tensor()}; // Return tensor with None
    }
    return std::vector<at::Tensor>{result.cast<at::Tensor>()};
}

std::vector<at::Tensor> quantize(at::Tensor tensor, TE_ONNX_ARG_QUANTIZER(quantizer), TE_ONNX_ARG_TENSOR(out)) {
    init_modules();
    py::gil_scoped_acquire gil;
    py::object quantizer_obj = TE_ONNX_QUANTIZER(quantizer);
    py::object out_obj = TE_ONNX_INPUT_TENSOR(out);
    std::cout << "quantize" << std::endl;
    std::cout << "tensor" << tensor << std::endl;
    std::cout << "quantizer_obj" <<  quantizer_obj << std::endl;
    std::cout << "out_obj" <<  out_obj << std::endl;
    py::object output = transformer_engine::pytorch::quantize(tensor, quantizer_obj, out_obj, std::nullopt);
    std::cout << "output" << output << std::endl;
    std::vector<at::Tensor> out = TE_ONNX_GET_DATA_TENSORS(output);
    return out;
}

at::Tensor dequantize(TE_ONNX_ARG_TENSOR(input), int64_t otype) {
    init_modules();
    py::gil_scoped_acquire gil;
    py::object input_obj = TE_ONNX_INPUT_TENSOR(input); 
    transformer_engine::DType otype_arg = reverse_map_dtype(otype);
    py::object out = transformer_engine::pytorch::dequantize(input_obj, otype_arg);
    return out.cast<at::Tensor>();
}

template<typename Func>
std::vector<at::Tensor> activation_wrapper(const at::Tensor& input, 
                                         TE_ONNX_ARG_QUANTIZER(quantizer),
                                         Func&& activation_func) {
    init_modules();
    py::gil_scoped_acquire gil;
    py::object quantizer_obj = TE_ONNX_QUANTIZER(quantizer);
    py::object out = activation_func(input, quantizer_obj);
    return TE_ONNX_GET_DATA_TENSORS(out);
}

std::vector<at::Tensor> gelu_ts(const at::Tensor& input, TE_ONNX_ARG_QUANTIZER(quantizer)) {
    return activation_wrapper(input, TE_ONNX_QUANTIZER_1(quantizer), 
                            transformer_engine::pytorch::gelu);
}

std::vector<at::Tensor> relu_ts(const at::Tensor& input, TE_ONNX_ARG_QUANTIZER(quantizer)) {
    return activation_wrapper(input, TE_ONNX_QUANTIZER_1(quantizer), 
                            transformer_engine::pytorch::relu);
}

std::vector<at::Tensor> geglu_ts(const at::Tensor& input, TE_ONNX_ARG_QUANTIZER(quantizer)) {
    return activation_wrapper(input, TE_ONNX_QUANTIZER_1(quantizer), 
                            transformer_engine::pytorch::geglu);
}

std::vector<at::Tensor> reglu_ts(const at::Tensor& input, TE_ONNX_ARG_QUANTIZER(quantizer)) {
    return activation_wrapper(input, TE_ONNX_QUANTIZER_1(quantizer), 
                            transformer_engine::pytorch::reglu);
}

std::vector<at::Tensor> swiglu_ts(const at::Tensor& input, TE_ONNX_ARG_QUANTIZER(quantizer)) {
    return activation_wrapper(input, TE_ONNX_QUANTIZER_1(quantizer), 
                            transformer_engine::pytorch::swiglu);
}

std::vector<at::Tensor> qgelu_ts(const at::Tensor& input, TE_ONNX_ARG_QUANTIZER(quantizer)) {
    return activation_wrapper(input, TE_ONNX_QUANTIZER_1(quantizer), 
                            transformer_engine::pytorch::qgelu);
}

std::vector<at::Tensor> qgeglu_ts(const at::Tensor& input, TE_ONNX_ARG_QUANTIZER(quantizer)) {
    return activation_wrapper(input, TE_ONNX_QUANTIZER_1(quantizer),
                            transformer_engine::pytorch::qgeglu);
}

std::vector<at::Tensor> srelu_ts(const at::Tensor& input, TE_ONNX_ARG_QUANTIZER(quantizer)) {
    return activation_wrapper(input, TE_ONNX_QUANTIZER_1(quantizer), 
                            transformer_engine::pytorch::srelu);
}


std::vector<at::Tensor> general_gemm_ts(TE_ONNX_ARG_TENSOR(A), bool transa, TE_ONNX_ARG_TENSOR(B), bool transb, TE_ONNX_ARG_TENSOR(D),
                             TE_ONNX_ARG_QUANTIZER(quantizer), std::optional<int64_t> out_dtype, MaybeTensor bias, int64_t bias_type, 
                             bool gelu, MaybeTensor gelu_in, bool grad, at::Tensor workspace, int64_t workspaceSize, bool accumulate, bool use_split_accumulator) {
    init_modules();
    py::gil_scoped_acquire gil;
    py::object quantizer_obj = TE_ONNX_QUANTIZER(quantizer);
    py::object A_obj = TE_ONNX_INPUT_TENSOR(A);
    py::object B_obj = TE_ONNX_INPUT_TENSOR(B);
    py::object D_obj = TE_ONNX_INPUT_TENSOR(D);
    auto out_dtype_arg = out_dtype ? std::make_optional(reverse_map_dtype(*out_dtype)) : std::nullopt;
    DType bias_type_arg = reverse_map_dtype(bias_type);

    std::vector<py::object> output = transformer_engine::pytorch::gemm(A_obj, transa, B_obj, transb, D_obj, quantizer_obj,
        out_dtype_arg, bias, bias_type_arg, gelu, gelu_in, grad, workspace, workspaceSize,
        accumulate, false, nullptr, std::nullopt, std::nullopt, use_split_accumulator);
    return TE_ONNX_GET_DATA_TENSORS(output[0]);
}

std::vector<at::Tensor> layernorm_fwd_ts(TE_ONNX_ARG_TENSOR(input), TE_ONNX_ARG_TENSOR(weight), MaybeTensor bias,
                                      double eps, TE_ONNX_ARG_TENSOR(ln_out), TE_ONNX_ARG_QUANTIZER(quantizer),
                                      int64_t out_dtype, const int64_t sm_margin,
                                      const bool zero_centered_gamma) {
    init_modules();
    py::gil_scoped_acquire gil;
    float eps_float = static_cast<float>(eps);
    py::object input_obj = TE_ONNX_INPUT_TENSOR(input);
    py::object weight_obj = TE_ONNX_INPUT_TENSOR(weight);
    py::object ln_out_obj = TE_ONNX_INPUT_TENSOR(ln_out);
    py::object quantizer_obj = TE_ONNX_QUANTIZER(quantizer);
    DType out_dtype_arg = reverse_map_dtype(out_dtype);
    std::vector<py::object> output = layernorm_fwd(input_obj, weight_obj, bias, eps_float, ln_out_obj, quantizer_obj,
                                                  out_dtype_arg, sm_margin, zero_centered_gamma);
    return TE_ONNX_GET_DATA_TENSORS(output[0]);
}

std::vector<at::Tensor> rmsnorm_fwd_ts(TE_ONNX_ARG_TENSOR(input), TE_ONNX_ARG_TENSOR(weight),
                                      double eps, TE_ONNX_ARG_TENSOR(ln_out), TE_ONNX_ARG_QUANTIZER(quantizer),
                                      int64_t out_dtype, const int64_t sm_margin,
                                      const bool zero_centered_gamma) {
    init_modules();
    py::gil_scoped_acquire gil;
    float eps_float = static_cast<float>(eps);
    py::object input_obj = TE_ONNX_INPUT_TENSOR(input);
    py::object weight_obj = TE_ONNX_INPUT_TENSOR(weight);
    py::object ln_out_obj = TE_ONNX_INPUT_TENSOR(ln_out);
    py::object quantizer_obj = TE_ONNX_QUANTIZER(quantizer);
    DType out_dtype_arg = reverse_map_dtype(out_dtype);
    std::vector<py::object> output = rmsnorm_fwd(input_obj, weight_obj, eps_float, ln_out_obj, quantizer_obj,
                                                out_dtype_arg, sm_margin, zero_centered_gamma);
    return TE_ONNX_GET_DATA_TENSORS(output[0]);
}

TORCH_LIBRARY(tex_ts, m) {
    m.def("quantize", &quantize);
    m.def("dequantize", &dequantize); 
    m.def("gelu_ts", &gelu_ts);
    m.def("relu_ts", &relu_ts);
    m.def("geglu_ts", &geglu_ts);
    m.def("reglu_ts", &reglu_ts);
    m.def("swiglu_ts", &swiglu_ts);
    m.def("qgelu_ts", &qgelu_ts);
    m.def("qgeglu_ts", &qgeglu_ts);
    m.def("srelu_ts", &srelu_ts);
    m.def("general_gemm_ts", &general_gemm_ts);
    m.def("layernorm_fwd_ts", &layernorm_fwd_ts);
    m.def("rmsnorm_fwd_ts", &rmsnorm_fwd_ts);
}
