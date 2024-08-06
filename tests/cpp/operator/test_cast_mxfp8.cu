/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <limits>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

constexpr int exponent_bias = 127;

template <typename OType, typename CType>
int compute_shared_unbiased_exponent(const CType amax) {
    if (amax == 0.0f) {
        return 0;
    }
    const int exponent = floorf(log2f(amax))
                         - floorf(log2f(Quantized_Limits<OType>::max()));
    
    const int exponent_clamped = (exponent < -127) ? -127 : exponent;
    return exponent_clamped;
}

template <typename InputType, typename OutputType>
void process_block(const InputType* data,
                   OutputType* output_c,
                   byte* output_scales,
                   const size_t scale_idx,
                   const size_t i_min,
                   const size_t i_max,
                   const size_t j_min,
                   const size_t j_max,
                   const size_t cols) {
    using ComputeType = float;
    ComputeType amax = 0.0f;

    // Find the absolute maximum value in the block
    for (size_t i = i_min; i < i_max; ++i) {
        for (size_t j = j_min; j < j_max; ++j) {
            const size_t idx = i * cols + j;
            const ComputeType elt = static_cast<ComputeType>(data[idx]);
            if (isinf(elt) || isnan(elt)) {
                continue;
            }
            amax = std::max(amax, std::abs(elt));
        }
    }

    const int unbiased_exponent = compute_shared_unbiased_exponent<OutputType>(amax);
    output_scales[scale_idx] = static_cast<byte>(unbiased_exponent + exponent_bias);
    const ComputeType scale_reciprocal = powf(2.0f, -unbiased_exponent);

    // Quantize elements in the block
    for (size_t i = i_min; i < i_max; ++i) {
        for (size_t j = j_min; j < j_max; ++j) {
            const size_t idx = i * cols + j;
            const ComputeType elt = static_cast<ComputeType>(data[idx]);
            output_c[idx] = static_cast<OutputType>(elt * scale_reciprocal);
        }
    }
}

template <typename InputType, typename OutputType>
void compute_ref(const InputType* data,
                 OutputType* output_c,
                 byte* output_scales,
                 const size_t rows,
                 const size_t cols,
                 const size_t block_size_Y,
                 const size_t block_size_X) {
    using ComputeType = float;

    const size_t blocks_Y = (rows + block_size_Y - 1) / block_size_Y;
    const size_t blocks_X = (cols + block_size_X - 1) / block_size_X;

    for (size_t ii = 0; ii < blocks_Y; ++ii) {
        const size_t i_min = ii * block_size_Y;
        const size_t i_max = std::min((ii + 1) * block_size_Y, rows);
        for (size_t jj = 0; jj < blocks_X; ++jj) {
            const size_t j_min = jj * block_size_X;
            const size_t j_max = std::min((jj + 1) * block_size_X, cols);
            const size_t scale_idx = ii * blocks_X + jj;
            process_block(data, output_c, output_scales, scale_idx, i_min, i_max, j_min, j_max, cols);
        }
    }
}

template <typename InputType, typename OutputType>
void performTest(const size_t rows,
                 const size_t cols,
                 const size_t block_size_rows,
                 const size_t block_size_cols,
                 InputsFillCase fill_case) {
    using namespace test;
    using EncodingType = fp32;
    DType itype = TypeInfo<InputType>::dtype;
    DType otype = TypeInfo<OutputType>::dtype;
    DType scale_type = TypeInfo<byte>::dtype;       // E8M0

    const size_t blocks_Y = (rows + block_size_rows - 1) / block_size_rows;
    const size_t blocks_X = (cols + block_size_cols - 1) / block_size_cols;

    Tensor input({ rows, cols }, itype);
    Tensor output_c({ rows, cols }, otype);
    Tensor output_scales({ blocks_Y, blocks_X }, scale_type,
                         {static_cast<int>(block_size_cols), static_cast<int>(block_size_rows), 0});

    std::unique_ptr<OutputType[]> ref_output_c = std::make_unique<OutputType[]>(rows * cols);
    std::unique_ptr<byte[]> ref_output_scales = std::make_unique<byte[]>(blocks_Y * blocks_X);

    fillCase<EncodingType>(&input, fill_case);
    nvte_fp8_quantize(input.data(), output_c.data(), 0, output_scales.data());

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    compute_ref<InputType, OutputType>(input.cpu_dptr<InputType>(),
                                       ref_output_c.get(),
                                       ref_output_scales.get(),
                                       rows,
                                       cols,
                                       block_size_rows,
                                       block_size_cols);

    auto [atol, rtol] = getTolerances(otype);
    compareResults("output_c", output_c, ref_output_c.get(), atol, rtol);
    compareResults("scales", output_scales, ref_output_scales.get(), atol, rtol);
}

std::vector<std::pair<size_t, size_t>> matrix_sizes = {
    {256, 256},
    {768, 1024},
    {256, 65536},
    {2048, 12288},
    {65536, 128},
    {16384, 6144},
};

std::vector<std::pair<size_t, size_t>> block_sizes = {
    {1, 32},
    // {1, 64},
    // {1, 128},
    // {32, 32},
    // {64, 64},
    // {128, 128}
};

std::vector<InputsFillCase> input_scenarios = {
    InputsFillCase::uniform,
    InputsFillCase::zeros,
    InputsFillCase::zero_to_minNorm,
    InputsFillCase::minNorm_to_maxNorm,
    InputsFillCase::maxNorm_to_inf
};

}  // namespace

class CastMXFP8TestSuite : public ::testing::TestWithParam<std::tuple<std::pair<size_t, size_t>,
                                                                      std::pair<size_t, size_t>,
                                                                      transformer_engine::DType,
                                                                      transformer_engine::DType,
                                                                      InputsFillCase>> {};

TEST_P(CastMXFP8TestSuite, TestCastMXFP8) {
    using namespace transformer_engine;
    using namespace test;

    const auto matrix_size = std::get<0>(GetParam());
    const auto block_size = std::get<1>(GetParam());
    const DType input_type = std::get<2>(GetParam());
    const DType output_type = std::get<3>(GetParam());
    const InputsFillCase fill_case = std::get<4>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OutputType,
            performTest<InputType, OutputType>(matrix_size.first, matrix_size.second,
                                               block_size.first, block_size.second, fill_case);
        );
    );
}


INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    CastMXFP8TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(matrix_sizes),
        ::testing::ValuesIn(block_sizes),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        ::testing::ValuesIn(input_scenarios)),
    [](const testing::TestParamInfo<CastMXFP8TestSuite::ParamType>& info) {
        std::string name = std::to_string(std::get<0>(info.param).first) + "X" +
                           std::to_string(std::get<0>(info.param).second) + "X" +
                           std::to_string(std::get<1>(info.param).first) + "X" +
                           std::to_string(std::get<1>(info.param).second) + "X" +
                           test::typeName(std::get<2>(info.param)) + "X" +
                           test::typeName(std::get<3>(info.param)) + "X" +
                           test::caseName(std::get<4>(info.param));
        return name;
    });
