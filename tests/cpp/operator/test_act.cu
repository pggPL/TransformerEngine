/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <limits>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/activation.h>
#include "../test_common.h"

using namespace transformer_engine;

template <float (*act)(const float), typename IT, typename OT, typename CT>
void compute_ref_act_cast(const IT *input_h,
                          OT *output_h,
                          const CT scale,
                          CT *amax_h,
                          const size_t N,
                          const size_t H) {
  CT amax  = 0.;

  #pragma omp parallel for schedule(static) reduction(max: amax) proc_bind(spread)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT elt = static_cast<CT>(input_h[i * H + j]);
      elt = act(elt);
      output_h[i * H + j] = static_cast<OT>(scale * elt);
      amax = std::abs(elt) > amax ? std::abs(elt) : amax;
    }
  }

  *amax_h = amax;
}

template <float (*dact)(const float), typename IT, typename OT>
void compute_ref_dact_cast(const IT *input_h,
                           const IT *grad_h,
                           OT *output_h,
                           const size_t N,
                           const size_t H) {
  using CT = float;
  #pragma omp parallel for schedule(static) proc_bind(spread)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT elt = static_cast<CT>(input_h[i * H + j]);
      elt = dact(elt);
      CT grad = static_cast<CT>(grad_h[i * H + j]);
      output_h[i * H + j] = static_cast<OT>(grad * elt);
    }
  }
}

template <float (*act)(const float), typename IT, typename OT, typename CT>
void compute_ref_glu_act_cast(const IT *input_h, OT *output_h, const CT scale, CT *amax_h,
                              const size_t N, const size_t H) {
  CT amax = 0.;

  const int col = H * 2;

  #pragma omp parallel for schedule(static) reduction(max: amax) proc_bind(spread)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT gelu_elt = static_cast<CT>(input_h[i * col + j]);
      gelu_elt = act(gelu_elt);
      CT gate_elt = static_cast<CT>(input_h[i * col + H + j]);
      CT elt = gelu_elt * gate_elt;
      output_h[i * H + j] = static_cast<OT>(scale * elt);
      amax = std::abs(elt) > amax ? std::abs(elt) : amax;
    }
  }

  *amax_h = amax;
}

template <float (*dact)(const float), float (*act)(const float),
          typename IT, typename OT>
void compute_ref_dglu_act_cast(const IT *input_h, const IT *grad_h, OT *output_h,
                               const size_t N, const size_t H) {
  const int col = H * 2;
  using CT = float;

  #pragma omp parallel for schedule(static) proc_bind(spread)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT grad = static_cast<CT>(grad_h[i * H + j]);
      CT gelu_elt = static_cast<CT>(input_h[i * col + j]);
      CT gate_elt = static_cast<CT>(input_h[i * col + H + j]);
      output_h[i * col + H + j] = static_cast<OT>(grad * act(gelu_elt));
      gelu_elt = dact(gelu_elt);
      CT elt = gelu_elt * gate_elt;
      output_h[i * col + j] = static_cast<OT>(grad * elt);
    }
  }
}


template <float (*ref_act)(const float),
          float (*ref_dact)(const float),
          void (*nvte_act)(const NVTETensor, NVTETensor, cudaStream_t),
          void (*nvte_dact)(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t),
         typename IType, typename OType>
void performTest(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  Tensor input("input", std::vector<size_t>{ N, H }, itype);
  Tensor output("output", std::vector<size_t>{ N, H }, otype);
  Tensor igrad("igrad", std::vector<size_t>{ N, H }, itype);
  Tensor ograd("ograd", std::vector<size_t>{ N, H }, itype);

  fillUniform(&input);
  fillUniform(&ograd);
  setRandomScale(&output);
  const float ref_scale = isFp8Type(otype) ? output.scale() : 1.0f;

  std::unique_ptr<OType[]> ref_output = std::make_unique<OType[]>(N*H);
  std::unique_ptr<IType[]> ref_igrad = std::make_unique<IType[]>(N*H);

  nvte_act(input.data(), output.data(), 0);

  float ref_amax;
  compute_ref_act_cast<ref_act>(input.rowwise_cpu_dptr<IType>(), ref_output.get(),
                                ref_scale, &ref_amax, N, H);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  if (otype == DType::kFloat8E4M3 || otype == DType::kFloat8E5M2) {
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("amax", output.amax(), ref_amax, atol_amax, rtol_amax);
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_act", output, ref_output.get(), atol, rtol);

  nvte_dact(ograd.data(), input.data(), igrad.data(), 0);

  compute_ref_dact_cast<ref_dact>(input.rowwise_cpu_dptr<IType>(), ograd.rowwise_cpu_dptr<IType>(),
                                  ref_igrad.get(), N, H);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  {
    auto [atol, rtol] = getTolerances(otype);
    compareResults("igrad_act", igrad, ref_igrad.get(), atol, rtol);
  }
}

template <float (*ref_act)(const float),
          float (*ref_dact)(const float),
          void (*nvte_act)(const NVTETensor, NVTETensor, cudaStream_t),
          void (*nvte_dact)(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t),
         typename IType, typename OType>
void performTestGLU(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  Tensor input("input", std::vector<size_t>{N, H * 2}, itype);
  Tensor output("output", std::vector<size_t>{N, H}, otype);
  Tensor igrad("igrad", std::vector<size_t>{ N, H * 2 }, itype);
  Tensor ograd("ograd", std::vector<size_t>{ N, H }, itype);

  fillUniform(&input);
  fillUniform(&ograd);
  setRandomScale(&output);
  const float ref_scale = isFp8Type(otype) ? output.scale() : 1.0f;

  std::unique_ptr<OType[]> ref_output = std::make_unique<OType[]>(N * H);
  std::unique_ptr<IType[]> ref_igrad = std::make_unique<IType[]>(2 * N * H);

  nvte_act(input.data(), output.data(), 0);

  float ref_amax;
  compute_ref_glu_act_cast<ref_act>(input.rowwise_cpu_dptr<IType>(), ref_output.get(),
                                    ref_scale, &ref_amax, N, H);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  if (otype == DType::kFloat8E4M3 || otype == DType::kFloat8E5M2) {
    auto [atol, rtol] = getTolerances(DType::kFloat32);
    compareResults("amax", output.amax(), ref_amax, atol, rtol);
    if (output.scaling_mode() == NVTE_DELAYED_TENSOR_SCALING) {
      const float ref_scale_inv = 1.f / ref_scale;
      compareResults("scale_inv", *output.rowwise_cpu_scale_inv_ptr<float>(), ref_scale_inv, atol, rtol);
    }
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_gelu", output, ref_output.get(), atol, rtol);

  nvte_dact(ograd.data(), input.data(), igrad.data(), 0);

  compute_ref_dglu_act_cast<ref_dact, ref_act>(input.rowwise_cpu_dptr<IType>(), ograd.rowwise_cpu_dptr<IType>(),
                                               ref_igrad.get(), N, H);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  {
    auto [atol, rtol] = getTolerances(otype);
    compareResults("igrad_act", igrad, ref_igrad.get(), atol, rtol);
  }
}


class ActTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                                transformer_engine::DType,
                                                                std::pair<size_t, size_t>>> {};

TEST_P(ActTestSuite, TestGELU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<gelu, dgelu, nvte_gelu, nvte_dgelu,
                    InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestSILU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<silu, dsilu, nvte_silu, nvte_dsilu,
                    InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestRELU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<relu, drelu, nvte_relu, nvte_drelu,
                    InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestQGELU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<qgelu, dqgelu, nvte_qgelu, nvte_dqgelu,
                    InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestSRELU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<srelu, dsrelu, nvte_srelu, nvte_dsrelu,
                    InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestGeGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<gelu, dgelu, nvte_geglu, nvte_dgeglu, InputType,
                         OutputType>(size.first, size.second);););
}

TEST_P(ActTestSuite, TestReGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<relu, drelu, nvte_reglu, nvte_dreglu, InputType,
                         OutputType>(size.first, size.second);););
}

TEST_P(ActTestSuite, TestSwiGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<silu, dsilu, nvte_swiglu, nvte_dswiglu, InputType,
                         OutputType>(size.first, size.second);););
}

TEST_P(ActTestSuite, TestQGeGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<qgelu, dqgelu, nvte_qgeglu, nvte_dqgeglu, InputType,
                         OutputType>(size.first, size.second);););
}

TEST_P(ActTestSuite, TestSReGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<srelu, dsrelu, nvte_sreglu, nvte_dsreglu, InputType,
                         OutputType>(size.first, size.second);););
}

namespace {

std::vector<std::pair<size_t, size_t>> act_test_cases = {{2048, 12288},
                                                         {768, 2816},
                                                         {256, 65536},
                                                         {65536, 128},
                                                         {256, 256},
                                                         {257, 259},
                                                         {128, 128+1}};

std::string test_name_generator(
    const testing::TestParamInfo<ActTestSuite::ParamType>& info) {
    std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                       test::typeName(std::get<1>(info.param)) + "X" +
                       std::to_string(std::get<2>(info.param).first) + "X" +
                       std::to_string(std::get<2>(info.param).second);
    return name;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    OperatorTest_ActTestSuite_BF16,
    ActTestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kBFloat16),
        ::testing::ValuesIn(test::all_fp_types),
        ::testing::ValuesIn(act_test_cases)),
    test_name_generator);

INSTANTIATE_TEST_SUITE_P(
    OperatorTest_ActTestSuite_DType,
    ActTestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::ValuesIn(test::all_fp_types),
        ::testing::Values(std::pair<size_t, size_t>{768, 2816})),
    test_name_generator);

namespace {

template <typename T>
void test_grouped_situglu(const size_t hidden, const int alignment_mode, const int count_mode,
                          const std::pair<float, float> caps) {
  constexpr size_t rows = 13;
  const size_t input_offset = alignment_mode == 1 ? 1 : 0;
  const size_t output_offset = alignment_mode == 0 ? 0 : 1;
  const size_t active = count_mode == 0 ? 8 : (count_mode == 1 ? rows : 0);
  const DType dtype = test::TypeInfo<T>::dtype;
  const std::vector<size_t> input_shape{rows, 2 * hidden}, output_shape{rows, hidden};
  test::Tensor input("group_situ_input", std::vector<size_t>{rows * 2 * hidden + input_offset + 1},
                     dtype);
  test::Tensor grad("group_situ_grad", std::vector<size_t>{rows * hidden + input_offset + 1},
                    dtype);
  test::Tensor output("group_situ_output", std::vector<size_t>{rows * hidden + output_offset + 1},
                      dtype);
  test::Tensor dx("group_situ_dx", std::vector<size_t>{rows * 2 * hidden + output_offset + 1},
                  dtype);
  test::Tensor counts("group_situ_counts", std::vector<size_t>{3}, DType::kInt64);
  for (test::Tensor *tensor : {&input, &grad, &output, &dx}) {
    std::fill_n(tensor->rowwise_cpu_dptr<T>(), test::product(tensor->shape()),
                static_cast<T>(std::numeric_limits<float>::quiet_NaN()));
  }
  auto *x = input.rowwise_cpu_dptr<T>() + input_offset;
  auto *dy = grad.rowwise_cpu_dptr<T>() + input_offset;
  for (size_t row = 0; row < active; ++row) {
    for (size_t col = 0; col < hidden; ++col) {
      x[row * 2 * hidden + col] = static_cast<T>(static_cast<float>(col % 11) - 5.25f);
      x[row * 2 * hidden + hidden + col] = static_cast<T>(static_cast<float>(row) - 4.5f);
      dy[row * hidden + col] = static_cast<T>(0.125f * static_cast<float>(col % 7) - 0.25f);
    }
  }
  auto *sizes = counts.rowwise_cpu_dptr<int64_t>();
  sizes[0] = count_mode == 2 ? 0 : 3;
  sizes[1] = 0;
  sizes[2] = count_mode == 2 ? 0 : 5;
  for (test::Tensor *tensor : {&input, &grad, &output, &dx, &counts}) tensor->from_cpu();

  GroupedTensorWrapper in_group(3, input_shape), grad_group(3, output_shape);
  GroupedTensorWrapper out_group(3, output_shape), dx_group(3, input_shape);
  in_group.set_rowwise_data(static_cast<T *>(input.rowwise_dptr()) + input_offset, dtype,
                            input_shape);
  grad_group.set_rowwise_data(static_cast<T *>(grad.rowwise_dptr()) + input_offset, dtype,
                              output_shape);
  out_group.set_rowwise_data(static_cast<T *>(output.rowwise_dptr()) + output_offset, dtype,
                             output_shape);
  dx_group.set_rowwise_data(static_cast<T *>(dx.rowwise_dptr()) + output_offset, dtype,
                            input_shape);
  if (count_mode != 1) {
    in_group.set_first_dims(counts.rowwise_dptr(), DType::kInt64, std::vector<size_t>{3});
  }
  nvte_group_situglu(in_group.data(), out_group.data(), caps.first, caps.second, nullptr);
  nvte_group_dsituglu(grad_group.data(), in_group.data(), dx_group.data(), caps.first, caps.second,
                      nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  output.to_cpu();
  dx.to_cpu();
  const auto *y = output.rowwise_cpu_dptr<T>() + output_offset;
  const auto *actual_dx = dx.rowwise_cpu_dptr<T>() + output_offset;
  const double tolerance =
      dtype == DType::kFloat32 ? 2e-5 : (dtype == DType::kFloat16 ? .003 : .02);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < hidden; ++col) {
      double expected_y = 0, expected_gate = 0, expected_up = 0;
      if (row < active) {
        const double gate = static_cast<float>(x[row * 2 * hidden + col]);
        const double up = static_cast<float>(x[row * 2 * hidden + hidden + col]);
        const double upstream = static_cast<float>(dy[row * hidden + col]);
        const double sigmoid = 1 / (1 + std::exp(-gate));
        const double tg = std::tanh(gate / caps.first), tu = std::tanh(up / caps.second);
        expected_y = caps.first * tg * sigmoid * caps.second * tu;
        expected_gate = upstream *
                        ((1 - tg * tg) * sigmoid + caps.first * tg * sigmoid * (1 - sigmoid)) *
                        caps.second * tu;
        expected_up = upstream * caps.first * tg * sigmoid * (1 - tu * tu);
      }
      EXPECT_NEAR(static_cast<float>(y[row * hidden + col]), expected_y,
                  tolerance * (1 + std::abs(expected_y)));
      EXPECT_NEAR(static_cast<float>(actual_dx[row * 2 * hidden + col]), expected_gate,
                  tolerance * (1 + std::abs(expected_gate)));
      EXPECT_NEAR(static_cast<float>(actual_dx[row * 2 * hidden + hidden + col]), expected_up,
                  tolerance * (1 + std::abs(expected_up)));
    }
  }
  for (test::Tensor *tensor : {&output, &dx}) {
    const auto *data = tensor->rowwise_cpu_dptr<T>();
    if (output_offset) EXPECT_TRUE(std::isnan(static_cast<float>(data[0])));
    EXPECT_TRUE(std::isnan(static_cast<float>(data[test::product(tensor->shape()) - 1])));
  }
}

class GroupedSiTUGLUTest : public ::testing::TestWithParam<
                               std::tuple<DType, size_t, int, int, std::pair<float, float>>> {};

TEST_P(GroupedSiTUGLUTest, AlignmentAndInactiveRows) {
  const auto [dtype, hidden, alignment_mode, count_mode, caps] = GetParam();
  switch (dtype) {
    case DType::kFloat32:
      test_grouped_situglu<float>(hidden, alignment_mode, count_mode, caps);
      break;
    case DType::kFloat16:
      test_grouped_situglu<test::fp16>(hidden, alignment_mode, count_mode, caps);
      break;
    case DType::kBFloat16:
      test_grouped_situglu<test::bf16>(hidden, alignment_mode, count_mode, caps);
      break;
    default:
      FAIL() << "Unexpected test dtype";
  }
}

INSTANTIATE_TEST_SUITE_P(Operator, GroupedSiTUGLUTest,
                         ::testing::Combine(::testing::Values(DType::kFloat32, DType::kFloat16,
                                                              DType::kBFloat16),
                                            ::testing::Values(size_t{3}, size_t{16}, size_t{32}),
                                            ::testing::Values(0, 1, 2), ::testing::Values(0, 1, 2),
                                            ::testing::Values(std::pair<float, float>{4, 25},
                                                              std::pair<float, float>{2, 8})));

}  // namespace
