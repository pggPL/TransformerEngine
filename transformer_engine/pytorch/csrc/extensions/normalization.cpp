/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"
#include <iostream>

namespace transformer_engine::pytorch {
  std::pair<TensorWrapper, py::object> createOutputTensor(const NVTEShape& shape,
                                                          DType dtype,
                                                          py::handle quantizer) {
    std::vector<size_t> shape_vec;
    for (int i = 0; i < shape.ndim; i++) {
      size_t t = shape.data[i];
      shape_vec.push_back(t);
    }
    std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
    return my_quantizer->create_tensor(shape_vec, dtype);
  }
  std::pair<TensorWrapper, py::object> createOutputTensor(std::vector<size_t>& shape,
                                                          DType dtype,
                                                          py::handle quantizer) {
    std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
    return my_quantizer->create_tensor(shape, dtype);
  }
}

std::vector<py::object> layernorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                      const at::Tensor &mu, const at::Tensor &rsigma,
                                      const at::Tensor &gamma, const int sm_margin,
                                      const bool zero_centered_gamma) {
  using namespace transformer_engine::pytorch;
  const auto &dz_ = dz.contiguous();
  const auto &x_ = x.contiguous();
  const auto &mu_ = mu.contiguous();
  const auto &rsigma_ = rsigma.contiguous();
  const auto &gamma_ = gamma.contiguous();

  auto dx = at::empty_like(x_);
  auto dgamma = at::empty_like(gamma_);
  auto dbeta = at::empty_like(gamma_);
  transformer_engine::TensorWrapper workspace, barrier, dgamma_part, dbeta_part;

  auto dz_cu = makeTransformerEngineTensor(dz_);
  auto x_cu = makeTransformerEngineTensor(x_);
  auto mu_cu = makeTransformerEngineTensor(mu_);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma_);
  auto gamma_cu = makeTransformerEngineTensor(gamma_);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);
  auto dbeta_cu = makeTransformerEngineTensor(dbeta);

  // This call populates tensors with the required config.
  const auto bwd_fun = zero_centered_gamma ? nvte_layernorm1p_bwd : nvte_layernorm_bwd;
  bwd_fun(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
          dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(), dbeta_part.data(),
          at::cuda::getCurrentCUDAStream(),
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
          barrier.data());

  // Alloc space for Tensors.
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());
  if (!std::getenv("NVTE_BWD_LAYERNORM_USE_CUDNN")) {
    auto barrier_data = allocateSpace(barrier.shape(), barrier.dtype(), true);
    auto dgamma_part_data = allocateSpace(dgamma_part.shape(), dgamma_part.dtype());
    auto dbeta_part_data = allocateSpace(dbeta_part.shape(), dbeta_part.dtype());
    barrier =
        makeTransformerEngineTensor(barrier_data.data_ptr(), barrier.shape(), barrier.dtype());
    dgamma_part = makeTransformerEngineTensor(dgamma_part_data.data_ptr(), dgamma_part.shape(),
                                              dgamma_part.dtype());
    dbeta_part = makeTransformerEngineTensor(dbeta_part_data.data_ptr(), dbeta_part.shape(),
                                             dbeta_part.dtype());
  }

  // Actual call to bwd kernel.
  bwd_fun(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
          dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(), dbeta_part.data(),
          at::cuda::getCurrentCUDAStream(),
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
          barrier.data());

  return {py::cast(dx), py::cast(dgamma), py::cast(dbeta)};
}

std::vector<py::object> layernorm_fwd(
    py::handle input, py::handle weight, MaybeTensor bias, float eps,
    py::object ln_out, py::handle quantizer, DType out_dtype,
    const int sm_margin, const bool zero_centered_gamma) {
  using namespace transformer_engine::pytorch;
  using namespace transformer_engine;

  auto none = py::none();
  const TensorWrapper& input_tensor = makeTransformerEngineTensor(input, none);
  const TensorWrapper& weight_tensor = makeTransformerEngineTensor(weight, none);

  TensorWrapper bias_tensor;
  MaybeTensor bias_grad = std::nullopt;
  if (bias.has_value()) {
      bias_tensor = makeTransformerEngineTensor(*bias);
  }

  // Choose kernel implementation
  const auto func = zero_centered_gamma ? nvte_layernorm1p_fwd : nvte_layernorm_fwd;

  // Tensor dimensions
  size_t N = static_cast<size_t>(input_tensor.size(0));
  size_t H = static_cast<size_t>(input_tensor.size(1));
  std::vector<size_t> size = {N, H};

  // Construct Transformer Engine tensors
  at::Tensor mu  = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
  at::Tensor rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));

  TensorWrapper ln_out_tensor;

  if (ln_out.is_none()) {
    std::tie(ln_out_tensor, ln_out) = createOutputTensor(size, out_dtype, quantizer);
  } else {
    ln_out_tensor = makeTransformerEngineTensor(ln_out, quantizer);
  }
  TensorWrapper mu_cu = makeTransformerEngineTensor(mu);
  TensorWrapper rsigma_cu = makeTransformerEngineTensor(rsigma);


  // Query workspace sizes
  transformer_engine::TensorWrapper workspace, barrier;
  func(input_tensor.data(), weight_tensor.data(), bias_tensor.data(), eps, ln_out_tensor.data(), mu_cu.data(),
       rsigma_cu.data(), at::cuda::getCurrentCUDAStream(),
       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
       barrier.data());

  // Allocate workspaces
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  if (!std::getenv("NVTE_FWD_LAYERNORM_USE_CUDNN")) {
    auto barrier_data = allocateSpace(barrier.shape(), barrier.dtype(), true);
    barrier =
        makeTransformerEngineTensor(barrier_data.data_ptr(), barrier.shape(), barrier.dtype());
  }

  // Launch kernel
  func(input_tensor.data(), weight_tensor.data(), bias_tensor.data(), eps, ln_out_tensor.data(), mu_cu.data(),
       rsigma_cu.data(), at::cuda::getCurrentCUDAStream(),
       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
       barrier.data());
  return {ln_out, py::cast(mu), py::cast(rsigma)};
}


std::vector<py::object> rmsnorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                    const at::Tensor &rsigma, const at::Tensor &gamma,
                                    const int sm_margin, const bool zero_centered_gamma) {
  using namespace transformer_engine::pytorch;
  const auto &dz_ = dz.contiguous();
  const auto &x_ = x.contiguous();
  const auto &rsigma_ = rsigma.contiguous();
  const auto &gamma_ = gamma.contiguous();

  auto dx = at::empty_like(x_);
  auto dgamma = at::empty_like(gamma_);
  transformer_engine::TensorWrapper workspace, barrier, dgamma_part;

  auto dz_cu = makeTransformerEngineTensor(dz_);
  auto x_cu = makeTransformerEngineTensor(x_);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma_);
  auto gamma_cu = makeTransformerEngineTensor(gamma_);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);

  // This call populates tensors with the required config.
  const auto bwd_fun = zero_centered_gamma ? nvte_rmsnorm1p_bwd : nvte_rmsnorm_bwd;
  bwd_fun(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
          dgamma_cu.data(), dgamma_part.data(), at::cuda::getCurrentCUDAStream(),
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
          barrier.data());

  // Alloc space for Tensors.
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());
  if (!std::getenv("NVTE_BWD_RMSNORM_USE_CUDNN")) {
    auto barrier_data = allocateSpace(barrier.shape(), barrier.dtype(), true);
    auto dgamma_part_data = allocateSpace(dgamma_part.shape(), dgamma_part.dtype());
    barrier =
        makeTransformerEngineTensor(barrier_data.data_ptr(), barrier.shape(), barrier.dtype());
    dgamma_part = makeTransformerEngineTensor(dgamma_part_data.data_ptr(), dgamma_part.shape(),
                                              dgamma_part.dtype());
  }

  // Actual call to bwd kernel.
  bwd_fun(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
          dgamma_cu.data(), dgamma_part.data(), at::cuda::getCurrentCUDAStream(),
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
          barrier.data());

  return {py::cast(dx), py::cast(dgamma)};
}

std::vector<py::object> rmsnorm_fwd(const py::handle &input, const py::handle &weight,
                                    float eps, py::object ln_out, py::handle quantizer,
                                    transformer_engine::DType otype,
                                    const int sm_margin, const bool zero_centered_gamma) {
  using namespace transformer_engine::pytorch;
  using namespace transformer_engine;

  auto none = py::none();
  const TensorWrapper& input_tensor = makeTransformerEngineTensor(input, none);
  const TensorWrapper& weight_tensor = makeTransformerEngineTensor(weight, none);

  // Choose kernel implementation
  const auto func = zero_centered_gamma ? nvte_rmsnorm1p_fwd : nvte_rmsnorm_fwd;

  // Tensor dimensions
  size_t N = static_cast<size_t>(input_tensor.shape().data[0]);
  size_t H = static_cast<size_t>(input_tensor.shape().data[1]);

  // Construct Transformer Engine tensors
  auto rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
  auto input_cu = makeTransformerEngineTensor(input, none);
  auto gamma_cu = makeTransformerEngineTensor(weight, none);
  std::vector<size_t> size = {N, H};  
  TensorWrapper ln_out_tensor;

  if (ln_out.is_none()) {
    std::tie(ln_out_tensor, ln_out) = createOutputTensor(size, otype, quantizer);
  } else {
    std::cout << "aaa" << std::endl;
    ln_out_tensor = makeTransformerEngineTensor(ln_out, quantizer);
    std::cout << "bbb" << std::endl;
  }
  auto rsigma_cu = makeTransformerEngineTensor(rsigma);

  // Query workspace sizes
  transformer_engine::TensorWrapper workspace, barrier;
  func(input_cu.data(), gamma_cu.data(), eps, ln_out_tensor.data(), rsigma_cu.data(),
       at::cuda::getCurrentCUDAStream(),
       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
       barrier.data());

  // Allocate workspaces
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());
  if (!std::getenv("NVTE_FWD_RMSNORM_USE_CUDNN")) {
    auto barrier_data = allocateSpace(barrier.shape(), barrier.dtype(), true);
    barrier =
        makeTransformerEngineTensor(barrier_data.data_ptr(), barrier.shape(), barrier.dtype());
  }

  // Launch kernel
  func(input_cu.data(), gamma_cu.data(), eps, ln_out_tensor.data(), rsigma_cu.data(),
       at::cuda::getCurrentCUDAStream(),
       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
       barrier.data());

  return {ln_out, py::none(), py::cast(rsigma)};
}
