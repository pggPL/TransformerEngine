/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file torch_backend.h
 *  \brief Single binary boundary between Transformer Engine and PyTorch.
 *
 *  This header is the ONLY place in the PyTorch extension that is allowed to
 *  include libtorch/ATen/c10 headers and to name ``at::`` / ``c10::`` /
 *  ``torch::`` symbols. Every other translation unit talks to PyTorch
 *  exclusively through the type aliases (``using``) and free functions
 *  (``methods``) declared here.
 *
 *  Why: it lets us swap the concrete tensor implementation from the classic
 *  ``at::Tensor`` to ``torch::stable::Tensor`` (the LibTorch stable ABI) by
 *  flipping a single compile flag, without touching the ~40 extension files.
 *  It also collapses the TE<->torch ABI surface to one auditable file, which a
 *  lint guard (``qa/L0_pytorch_lint/check_torch_boundary.sh``) enforces.
 *
 *  Switch:
 *    - default            -> classic libtorch (``at::Tensor``)
 *    - -DTE_WITH_STABLE_ABI -> torch stable ABI (``torch::stable::Tensor``)
 */

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_BACKEND_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_BACKEND_H_

// ===========================================================================
// The one and only libtorch include site.
// ===========================================================================
#ifdef TE_WITH_STABLE_ABI
// ---------------------------------------------------------------------------
// Stable-ABI path (migration target). torch::stable::Tensor exposes a narrower
// interface than at::Tensor, so operations are routed through the free-function
// wrappers below (implemented in torch_backend.cpp against the stable C shim).
// The aliases that have a direct stable counterpart are provided here; anything
// still missing a stable wrapper is tracked in the branch CLAUDE.md.
// ---------------------------------------------------------------------------
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#else
// ---------------------------------------------------------------------------
// Classic libtorch path (default).
// ---------------------------------------------------------------------------
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/DispatchStub.h>
#include <c10/macros/Macros.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/extension.h>
#include <torch/torch.h>

#include "c10/util/ArrayRef.h"
#endif

#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace transformer_engine::pytorch {

// ===========================================================================
// Type aliases -- the "usings" every extension file must use instead of naming
// at::/c10::/torch:: types directly.
// ===========================================================================
#ifdef TE_WITH_STABLE_ABI

using Tensor = torch::stable::Tensor;
// NOTE: ScalarType/Device/Stream/ProcessGroup/PhiloxCudaState/CUDAGeneratorImpl
// stable equivalents are wired incrementally; see branch CLAUDE.md.

#else

using Tensor = at::Tensor;
using ScalarType = at::ScalarType;
using Device = at::Device;
using Stream = at::Stream;
using TensorOptions = at::TensorOptions;
using IntArrayRef = c10::IntArrayRef;
template <typename T>
using ArrayRef = c10::ArrayRef<T>;
template <typename T>
using IntrusivePtr = c10::intrusive_ptr<T>;

// Distributed process group (python: torch.distributed.ProcessGroup).
using ProcessGroup = c10d::ProcessGroup;

// CUDA RNG interop.
using PhiloxCudaState = at::PhiloxCudaState;
using CUDAGeneratorImpl = at::CUDAGeneratorImpl;

#endif

// Optional tensor, matches python's ``Optional[torch.Tensor]``.
using MaybeTensor = std::optional<Tensor>;

// ===========================================================================
// dtype mapping -- the TE<->torch dtype boundary.
// ===========================================================================

/*! \brief Map a TE DType to the corresponding torch scalar type. */
ScalarType GetATenDType(transformer_engine::DType t);

/*! \brief Map a torch scalar type to the corresponding TE DType. */
transformer_engine::DType GetTransformerEngineDType(ScalarType t);

// ===========================================================================
// Tensor factories -- wrappers over at::empty/at::zeros/... so no extension
// file has to name the ATen factory functions.
// ===========================================================================

/*! \brief Allocate a CUDA tensor of the given shape/dtype.
 *  \param zero_init  If true, zero-initialize; otherwise leave uninitialized.
 */
Tensor new_cuda_tensor(const std::vector<int64_t>& shape, ScalarType dtype, bool zero_init);

// ===========================================================================
// CUDA stream interop -- the only place that names at::cuda::getCurrentCUDAStream.
// ===========================================================================

/*! \brief Current CUDA stream for the active device, as a raw ``cudaStream_t``. */
cudaStream_t getCurrentCUDAStream();

}  // namespace transformer_engine::pytorch

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_BACKEND_H_
