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

// --- Types -----------------------------------------------------------------
using Tensor = at::Tensor;               // also torch::Tensor
using ScalarType = at::ScalarType;       // also c10::ScalarType
using Device = at::Device;               // also c10::Device
using Stream = at::Stream;
using TensorOptions = at::TensorOptions;  // also torch::TensorOptions
using IntArrayRef = c10::IntArrayRef;
template <typename T>
using ArrayRef = c10::ArrayRef<T>;
template <typename T>
using IntrusivePtr = c10::intrusive_ptr<T>;
using Generator = at::Generator;
using CustomClassHolder = torch::CustomClassHolder;

// Distributed process group (python: torch.distributed.ProcessGroup) and the
// collective option structs used for amax reduction.
using ProcessGroup = c10d::ProcessGroup;
using ReduceOp = c10d::ReduceOp;
using c10d::AllreduceCoalescedOptions;
using c10d::AllreduceOptions;
using c10d::BroadcastOptions;

// CUDA interop.
using PhiloxCudaState = at::PhiloxCudaState;
using CUDAGeneratorImpl = at::CUDAGeneratorImpl;
using CUDAStream = at::cuda::CUDAStream;
using CUDAGuard = at::cuda::CUDAGuard;

// --- dtype / device constants (torch-free spellings) -----------------------
// RHS keeps the original torch spelling on purpose (this is the boundary).
inline constexpr auto kCUDA = at::kCUDA;
inline constexpr auto kCPU = at::kCPU;
inline constexpr auto kByte = at::kByte;
inline constexpr auto kUInt8 = torch::kUInt8;
inline constexpr auto kInt8 = torch::kInt8;
inline constexpr auto kInt32 = torch::kInt32;
inline constexpr auto kInt64 = torch::kInt64;
inline constexpr auto kLong = at::kLong;
inline constexpr auto kFloat = at::kFloat;
inline constexpr auto kFloat32 = torch::kFloat32;
inline constexpr auto kHalf = at::kHalf;
inline constexpr auto kBFloat16 = at::kBFloat16;
inline constexpr auto kBool = at::kBool;
inline constexpr auto kFloat8_e4m3fn = at::kFloat8_e4m3fn;
inline constexpr auto kFloat8_e5m2 = at::kFloat8_e5m2;

// --- Factories / ops re-exported with identical semantics ------------------
// (using-declarations bring every overload; behaviour is unchanged.)
// NOTE: at::device()/at::dtype() are intentionally NOT re-exported -- their bare
// names collide with the many locals/params called `device`/`dtype`. Build
// TensorOptions explicitly instead: TensorOptions().device(...).dtype(...).
using at::CUDA;
using at::empty;
using at::empty_like;
using at::from_blob;
using at::get_generator_or_default;
using at::reciprocal;
using at::sum_out;
using at::zeros;
using c10::elementSize;
using torch::range;

// --- CUDA helpers ----------------------------------------------------------
using at::cuda::current_device;
using at::cuda::getCurrentCUDAStream;
using at::cuda::getCurrentDeviceProperties;
using at::cuda::getStreamFromExternal;
using at::cuda::detail::getDefaultCUDAGenerator;

// --- torch.Tensor indexing (Slice/None/TensorIndex) ------------------------
namespace indexing = torch::indexing;

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

}  // namespace transformer_engine::pytorch

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_BACKEND_H_
