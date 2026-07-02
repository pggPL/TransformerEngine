/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

#ifdef NVTE_ENABLE_NVSHMEM
#include <nvshmem.h>
#include <nvshmem_api/nvshmem_waitkernel.h>
#include <nvshmemx.h>
#endif

#include <cuda.h>
#include <cuda_fp8.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

// ================= HARD BLOCKER (c10d / torch.distributed) ================
// init_nvshmem_backend() takes a c10d::ProcessGroup* and calls
// getRank()/getSize()/getBackendType()/broadcast(). The c10d process-group API
// is NOT part of the torch stable ABI (no header-only / C-shim equivalent).
// This function cannot be migrated until a stable distributed handle exists.
// Proposed fix (see report): drive the bootstrap broadcast from Python and pass
// the exchanged unique-id bytes down as a plain tensor / byte buffer.
// ==========================================================================

namespace transformer_engine::pytorch {

namespace {
// TODO(stable-abi): torch::stable::accelerator lacks a native cudaStream_t handle.
inline cudaStream_t current_cuda_stream() {
  return static_cast<cudaStream_t>(
      torch::stable::accelerator::getCurrentStream(
          torch::stable::accelerator::getCurrentDeviceIndex())
          .stream());
}
}  // namespace

void init_nvshmem_backend(c10d::ProcessGroup *process_group) {
#ifdef NVTE_ENABLE_NVSHMEM
  nvshmemx_init_attr_t attr = {};
  nvshmemx_uniqueid_t id = {};

  int my_rank = process_group->getRank();
  int num_ranks = process_group->getSize();
  if (my_rank == 0) {
    nvshmemx_get_uniqueid(&id);
  }

  auto backend_is_nccl = (process_group->getBackendType() == c10d::ProcessGroup::BackendType::NCCL);
  NVTE_CHECK(backend_is_nccl, "Currently only support NCCL boostrap for NVSHMEM");
  // TODO(stable-abi): needs torch::stable::from_blob(void*, IntArrayRef,
  // ScalarType, Device) for a CPU buffer.
  auto datatensor = torch::stable::from_blob(
      reinterpret_cast<void *>(&id),
      {static_cast<int64_t>(sizeof(nvshmemx_uniqueid_t) / sizeof(uint8_t))},
      torch::headeronly::ScalarType::Byte, torch::stable::Device(torch::headeronly::DeviceType::CPU));
  // TODO(stable-abi): needs torch::stable::to(Tensor, Device).
  auto datatmp = (backend_is_nccl)
                     ? torch::stable::to(datatensor,
                                         torch::stable::Device(torch::headeronly::DeviceType::CUDA))
                     : datatensor;

  // HARD BLOCKER: c10d::BroadcastOptions / ProcessGroup::broadcast / Work are
  // not in the stable ABI.
  c10d::BroadcastOptions bcast_opts;
  bcast_opts.rootRank = 0;
  std::vector<torch::stable::Tensor> datachunk = {datatmp};
  auto work = process_group->broadcast(datachunk, bcast_opts);
  work->wait();

  if (backend_is_nccl) {
    torch::stable::copy_(datatensor,
                         torch::stable::to(datatmp, torch::stable::Device(
                                                        torch::headeronly::DeviceType::CPU)));
    datatmp = torch::stable::Tensor();
  }

  nvshmemx_set_attr_uniqueid_args(my_rank, num_ranks, &id, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

  NVTE_CHECK(my_rank == nvshmem_my_pe(), "my_rank: ", my_rank,
             " != nvshmem_my_pe(): ", nvshmem_my_pe());
  NVTE_CHECK(num_ranks == nvshmem_n_pes(), "num_ranks: ", num_ranks,
             " != nvshmem_n_pes(): ", nvshmem_n_pes());
#else
  NVTE_ERROR("Internal TE error: init_nvshmem_backend cannot be initialized with valid PyTorch ",
             "distributed process groups when TE is compiled with NVTE_ENABLE_NVSHMEM=1!");
#endif
}

void nvshmem_wait_on_current_stream(torch::stable::Tensor signal, const std::string &wait_kind) {
#ifdef NVTE_ENABLE_NVSHMEM
  uint64_t *sig_addr = reinterpret_cast<uint64_t *>(signal.data_ptr());
  cudaStream_t cur_stream = current_cuda_stream();

  WaitKind wait_kind_enum = WaitKind::STREAM_WAIT;

  if (wait_kind == "kernel") {
    wait_kind_enum = WaitKind::KERNEL_WAIT;
  } else if (wait_kind == "nvshmem") {
    wait_kind_enum = WaitKind::NVSHMEM_WAIT;
  } else if (wait_kind == "stream") {
    wait_kind_enum = WaitKind::STREAM_WAIT;
  } else {
    NVTE_ERROR("Invalid wait kind: ", wait_kind);
  }
  nvshmem_wait_on_stream(sig_addr, wait_kind_enum, cur_stream);

#else
  NVTE_ERROR(
      "Internal TE error: nvshmem_wait_on_current_stream cannot be initialized with valid PyTorch ",
      "distributed process groups when TE is compiled with NVTE_ENABLE_NVSHMEM=1!");
#endif
}

torch::stable::Tensor create_nvshmem_tensor(const std::vector<int64_t> &shape,
                                            torch::headeronly::ScalarType dtype) {
#ifdef NVTE_ENABLE_NVSHMEM
  auto device = torch::stable::Device(torch::headeronly::DeviceType::CUDA,
                                      torch::stable::accelerator::getCurrentDeviceIndex());
  // TODO(stable-abi): needs torch::headeronly::elementSize(ScalarType).
  auto size = torch::headeronly::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  // TODO(stable-abi): needs torch::stable::from_blob(void*, IntArrayRef, deleter,
  // ScalarType, Device) with a custom deleter.
  return torch::stable::from_blob(
      nvshmem_malloc(size), shape, [](void *ptr) { nvshmem_free(ptr); }, dtype, device);
#else
  NVTE_ERROR("Internal TE error: create_nvshmem_tensor cannot be initialized with valid PyTorch ",
             "distributed process groups when TE is compiled with NVTE_ENABLE_NVSHMEM=1!");
#endif
}

void nvshmem_send_on_current_stream(torch::stable::Tensor src, torch::stable::Tensor dst, int peer,
                                    torch::stable::Tensor signal) {
#ifdef NVTE_ENABLE_NVSHMEM
  void *src_ptr = reinterpret_cast<void *>(src.data_ptr());
  void *dst_ptr = reinterpret_cast<void *>(dst.data_ptr());
  uint64_t *sig_addr = reinterpret_cast<uint64_t *>(signal.data_ptr());
  // TODO(stable-abi): needs torch::stable::Tensor::element_size().
  auto nelement = src.numel() * src.element_size();
  uint64_t sigval = 1;
  cudaStream_t cur_stream = current_cuda_stream();

  nvshmemx_putmem_signal_on_stream(dst_ptr, src_ptr, nelement, sig_addr, sigval, NVSHMEM_SIGNAL_SET,
                                   peer, cur_stream);
#else
  NVTE_ERROR(
      "Internal TE error: nvshmem_send_on_current_stream cannot be initialized with valid PyTorch ",
      "distributed process groups when TE is compiled with NVTE_ENABLE_NVSHMEM=1!");
#endif
}
void nvshmem_finalize() {
#ifdef NVTE_ENABLE_NVSHMEM
  nvshmem_finalize();
#else
  NVTE_ERROR("Internal TE error: nvshmem_finalize cannot be initialized with valid PyTorch ",
             "distributed process groups when TE is compiled with NVTE_ENABLE_NVSHMEM=1!");
#endif
}

}  // namespace transformer_engine::pytorch
