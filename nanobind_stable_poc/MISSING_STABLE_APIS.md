# Stable ABI gaps for migrating Transformer Engine (PyTorch) to nanobind + torch::stable

Result of an exploratory migration of all of `transformer_engine/pytorch/csrc`
(~35 files) to nanobind + the PyTorch stable ABI. The code compiles *as if* the
items below existed (call sites are marked `// TODO(stable-abi)`).

Cross-checked against the actual `torch/csrc/stable` API: many things the
migration first assumed missing already exist and are **not** gaps (see bottom).

## A. Genuinely missing `torch::stable` ops

- `add` (Tensor+Tensor and Tensor+Scalar) — only `subtract` exists today.
- `arange`
- `reciprocal`
- `zeros` (standalone; `full`/`new_zeros` cover most cases as a workaround)
- Overload/parameter gaps on ops that otherwise exist:
  - `empty` / `new_empty` with an explicit dtype override (+ device / pin_memory).
  - `from_blob` with a custom deleter, and with an explicit device.
  - `narrow` with negative `start`.

## B. Dispatcher

- `Scalar` arguments are not supported by `torch_call_dispatcher`, which is why
  `add` (and any Scalar-arg op) can't just go through the stable dispatcher.

## C. Device / stream

- Device properties: multiprocessor (SM) count and compute capability
  (`sm_arch`). Used by normalization and bias kernels for launch config.
- Constructing a stable `Stream` from an external raw `cudaStream_t`, and
  modelling separate send/recv streams (needed by comm_gemm_overlap).

## D. Python-interop (same family as from_pyobject/to_pyobject)

- `torch.dtype`  <->  `ScalarType` (both directions): read a dtype off a Python
  object, and produce a `torch.dtype` PyObject to pass into Python constructors.
- `torch.device` <->  `Device`: same, for device.
- (TE-side, not a torch API) a nanobind `type_caster<torch::stable::Tensor>`
  built on from_pyobject/to_pyobject, and a nanobind caster for TE's `DType`
  enum (currently worked around via IntEnum -> int).

## E. Hard blockers (no stable surface; need redesign or a new subsystem)

- **CUDA RNG / Philox generator state** — `at::CUDAGeneratorImpl`,
  `at::PhiloxCudaState`, `philox_cuda_state()`. Blocks dropout, attention,
  cast/NVFP4 and quantizer stochastic rounding. Proposed fix: a stable API like
  `accelerator::philox_cuda_state(generator, elts_per_thread) -> {seed, offset}`,
  or pass seed/offset (int64) down from Python.
- **c10d ProcessGroup / collectives** — allreduce, broadcast, allgather,
  barrier. Blocks quantizer amax reduction, nvshmem_comm, comm_gemm_overlap.
  Proposed fix: do the collective in Python and pass buffers down, or a stable
  c10d shim.
- **`torch::CustomClassHolder`** — base of the CommOverlap classes bound as
  torch custom classes. Not in the stable ABI.
- **Shared `NVTE_DECLARE_COMMON_PYBIND11_HANDLES`** macro in
  `transformer_engine/common/util/pybind_helper.h` (shared by JAX + PyTorch,
  still pure pybind11) — must be ported / made framework-agnostic before the
  module builds.
- **nanobind class/enum limitations** — no `shared_ptr` holder, single base
  only (CommOverlap uses multiple inheritance + shared_ptr holder), and
  `nb::enum_` has no `module_local()`.

## Not gaps (already exist — flagged by the migration only because the agents
## lacked the headers)

- `torch::stable::accelerator`: `Stream::nativeHandle()` (raw `cudaStream_t`),
  `getCurrentStream(DeviceIndex)`, `getCurrentDeviceIndex()`, `DeviceGuard(DeviceIndex)`.
- `torch::stable::Tensor` methods: `data_ptr`, `dim`, `numel`, `sizes`,
  `strides`, `size`, `stride`, `scalar_type`, `device`, `get_device`,
  `get_device_index`, `is_cuda`, `is_cpu`, `is_contiguous`, `element_size`.
- `torch::stable` ops: `contiguous`, `empty`, `empty_like`, `from_blob` (basic),
  `full`, `narrow`, `new_empty`, `new_zeros`, `select`, `squeeze`, `sum`,
  `sum_out`, `to`, `transpose`, `reshape`, `view`, `clone`, `copy_`, `fill_`.
