# TE pytorch csrc -> nanobind + torch stable ABI: migration convention

Follow this exactly so all migrated files are consistent. This is an
exploratory migration: it will NOT compile until the missing stable APIs
(which you must report) are added upstream. Where a needed stable API does not
exist, **call it as if it existed** and record it.

## Headers / namespaces
- Remove: `<ATen/*>`, `<torch/torch.h>`, `<torch/extension.h>`, `pybind11/*`,
  `torch/csrc/autograd/python_variable.h`, `ATen/cuda/*`.
- Add as needed: `<nanobind/nanobind.h>` (+ `nanobind/stl/*.h`),
  `<torch/csrc/stable/tensor.h>`, `<torch/csrc/stable/ops.h>`,
  `<torch/csrc/stable/python/interop.h>`, `<torch/csrc/stable/accelerator.h>`,
  `<torch/headeronly/...>`.
- `namespace py = pybind11;` -> `namespace nb = nanobind;` (and all `py::` -> `nb::`).

## Types
- `at::Tensor` -> `torch::stable::Tensor`.
- `py::object`/`py::handle` -> `nb::object`/`nb::handle`.
- `at::ScalarType` -> `torch::headeronly::ScalarType`.
- `TORCH_CHECK` -> `STD_TORCH_CHECK`; `NVTE_CHECK` stays.

## Python tensor <-> stable Tensor
- Receive a python tensor: `torch::stable::from_pyobject(h.ptr())` (h is nb::handle).
- Return a python tensor:
  `nb::steal<nb::object>(nb::handle(static_cast<PyObject*>(torch::stable::to_pyobject(t))))`.

## Shared helpers (assume these migrated signatures exist)
- `TensorWrapper makeTransformerEngineTensor(nb::handle tensor, nb::handle quantizer = nb::none())`
  -- builds an NVTETensor from a python tensor (plain: from_pyobject; quantized:
  read attrs via nb + from_pyobject on the underlying data tensors).
- `transformer_engine::DType GetTransformerEngineDType(torch::headeronly::ScalarType)`.
- Output allocation: prefer `torch::stable::new_empty` / `empty` / `empty_like`.
- CUDA stream: `torch::stable::accelerator::getCurrentStream()` and its native
  handle for `cudaStream_t` (record the exact call if the native-handle path is unclear).

## ATen ops
- Use `torch::stable::<op>` where it exists (sum, empty, empty_like, fill_,
  subtract, transpose, reshape, view, clone, copy_, contiguous, narrow, ...).
- `a + b` scalar add: stable lacks `add`; use it AS IF it exists
  (`torch::stable::add(a, b)`) and report it (do NOT silently rewrite to
  subtract; we want the gap in the list).
- If a needed op/overload/utility is missing from `torch::stable`, the stable C
  shim, or `accelerator`, call it as if it existed and add:
  `// TODO(stable-abi): needs <proposed decl>` at the call site, and record it.

## Registration
- `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` -> `NB_MODULE(TORCH_EXTENSION_NAME, m)`.
- `m.def("name", &fn, ...)` stays (nanobind syntax is compatible for simple defs).

## Output you must return
1. Rewrite each assigned file in place.
2. Return a MISSING-API list: one entry per distinct missing stable API, as:
   `- <proposed C++ signature> | needed by <file>::<op> | <one-line why>`
   Deduplicate within your batch. Also flag anything you could NOT migrate at
   all (hard blockers, e.g. RNG/generator internals) separately.
