# Nanobind + PyTorch stable ABI POC (with the python-interop shims)

Adapted from the `nanobind_example` branch. Same kernel
(`kernel(tensor, obj)` computes `tensor + obj.value_test`), but the stable-ABI
path uses the PyTorch 2.14 python-interop shims
(`torch::stable::from_pyobject` / `to_pyobject`) instead of the workarounds the
original example needed.

## What changed vs `nanobind_example`

| step | `nanobind_example` (workaround) | this POC (2.14 shims) |
|---|---|---|
| Py -> stable Tensor | extract `data_ptr`/`shape`/`stride` + `torch::stable::from_blob` | `torch::stable::from_pyobject(obj)` (zero-copy) |
| stable Tensor -> Py | `torch.empty_like` + `torch::stable::copy_` (a real copy) | `torch::stable::to_pyobject(t)` (zero-copy) |

So the manual `data_ptr`/`shape`/`stride`/`device`/`dtype` plumbing and the
extra allocation+copy are both gone.

## Tradeoff

The stable path now links `torch_python` (the shims live there), so
`CMakeLists.txt` links it in both modes. This is expected: the python-interop
shims are not libtorch-only.

## Build & run

```bash
pip install nanobind
mkdir -p build && cd build
STABLE_TORCH_COMPILE=1 cmake .. && make -j   # needs torch >= 2.14
cd .. && python main.py
```

Verified on a source build of PyTorch 2.14: builds, runs, output matches
`tensor + 5`.
