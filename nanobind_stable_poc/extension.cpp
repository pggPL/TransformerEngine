#include <nanobind/nanobind.h>

#ifndef STABLE_TORCH_COMPILE
#define STABLE_TORCH_COMPILE 0
#endif

#if STABLE_TORCH_COMPILE
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/python/interop.h>  // from_pyobject / to_pyobject (2.14+)
#include <torch/csrc/stable/tensor.h>
#else
#include <ATen/ATen.h>
#include <torch/csrc/autograd/python_variable.h>
#endif

namespace nb = nanobind;

// ============================================================
// Abstraction layer
// ============================================================

#if STABLE_TORCH_COMPILE

using Tensor = torch::stable::Tensor;

// Py -> stable Tensor. With the 2.14 python-interop shim this is a single
// zero-copy call. (Before the shim this needed manual data_ptr/shape/stride
// extraction + torch::stable::from_blob -- see the nanobind_example branch.)
static Tensor to_tensor(nb::handle h) { return torch::stable::from_pyobject(h.ptr()); }

static Tensor add_scalar(const Tensor& t, double v) {
  // stable ops still lack `add`/Scalar, so add is done as subtract(t, v, -1).
  auto tmp = torch::stable::empty_like(t);
  torch::stable::fill_(tmp, v);
  return torch::stable::subtract(t, tmp, -1.0);
}

// stable Tensor -> Py. Single zero-copy call with the shim. (Before the shim
// this allocated a new tensor in Python and copy_'d into it -- a real copy.)
static nb::object to_python(Tensor t, nb::handle /*like*/) {
  auto* p = static_cast<PyObject*>(torch::stable::to_pyobject(t));  // new ref
  return nb::steal<nb::object>(nb::handle(p));
}

#else  // !STABLE_TORCH_COMPILE

using Tensor = at::Tensor;

static Tensor to_tensor(nb::handle h) { return THPVariable_Unpack(h.ptr()); }

static Tensor add_scalar(const Tensor& t, double v) { return t + v; }

static nb::object to_python(Tensor t, nb::handle) {
  PyObject* p = THPVariable_Wrap(std::move(t));
  return nb::steal<nb::object>(nb::handle(p));
}

#endif  // STABLE_TORCH_COMPILE

// ============================================================
// Kernel -- identical regardless of STABLE_TORCH_COMPILE
// ============================================================

nb::object kernel(nb::handle a, nb::handle obj) {
  auto ta = to_tensor(a);
  double value = nb::cast<double>(obj.attr("value_test"));
  auto result = add_scalar(ta, value);
  return to_python(result, a);
}

NB_MODULE(my_ext, m) { m.def("kernel", &kernel); }
