"""torch.compile(fullgraph=True) + __tensor_flatten__ + passthrough custom op."""

import torch


_passthrough_ops: set = set()

class ScaledTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, scale):
        t = torch.Tensor._make_wrapper_subclass(cls, data.shape, dtype=data.dtype, device=data.device)
        t._data = data
        t._scale = scale
        return t

    def __init__(self, data, scale):
        pass

    def __repr__(self):
        return f"ScaledTensor(shape={self.shape}, scale={self._scale})"

    def __tensor_flatten__(self):
        return ["_data"], {"scale": self._scale}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return ScaledTensor(inner_tensors["_data"], metadata["scale"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func in _passthrough_ops:
            return super().__torch_dispatch__(func, types, args, kwargs or {})
        def unwrap(t):
            return t._data if isinstance(t, ScaledTensor) else t
        return func(*torch.utils._pytree.tree_map(unwrap, args),
                    **torch.utils._pytree.tree_map(unwrap, kwargs or {}))


@torch.library.custom_op("mylib::double_v2", mutates_args=[])
def double_v2(x: torch.Tensor) -> torch.Tensor:
    assert isinstance(x, ScaledTensor), f"Expected ScaledTensor, got {type(x).__name__}"
    return ScaledTensor(x._data * 2, x._scale)

@double_v2.register_fake
def _(x):
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)

_passthrough_ops.add(torch.ops.mylib.double_v2.default)

def fn(x):
    return torch.ops.mylib.double_v2(x)


t = ScaledTensor(torch.randn(4, device="cuda"), scale=0.5)
r = torch.compile(fn, fullgraph=True)(t)
