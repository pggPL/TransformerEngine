"""torch.compile(fullgraph=True) can't proxy a tensor subclass without __tensor_flatten__."""

import torch


class ScaledTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, scale):
        t = torch.Tensor._make_wrapper_subclass(cls, data.shape, dtype=data.dtype, device=data.device)
        t._data = data
        t._scale = scale
        return t

    def __init__(self, data, scale):
        pass

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            return t._data if isinstance(t, ScaledTensor) else t
        return func(*torch.utils._pytree.tree_map(unwrap, args),
                    **torch.utils._pytree.tree_map(unwrap, kwargs or {}))


@torch.library.custom_op("mylib::double", mutates_args=[])
def double(x: torch.Tensor) -> torch.Tensor:
    return x * 2

@double.register_fake
def _(x):
    return torch.empty_like(x)


def fn(x):
    return torch.ops.mylib.double(x)


t = ScaledTensor(torch.randn(4, device="cuda"), scale=0.5)

print("eager:", fn(t))
print("compile:", torch.compile(fn, fullgraph=True)(t))
