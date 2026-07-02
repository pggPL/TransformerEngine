import os
import sys

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import my_ext


class MyObject:
    def __init__(self):
        self.value_test = 5


tensor_in = torch.randn(4, 4)
obj = MyObject()

tensor_out = my_ext.kernel(tensor_in, obj)

print("Input:\n", tensor_in)
print("Output (should be input + 5):\n", tensor_out)
print("Match:", torch.allclose(tensor_out, tensor_in + 5))
assert isinstance(tensor_out, torch.Tensor)
assert torch.allclose(tensor_out, tensor_in + 5)
print("OK")
