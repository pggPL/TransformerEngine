# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Repro: NVFP4 RHT matrix baked into the torch.compile graph.

Demonstrates whether the NVFP4 Random Hadamard Transform (RHT) matrix is rebuilt
inside the compiled ``te.Linear`` forward on every invocation.

Background
----------
``get_rht_matrix`` (transformer_engine/pytorch/tensor/nvfp4_tensor.py) is
``@functools.lru_cache``-decorated, so in eager mode the 16x16 RHT matrix is
constructed exactly once per (sign_mask, device). Before the fix, ``Linear.forward``
called ``get_rht_matrix`` from inside the traced region to "pin" the matrix as an
op input. Dynamo ignores the ``lru_cache`` wrapper and traces the construction
(``torch.eye`` -> ``mul`` -> ``mm`` -> cast) straight into the graph, and Inductor
does not constant-fold that all-constant subgraph -- so the matrix is rebuilt on
every compiled call.

How it detects the regression
-----------------------------
The actual linear GEMM runs inside the opaque custom op, so it never appears in
Inductor's generated wrapper code ("output_code"). The ONLY matmul / eye kernels
that can show up in output_code are the RHT construction. We capture output_code
and look for them.

Expected output
---------------
* Before the fix: ``RHT matrix rebuilt inside the compiled forward `call()`: True``
* After the fix:  ``RHT matrix rebuilt inside the compiled forward `call()`: False``

Run with ``TORCHINDUCTOR_COMPILE_THREADS=1`` for a clean single-process compile:

    TORCHINDUCTOR_COMPILE_THREADS=1 python repro_rht_baked_in_graph.py
"""

import logging
import sys

import torch
import torch._logging
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch import is_nvfp4_available


def main() -> int:
    if not is_nvfp4_available():
        print("SKIP: NVFP4 is not available on this device.")
        return 0

    dtype, device = torch.bfloat16, "cuda"

    # Capture Inductor's generated wrapper code ("output_code").
    torch._logging.set_logs(output_code=True)
    captured: list[str] = []

    class _Grab(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    inductor_logger = logging.getLogger("torch._inductor")
    inductor_logger.addHandler(_Grab())
    inductor_logger.setLevel(logging.DEBUG)

    model = te.Linear(64, 32, params_dtype=dtype, device=device)
    nvfp4 = recipe.NVFP4BlockScaling()  # RHT enabled by default

    def fn(inp):
        with te.autocast(recipe=nvfp4):
            return model(inp)

    torch._dynamo.reset()

    # Eager warmup: populates the weight FP8 workspace cache so the compiled
    # fake path matches eager (mirrors tests/pytorch/test_torch_compile.py).
    warmup = torch.randn(32, 64, dtype=dtype, device=device, requires_grad=True)
    model.zero_grad(set_to_none=True)
    fn(warmup).sum().backward()
    model.zero_grad(set_to_none=True)

    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(torch.randn(32, 64, dtype=dtype, device=device, requires_grad=True))
    out.sum().backward()
    torch.cuda.synchronize()

    code = "\n".join(captured)
    signals = {
        "extern_kernels.mm (RHT matmul)": "extern_kernels.mm" in code,
        "triton eye/hadamard kernels": (
            "triton_poi_fused_eye_lift_fresh" in code
            or "triton_poi_fused_lift_fresh_mul" in code
        ),
        "get_rht_matrix source lines (nvfp4_tensor.py:10x)": "nvfp4_tensor.py:10" in code,
    }
    baked = any(signals.values())

    print("=" * 64)
    for name, present in signals.items():
        print(f"  {'PRESENT' if present else 'absent ':>8}  {name}")
    print("=" * 64)
    print("RHT matrix rebuilt inside the compiled forward `call()`:", baked)

    # Exit non-zero if the regression is present, so this doubles as a check.
    return 1 if baked else 0


if __name__ == "__main__":
    sys.exit(main())
