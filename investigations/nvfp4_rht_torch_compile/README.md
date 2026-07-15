# NVFP4 RHT matrix rebuilt on every `torch.compile` invocation

## Summary

On the `torch.compile` custom-op path for `te.Linear`, running the **NVFP4**
recipe with the Random Hadamard Transform enabled (the default
`recipe.NVFP4BlockScaling()`) rebuilds the 16×16 RHT matrix **inside the compiled
forward, on every invocation** — a `torch.eye` + two pointwise kernels + a cuBLAS
`mm` per linear per step. In eager mode the same matrix is built exactly once.

The absolute cost is small (a 16×16 GEMM plus two tiny kernels), but it is fully
redundant, adds kernel launches to every step, and negates the workspace-pinning
optimization the compile path was written to provide.

## Root cause

`get_rht_matrix` (`transformer_engine/pytorch/tensor/nvfp4_tensor.py`) is
`@functools.lru_cache`-decorated:

```python
@functools.lru_cache(maxsize=None)
def get_rht_matrix(with_random_sign_mask: bool, device: int) -> torch.Tensor:
    ...
    sign_matrix = signs * torch.eye(16, dtype=torch.float32, device=device)
    rht_matrix = sign_matrix @ get_hadamard_matrix(16, device=device)   # the mm
    return rht_matrix.to(dtype=torch.bfloat16)
```

In eager mode the cache makes it a one-time build. `Linear.forward` used to call
it from **inside the traced region** to pin the matrix as a custom-op input
(`LinearFwdArgs.rht_matrix`):

```python
if use_compiled_op:
    cublas_workspace = get_cublas_workspace(inp.device.index, False, False)
    from ..tensor.nvfp4_tensor import NVFP4Quantizer, get_rht_matrix
    if isinstance(input_quantizer, NVFP4Quantizer):
        rht_matrix = get_rht_matrix(input_quantizer._with_random_sign_mask, inp.device.index)
```

Two things then go wrong under compile:

1. **Dynamo ignores the `lru_cache` wrapper** and traces the *body* of
   `get_rht_matrix` into the graph. Dynamo emits this explicit warning at the
   call site:

   ```
   UserWarning: Dynamo detected a call to a functools.lru_cache-wrapped function
   at 'linear.py:2341'. Dynamo ignores the cache wrapper and directly traces the
   wrapped function.
   ```

2. **Inductor does not constant-fold** the resulting all-constant subgraph, so it
   survives into the per-invocation `call()`.

### Evidence — AOT forward graph

The construction appears in the forward graph, annotated to `nvfp4_tensor.py:101-103`:

```
mul   = aten.mul(sign_vector, eye)      # signs * eye
mul_1 = aten.mul(hadamard_const, 0.25)  # hadamard * scale
mm    = aten.mm.default(mul, mul_1)     # sign_matrix @ hadamard   <-- RHT matmul
convert_element_type_1 = to(mm, bf16)   # feeds the custom op
linear = transformer_engine_compile.linear.default(..., convert_element_type_1, ...)
```

### Evidence — final Inductor `call()` (runs every forward)

```python
def call(self, args):
    ...
    triton_poi_fused_eye_lift_fresh_mul_0.run(_tensor_constant0, buf1, ...)  # signs·eye
    triton_poi_fused_lift_fresh_mul_1.run(_tensor_constant1, buf2, ...)      # hadamard·0.25
    extern_kernels.mm(buf1, buf2, out=buf3)                                  # RHT matmul
```

The actual linear GEMM runs *inside* the opaque custom op, so it never shows up in
`output_code` — the only `mm`/eye kernels that can appear there are the RHT build.
That is what the repro keys on.

### Why the pinned field never helped

`LinearFwdArgs.rht_matrix` is declared and set but **never read** by
`_linear_forward_impl`. As the `LinearFwdArgs` comment itself notes, "the op body
never reads them ... the quantizer fetch the same globals by address." The real
RHT consumer is `tex.quantize`, which reads `rht_matrix` off the **reconstructed
quantizer** that crosses the op boundary. That quantizer is rebuilt once at
compile time (via the `_rebuild_quantizer(...)` expression baked into the opaque
bundle → `_rebuild_derived_state` → `get_rht_matrix`) and reused every call, so it
is already a compile-time constant with a static address — including under CUDA
graphs. The pinned field was therefore redundant, and calling `get_rht_matrix`
from the traced region was pure overhead.

The graph shows the pinning was not merely redundant but silently **defeated**:
the trace-time-materialized matrix and the in-graph recompute coexist, and the
recompute wins. In the AOT forward graph the pinned tensor is a **dead input** —

```
def forward(self, primals_1, primals_2, primals_3, primals_4: "bf16[16, 16]..."):
    ...                                     # primals_4 is the pinned rht_matrix
    mm = aten.mm.default(mul, mul_1)        # the recompute
    convert_element_type_1 = to(mm, bf16)
    linear = ...linear.default(..., convert_element_type_1, ...)  # op consumes THIS
```

`primals_4` appears only in the signature and is never consumed (confirmed: one
textual occurrence, no uses), while `convert_element_type_1` — the freshly
recomputed matrix — is the value actually threaded into the op. So threading the
cached tensor in as an input did nothing; Dynamo inlined the `lru_cache` call
alongside it and Inductor kept the inlined build.

## The fix

Drop the RHT fetch from the traced `forward` (keep pinning the cuBLAS workspace,
which is consumed by `general_gemm`). See the diff on
`transformer_engine/pytorch/module/linear.py` — the `if isinstance(input_quantizer,
NVFP4Quantizer): rht_matrix = get_rht_matrix(...)` block is removed and
`rht_matrix` stays `None`.

Because the quantizer already carries its own compile-time-baked RHT matrix, this
is numerically identical to before and remains CUDA-graph-safe.

## Reproducing

```bash
# Single-process compile avoids the flaky Inductor subprocess pool in some envs.
TORCHINDUCTOR_COMPILE_THREADS=1 python repro_rht_baked_in_graph.py
```

`repro_rht_baked_in_graph.py` captures Inductor's `output_code` and reports
whether the RHT construction landed in the compiled `call()`. It exits non-zero
when the regression is present, so it can be used as a check.

| Tree | Output | Exit |
|------|--------|------|
| Before fix | `RHT matrix rebuilt inside the compiled forward call(): True`  | 1 |
| After fix  | `RHT matrix rebuilt inside the compiled forward call(): False` | 0 |

## Verification of the fix

- `TORCH_LOGS=aot_graphs`: the `aten.mm` / `nvfp4_tensor.py:101-103` nodes and the
  `lru_cache` warning at `linear.py` are gone.
- `TORCH_LOGS=output_code`: the `extern_kernels.mm` and eye/hadamard triton kernels
  are gone from `call()`.
- `tests/pytorch/test_torch_compile.py -k te_linear` — 21 passed (all recipes,
  default + `reduce-overhead`/CUDA-graph modes), including the compiled-vs-eager
  numerical equivalence checks for all three NVFP4 recipes.

## Notes

- Investigated on branch `linear_compile` = PR
  [pggPL/TransformerEngine#9](https://github.com/pggPL/TransformerEngine/pull/9)
  ("[PyTorch] Add torch.compile custom-op path for Linear").
- Only the default `NVFP4BlockScaling()` enables RHT; `nvfp4_4over6` and
  `nvfp4_row_scaled` disable it, so they are unaffected.
- Unrelated environment note: `flash-attn-4`'s `quack`/`cutlass` are incompatible
  in this container (`cute.core.ThrMma` missing), which blocks `import
  transformer_engine` until the FA4 import in
  `attention/dot_product_attention/backends.py` is made to degrade gracefully.
  That is a local dev workaround, not part of this fix.
