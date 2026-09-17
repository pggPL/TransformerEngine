# Sequential operations: CPU overhead

`benchmark_ops_cpu_overhead.py` compares eager execution and `torch.compile`
for one Linear+Bias, a stack of independent pairs, pairs in one operations
group, and MLP blocks. Each MLP uses two projections and a native PyTorch
GELU with the tanh approximation. This allows measuring a larger compiled
graph without requiring compile support in TE's activation operations.

Run each configuration in a fresh process on an otherwise idle GPU and CPU:

```bash
python benchmarks/linear/benchmark_ops_cpu_overhead.py \
  --workload mlp --layers 4 --tokens 32 --width 1024 \
  --format fp8 --pass-kind train --mode compile --output mlp4-fp8-compile.json
```

Repeat with `--mode eager`. `--format` accepts `bf16`, `fp16`, and `fp8`;
FP8 uses CurrentScaling with BF16 inputs and weights. `--implementation`
selects `ops` (default), `module` (`te.Linear`), or `torch`
(`torch.nn.Linear`; BF16/FP16 only). `--workload stack` keeps each pair in
its own operations group; `chain` puts every pair in one group, exercising
quantized intermediate outputs. Unsupported configurations are recorded as
errors rather than assigned performance numbers.

Compilation defaults to `fullgraph=True` and default Inductor mode.
`--allow-graph-breaks` permits fallback for baseline comparisons.
`--compile-mode reduce-overhead` requests CUDA graph optimizations; inspect
the logs to determine whether capture was actually supported.
`--eager-warmup` initializes the model before compilation; this is an
explicitly separate experiment from the default cold-model startup.
CUDA graph mode marks iteration boundaries and releases correctness-check
outputs before the next call.

Before timing, the script compares outputs and, for training, gradients
against an independent eager copy. Training measures forward plus
`autograd.grad` for input and every parameter, without an optimizer.
The script records startup calls, warms up, then reports batch medians.
Graph counts must remain stable during timed batches.

Recorded timings are microseconds per call:

- `process_us`: CPU time summed over process threads, including autograd.
- `thread_us`: main thread CPU time.
- `host_us`: elapsed submission time.
- `complete_us`: elapsed time including final CUDA synchronization.

CUDA synchronization surrounds each batch and is excluded from the first
three metrics. Large GPU workloads can cause submission backpressure;
CPU time is not a direct measurement of kernel execution time. The main
thread is pinned before importing PyTorch so worker threads inherit the
affinity. Use `--cpu` to choose an available core. PyTorch and Inductor
worker counts are set to one by default.

For native sampling, build PyTorch and TE with
`-fno-omit-frame-pointer` (including NVCC host compilation), start Python
with `-X perf`, and supply `--profile-prefix /tmp/case`. After warmup and
timing, `/tmp/case.ready` contains the PID. Attach `perf record` to that PID,
then create `/tmp/case.go`. The script runs the steady workload for
`--profile-seconds` and writes `/tmp/case.done` on completion.
Use unique prefixes. `--cprofile file.pstats` provides separate Python
instrumentation. Neither profile should be used as an uninstrumented
latency measurement.
