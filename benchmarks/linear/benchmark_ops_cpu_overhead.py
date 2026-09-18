# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CPU submission cost of Linear, stacks of Linear, and larger MLPs.

Run each configuration in a fresh process. See --help. Optional perf/cProfile
collection happens after correctness checks, compilation, and timed batches.
Profiling results must not be interpreted as uninstrumented timings.
"""

import argparse
import contextlib
import copy
import gc
import json
import os
from pathlib import Path
import statistics
import time
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("ops", "module", "torch"), default="ops")
    parser.add_argument("--workload", choices=("linear", "stack", "chain", "mlp"), default="linear")
    parser.add_argument("--mode", choices=("eager", "compile"), default="eager")
    parser.add_argument("--format", choices=("bf16", "fp16", "fp8"), default="bf16")
    parser.add_argument("--pass-kind", choices=("forward", "train"), default="forward")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--allow-graph-breaks", action="store_true")
    parser.add_argument(
        "--eager-warmup", action="store_true", help="Initialize the model before compile"
    )
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead"), default="default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-seconds", type=float, default=15.0)
    parser.add_argument(
        "--profile-prefix", type=Path, help="Write .ready; wait for .go before sampling"
    )
    parser.add_argument("--cprofile", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    # Set affinity before importing torch so its worker threads inherit it.
    os.sched_setaffinity(0, {args.cpu})
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    import torch  # pylint: disable=import-outside-toplevel
    import transformer_engine.pytorch as te  # pylint: disable=import-outside-toplevel
    from transformer_engine.common.recipe import (  # pylint: disable=import-outside-toplevel
        Float8CurrentScaling,
    )
    from torch._dynamo.utils import counters  # pylint: disable=import-outside-toplevel

    torch.set_num_threads(1)
    torch.manual_seed(1234)
    counters.clear()
    graphs = []
    result = {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()}
    result.update(
        torch=torch.__version__,
        torch_source=torch.__file__,
        te_source=te.__file__,
        gpu=torch.cuda.get_device_name(),
        status="running",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def backend(gm, inputs):
        graphs.append(gm)
        options = torch._inductor.list_mode_options(args.compile_mode)
        return torch._dynamo.lookup_backend("inductor")(gm, inputs, config_patches=options)

    try:
        if args.implementation == "torch" and args.format == "fp8":
            raise ValueError("The native torch reference does not implement FP8")
        dtype = torch.float16 if args.format == "fp16" else torch.bfloat16

        def linear(in_features, out_features):
            if args.implementation == "ops":
                return te.ops.Sequential(
                    te.ops.BasicLinear(in_features, out_features, dtype=dtype),
                    te.ops.Bias(out_features, dtype=dtype),
                )
            if args.implementation == "module":
                return te.Linear(in_features, out_features, params_dtype=dtype)
            return torch.nn.Linear(in_features, out_features, device="cuda", dtype=dtype)

        if args.workload == "linear":
            model = linear(args.width, args.width)
        elif args.workload == "stack":
            model = torch.nn.Sequential(
                *(linear(args.width, args.width) for _ in range(args.layers))
            )
        elif args.workload == "chain":
            if args.implementation != "ops":
                raise ValueError("chain denotes multiple Linear+Bias pairs in a single ops group")
            model = te.ops.Sequential(
                *[
                    op
                    for _ in range(args.layers)
                    for op in (
                        te.ops.BasicLinear(args.width, args.width, dtype=dtype),
                        te.ops.Bias(args.width, dtype=dtype),
                    )
                ]
            )
        else:
            # Native GELU is intentional: it is traceable without adding compile
            # support to TE's activation ops, and exposes inter-op compiler work.
            model = torch.nn.Sequential(
                *[
                    torch.nn.Sequential(
                        linear(args.width, args.width * args.expansion),
                        torch.nn.GELU(approximate="tanh"),
                        linear(args.width * args.expansion, args.width),
                    )
                    for _ in range(args.layers)
                ]
            )
        reference_model = copy.deepcopy(model)
        training = args.pass_kind == "train"
        x = torch.randn(args.tokens, args.width, device="cuda", dtype=dtype, requires_grad=training)
        reference_x = x.detach().clone().requires_grad_(training)
        dy = torch.randn_like(x)
        targets = (x, *model.parameters())
        reference_targets = (reference_x, *reference_model.parameters())
        quant_recipe = Float8CurrentScaling() if args.format == "fp8" else None

        def forward(value, module=model):
            if quant_recipe is None:
                return module(value)
            with te.autocast(recipe=quant_recipe):
                return module(value)

        if args.eager_warmup:
            with contextlib.nullcontext() if training else torch.no_grad():
                warmup_output = forward(x)
                if training:
                    torch.autograd.grad(warmup_output, targets, dy)
                del warmup_output
            torch.cuda.synchronize()

        fn = (
            torch.compile(
                forward,
                backend=backend,
                fullgraph=not args.allow_graph_breaks,
            )
            if args.mode == "compile"
            else forward
        )

        def step():
            if args.mode == "compile" and args.compile_mode == "reduce-overhead":
                torch.compiler.cudagraph_mark_step_begin()
            y = fn(x)
            if training:
                return torch.autograd.grad(y, targets, dy)
            return y

        with contextlib.nullcontext() if training else torch.no_grad():
            startup = []
            for _ in range(5):
                torch.cuda.synchronize()
                if args.mode == "compile" and args.compile_mode == "reduce-overhead":
                    torch.compiler.cudagraph_mark_step_begin()
                start = time.perf_counter_ns()
                actual = fn(x)
                grads = torch.autograd.grad(actual, targets, dy) if training else None
                host_ms = (time.perf_counter_ns() - start) / 1e6
                torch.cuda.synchronize()
                startup.append(dict(host_ms=host_ms, graphs=len(graphs)))
                expected = forward(reference_x, reference_model)
                torch.testing.assert_close(actual, expected)
                if training:
                    expected_grads = torch.autograd.grad(expected, reference_targets, dy)
                    torch.testing.assert_close(grads, expected_grads)
                    del expected_grads
                del actual, expected, grads
            result["correctness"] = (
                "passed: independent eager model, outputs and requested gradients"
            )
            result["startup"] = startup
            deadline = time.monotonic() + args.warmup_seconds
            while time.monotonic() < deadline:
                step()
            torch.cuda.synchronize()
            graphs_before = len(graphs)
            gc.collect()
            gc.disable()
            measurements = {
                key: [] for key in ("process_us", "thread_us", "host_us", "complete_us")
            }
            for _ in range(args.samples):
                torch.cuda.synchronize()
                wall_start = time.perf_counter_ns()
                process_start = time.process_time_ns()
                thread_start = time.thread_time_ns()
                for _ in range(args.iterations):
                    step()
                thread_ns = time.thread_time_ns() - thread_start
                process_ns = time.process_time_ns() - process_start
                wall_ns = time.perf_counter_ns() - wall_start
                torch.cuda.synchronize()
                complete_ns = time.perf_counter_ns() - wall_start
                for key, value in zip(measurements, (process_ns, thread_ns, wall_ns, complete_ns)):
                    measurements[key].append(value / args.iterations / 1000)
            gc.enable()
            result.update(
                {
                    key: dict(median=statistics.median(values), samples=values)
                    for key, values in measurements.items()
                }
            )
            result["stable_graph_count"] = len(graphs) == graphs_before
            result["graphs"] = len(graphs)
            if not result["stable_graph_count"]:
                raise RuntimeError("Recompilation occurred during timing")
            result["status"] = "ok"
            args.output.write_text(json.dumps(result, indent=2))
            if args.profile_prefix:
                prefix = str(args.profile_prefix)
                Path(prefix + ".ready").write_text(str(os.getpid()))
                deadline = time.monotonic() + 120
                while not Path(prefix + ".go").exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("Profiler did not signal .go within 120 seconds")
                    time.sleep(0.05)
                iterations = 0
                result["profile_start_monotonic_ns"] = time.monotonic_ns()
                deadline = time.monotonic() + args.profile_seconds
                while time.monotonic() < deadline:
                    for _ in range(100):
                        step()
                    iterations += 100
                result["profile_end_monotonic_ns"] = time.monotonic_ns()
                torch.cuda.synchronize()
                result["profile_iterations"] = iterations
                Path(prefix + ".done").touch()
            if args.cprofile:
                import cProfile  # pylint: disable=import-outside-toplevel

                profiler = cProfile.Profile()
                profiler.enable()
                for _ in range(1000):
                    step()
                profiler.disable()
                torch.cuda.synchronize()
                profiler.dump_stats(str(args.cprofile))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result.update(status="error", error=str(exc), traceback=traceback.format_exc())
    finally:
        gc.enable()
        result["graph_breaks"] = dict(counters["graph_break"])
        result["inductor_counters"] = dict(counters["inductor"])
        result["custom_ops"] = sorted(
            {
                str(node.target)
                for gm in graphs
                for sub in gm.modules()
                if isinstance(sub, torch.fx.GraphModule)
                for node in sub.graph.nodes
                if "transformer_engine_compile" in str(node.target)
            }
        )
        args.output.write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in ("status", "workload", "mode", "format", "process_us", "error")
                }
            ),
            flush=True,
        )
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
