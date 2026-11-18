# KimiK3Layer with expert parallelism

A forward + backward benchmark of one `KimiK3Layer`: Kimi Delta Attention (or NoPE
multi-latent attention) and the latent MoE with a shared expert. Routed experts are sharded
across GPUs using NCCL EP. Timed iterations run inside a `torch.cuda.profiler` window.

## Quick start

Requires Blackwell GPUs (SM100+) with NVLink for the KDA kernel and NCCL EP;
see [requirements](#requirements). From this directory:

```bash
bash run_kimi_k3_layer_ep.sh --kimi                     # KDA layer, bf16
bash run_kimi_k3_layer_ep.sh --kimi --attention mla     # MLA layer, bf16
NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 bash run_kimi_k3_layer_ep.sh --kimi --recipe mxfp8
```

## Requirements

- SM100-class GPUs connected with NVLink: the KDA recurrence runs through the cuDNN frontend
  KDA kernel (`nvidia-cudnn-frontend` with the CuTe DSL runtime), the experts through NCCL EP.
- NCCL >= 2.30.4, PyTorch >= 2.11 (symmetric memory), Transformer Engine built with NCCL EP
  enabled (`NVTE_WITH_NCCL_EP`, on by default when an SM90+ architecture is targeted).
- `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` additionally needs the CuTe DSL (`nvidia-cutlass-dsl`)
  for the fused MXFP8 grouped MLP.

## Running

```bash
bash run_kimi_k3_layer_ep.sh                              # small dims, KDA, bf16
bash run_kimi_k3_layer_ep.sh --kimi                       # Kimi-K3 layer dims
bash run_kimi_k3_layer_ep.sh --kimi --attention mla       # the MLA layer (every 4th layer)
bash run_kimi_k3_layer_ep.sh --kimi --recipe mxfp8        # MXFP8 experts and projections
NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 bash run_kimi_k3_layer_ep.sh --kimi --recipe mxfp8
NSYS=1 bash run_kimi_k3_layer_ep.sh --kimi                # nsys report in results/
```

The benchmark times layer 2 of the model (the first MoE layer): with no committed block
residuals the AttnRes pooling is the identity, so the trace holds the attention block and
the latent MoE. Every rank owns `--num-local-experts` experts (default 16), so the expert
count is `16 * world_size`; the full model has 896. Other knobs: `--tokens-per-rank`,
`--topk`, `--hidden`, `--num-heads`, `--head-dim`, `--latent`, `--moe-ffn`, `--shared-ffn`,
the MLA dims, `--warmup`, `--iters`. The quantile-balancing bias is updated once after the
warmup iterations from the pooled histograms of all ranks.

## Results

See the `Kimi-K3` page of the documentation (`docs/examples/models/kimi_k3.rst`) for the
measurements and the redrawn traces.
