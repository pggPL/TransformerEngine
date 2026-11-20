# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.ops.fused.grouped_mlp import (
    GroupedMLP_CuTeGEMMGLU,
    fuse_glu_ops,
)
from transformer_engine.pytorch.ops.fuser import OperationFuser
from transformer_engine.pytorch.utils import deinterleave_glu_tensor
from transformer_engine.pytorch.models import (
    DeepSeekV3MoE,
    KimiK3Layer,
    KimiK3MoE,
    KimiLinearAttention,
    MultiLatentAttention,
)
from transformer_engine.pytorch.models.deepseek_v3.moe import expert_bias_from_qb_histogram
from transformer_engine.pytorch.models.kimi_k3 import attn_res_pool
from transformer_engine.pytorch.models.kimi_k3.kimi_linear_attention import (
    _causal_depthwise_conv1d,
    _gated_rmsnorm,
)
from transformer_engine.pytorch.attention.linear_attention import kda_block_kernels
from utils import make_recipe, quantization_tols

SEQ_LEN = 128
BATCH = 2
HIDDEN = 256
HEADS = 4
DTYPE = torch.bfloat16

MLA_KWARGS = dict(
    q_lora_rank=96,
    kv_lora_rank=64,
    qk_nope_head_dim=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
)


def _input(requires_grad=True):
    torch.manual_seed(1234)
    return torch.randn(
        SEQ_LEN, BATCH, HIDDEN, dtype=DTYPE, device="cuda", requires_grad=requires_grad
    )


@pytest.mark.parametrize("mscale_all_dim", [0.0, 1.0])
def test_mla_yarn_softmax_scale(mscale_all_dim):
    mla = MultiLatentAttention(
        HIDDEN,
        HEADS,
        params_dtype=DTYPE,
        rope_scaling_factor=40.0,
        original_max_position_embeddings=64,
        mscale_all_dim=mscale_all_dim,
        **MLA_KWARGS,
    )
    m = 0.1 * mscale_all_dim * math.log(40.0) + 1.0
    qk_head_dim = MLA_KWARGS["qk_nope_head_dim"] + MLA_KWARGS["qk_rope_head_dim"]
    assert mla.softmax_scale == pytest.approx(m * m / math.sqrt(qk_head_dim))


@pytest.mark.parametrize("shared", [False, True], ids=["no_shared", "shared"])
@pytest.mark.parametrize("grouped", [False, True], ids=["ungrouped", "grouped"])
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("quantization", [None, "fp8_block_scaling"])
def test_moe_matches_dense_reference(shared, grouped, topk, quantization):
    """Routed output must equal the prob-weighted sum of the selected expert MLPs."""
    if quantization is not None:
        available, reason = te.is_fp8_block_scaling_available(return_reason=True)
        if not available:
            pytest.skip(reason)
    torch.manual_seed(0)
    num_experts = 4
    moe = DeepSeekV3MoE(
        HIDDEN,
        moe_ffn_hidden_size=128,
        num_experts=num_experts,
        topk=topk,
        num_groups=2 if grouped else None,
        group_topk=topk // 2 if grouped else None,
        shared_expert_ffn_hidden_size=128 if shared else None,
        params_dtype=DTYPE,
    )
    x = _input()
    with te.autocast(enabled=quantization is not None, recipe=make_recipe(quantization)):
        out = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    for name, parameter in moe.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name

    tokens = x.detach().reshape(-1, HIDDEN)
    probs, _ = moe._route(moe.gate(tokens).float())
    assert (probs > 0).sum(dim=1).eq(topk).all()
    assert moe._last_tokens_per_expert.sum().item() == tokens.shape[0] * topk

    fc1, _, fc2 = moe.experts
    quantizers = [
        te.Float8BlockQuantizer(
            fp8_dtype=te.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            block_scaling_dim=dim,
        )
        for dim in (1, 2)
    ]

    def qdq(tensor, block_dim):
        if quantization is None:
            return tensor
        return quantizers[block_dim - 1](tensor).dequantize(dtype=DTYPE)

    ref_tokens = qdq(tokens, 1)
    ref = torch.zeros_like(tokens)
    for e in range(num_experts):
        w1 = deinterleave_glu_tensor(qdq(getattr(fc1, f"weight{e}"), 2), 32)
        w2 = qdq(getattr(fc2, f"weight{e}"), 2)
        gate_part, lin_part = (ref_tokens @ w1.t()).chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate_part.float()) * lin_part.float()
        act = qdq(act.to(DTYPE) * probs[:, e : e + 1].to(DTYPE), 1)
        ref += act @ w2.t()
    if shared:
        with te.autocast(enabled=quantization is not None, recipe=make_recipe(quantization)):
            ref += moe.shared_expert(tokens)
    torch.testing.assert_close(out.reshape(-1, HIDDEN), ref, rtol=0.05, atol=0.05)

    bias_before = moe.expert_bias.clone()
    moe.update_expert_bias()
    assert torch.isfinite(moe.expert_bias).all()
    if topk < num_experts:
        assert not torch.equal(bias_before, moe.expert_bias)


@pytest.mark.parametrize("quantization", ["mxfp8", "nvfp4_rht"])
def test_moe_fused_quantized_uneven_expert_rows(monkeypatch, quantization):
    available_fn = te.is_mxfp8_available if quantization == "mxfp8" else te.is_nvfp4_available
    available, reason = available_fn(return_reason=True)
    if not available:
        pytest.skip(reason)

    recipe = make_recipe(quantization)
    torch.manual_seed(0)
    moe = DeepSeekV3MoE(HIDDEN, 128, num_experts=2, topk=1, params_dtype=DTYPE)
    fused_ops = fuse_glu_ops(list(moe.experts), recipe=recipe)
    if (
        fuse_glu_ops not in OperationFuser.forward_backward_fusion_functions
        or len(fused_ops) != 1
        or not isinstance(fused_ops[0], GroupedMLP_CuTeGEMMGLU)
    ):
        pytest.skip("requires the fused grouped MLP")
    reference = DeepSeekV3MoE(HIDDEN, 128, num_experts=2, topk=1, params_dtype=DTYPE)
    reference.load_state_dict(moe.state_dict())
    with torch.no_grad():
        for module in (moe, reference):
            module.gate.weight.zero_()
            module.gate.weight[0, 0] = 1
            module.gate.weight[1, 0] = -1

    tokens = torch.empty(512, HIDDEN, device="cuda", dtype=DTYPE).uniform_(-0.25, 0.25)
    tokens[:128, 0] = 2
    tokens[128:, 0] = -2
    x = tokens.detach().requires_grad_()
    x_ref = tokens.detach().clone().requires_grad_()
    grad = torch.empty_like(tokens).uniform_(-0.25, 0.25)
    splits = []
    moe.experts.register_forward_pre_hook(lambda _module, args: splits.append(args[1].clone()))

    with te.autocast(enabled=True, recipe=recipe):
        out = moe(x)
    out.backward(grad.clone())
    assert torch.equal(splits[0], torch.tensor([256, 512], device="cuda"))
    fused_op = moe.experts._module_groups[0]._forward_ops[0][0]
    assert isinstance(fused_op, GroupedMLP_CuTeGEMMGLU)

    monkeypatch.setattr(
        OperationFuser,
        "forward_backward_fusion_functions",
        [fn for fn in OperationFuser.forward_backward_fusion_functions if fn is not fuse_glu_ops],
    )
    ref_splits = []
    reference.experts.register_forward_pre_hook(
        lambda _module, args: ref_splits.append(args[1].clone())
    )
    with te.autocast(enabled=True, recipe=make_recipe(quantization)):
        ref_out = reference(x_ref)
    ref_out.backward(grad.clone())
    assert torch.equal(ref_splits[0], splits[0])

    tols = quantization_tols(quantization)
    torch.testing.assert_close(out, ref_out, **tols)
    torch.testing.assert_close(x.grad, x_ref.grad, **tols)
    ref_params = dict(reference.named_parameters())
    for name, param in moe.named_parameters():
        torch.testing.assert_close(param.grad, ref_params[name].grad, **tols)


def _rmsnorm(x, weight, eps):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(
        weight.dtype
    )


def _situ_glu(gate, up):
    gate, up = gate.float(), up.float()
    return 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * 25.0 * torch.tanh(up / 25.0)


def _kda_reference_core(core):
    """Pure PyTorch replacement for the cuDNN KDA kernel (sbhd/bshd, in-kernel l2norm,
    beta sigmoid and safe gate), for GPUs without the kernel."""

    def forward(q, k, v, g, beta, *, a_log, dt_bias, gate_lower_bound, **_):
        if core.qkv_format == "sbhd":
            q, k, v, g, beta = (t.transpose(0, 1) for t in (q, k, v, g, beta))
        q = torch.nn.functional.normalize(q.float(), dim=-1) * core.scale
        k = torch.nn.functional.normalize(k.float(), dim=-1)
        v = v.float()
        alpha = torch.exp(
            gate_lower_bound * torch.sigmoid(a_log.exp()[:, None] * (g.float() + dt_bias))
        )
        beta = torch.sigmoid(beta.float())
        batch, seq, heads, dim = q.shape
        state = q.new_zeros(batch, heads, dim, v.shape[-1])
        outs = []
        for t in range(seq):
            state = alpha[:, t, :, :, None] * state
            pred = torch.einsum("bhk,bhkv->bhv", k[:, t], state)
            v_new = beta[:, t, :, None] * (v[:, t] - pred)
            state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], v_new)
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
        out = torch.stack(outs, dim=1).reshape(batch, seq, -1)
        if core.qkv_format == "sbhd":
            out = out.transpose(0, 1)
        return out.to(v.dtype if v.dtype != torch.float32 else torch.bfloat16)

    return forward


def _kda_kernel_available():
    try:
        from cudnn.linear_attention.ops import kimi_delta_attention  # noqa: F401
    except (AttributeError, ImportError):
        return False
    return torch.cuda.get_device_capability() in {(10, 0), (10, 3), (10, 7)}


def _use_reference_kda(module):
    if _kda_kernel_available():
        return
    for core in [m for m in module.modules() if isinstance(m, te.KimiDeltaAttention)]:
        core.forward = _kda_reference_core(core)


def _finite_grads(module):
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_attn_res_pool_matches_reference():
    torch.manual_seed(0)
    eps = 1e-5
    prefix = torch.randn(SEQ_LEN, BATCH, HIDDEN, dtype=DTYPE, device="cuda")
    blocks = torch.randn(SEQ_LEN, BATCH, 3, HIDDEN, dtype=DTYPE, device="cuda")
    proj = torch.randn(HIDDEN, dtype=DTYPE, device="cuda") * 0.1
    norm = torch.rand(HIDDEN, dtype=DTYPE, device="cuda") + 0.5

    assert attn_res_pool(prefix, None, proj, norm, eps) is prefix

    values = torch.cat([blocks, prefix.unsqueeze(-2)], dim=-2).float()
    scores = torch.stack(
        [_rmsnorm(values[..., i, :], norm.float(), eps) @ proj.float() for i in range(4)],
        dim=-1,
    )
    ref = torch.einsum("...k,...kd->...d", torch.softmax(scores, -1), values).to(DTYPE)
    torch.testing.assert_close(attn_res_pool(prefix, blocks, proj, norm, eps), ref)

    uniform = attn_res_pool(prefix, blocks, torch.zeros_like(proj), norm, eps)
    torch.testing.assert_close(uniform, values.mean(-2).to(DTYPE))


@pytest.mark.parametrize("qkv_format", ["sbhd", "bshd"])
def test_kimi_linear_attention_forward_backward(qkv_format):
    torch.manual_seed(0)
    attention = KimiLinearAttention(
        HIDDEN, HEADS, head_dim=64, qkv_format=qkv_format, params_dtype=DTYPE
    )
    _use_reference_kda(attention)
    x = _input()
    if qkv_format == "bshd":
        x = x.detach().transpose(0, 1).contiguous().requires_grad_()
    out = attention(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.float().pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()
    _finite_grads(attention)
    for name in ("A_log", "dt_bias", "conv_weight", "o_norm_weight"):
        assert getattr(attention, name).dtype == torch.float32


@pytest.mark.skipif(not kda_block_kernels.HAVE_TRITON, reason="Triton is required")
@pytest.mark.parametrize("qkv_format", ["sbhd", "bshd"])
def test_kda_conv_silu_kernel_matches_reference(qkv_format):
    torch.manual_seed(0)
    seq_dim = 0 if qkv_format == "sbhd" else 1
    channels, ksize = 3 * HEADS * 64, 4
    shape = (SEQ_LEN, BATCH, channels) if seq_dim == 0 else (BATCH, SEQ_LEN, channels)
    x = torch.randn(shape, dtype=DTYPE, device="cuda", requires_grad=True)
    weight = torch.randn(channels, ksize, dtype=torch.float32, device="cuda", requires_grad=True)

    q, k, v = kda_block_kernels.causal_conv_silu_split(x, weight, seq_dim)
    ref = _causal_depthwise_conv1d(x, weight, seq_dim)
    if seq_dim == 0:
        ref = ref.transpose(0, 1)
    ref_q, ref_k, ref_v = ref.chunk(3, dim=-1)
    for out, expected in ((q, ref_q), (k, ref_k), (v, ref_v)):
        assert out.shape == (BATCH, SEQ_LEN, channels // 3)
        torch.testing.assert_close(out, expected)

    grads = [torch.randn_like(t) for t in (q, k, v)]
    torch.autograd.backward((q, k, v), grads)
    dx, dw = x.grad.clone(), weight.grad.clone()
    x.grad = None
    weight.grad = None
    torch.autograd.backward((ref_q, ref_k, ref_v), grads)
    torch.testing.assert_close(dx, x.grad)
    torch.testing.assert_close(dw, weight.grad, rtol=1e-3, atol=1e-2)


@pytest.mark.skipif(not kda_block_kernels.HAVE_TRITON, reason="Triton is required")
@pytest.mark.parametrize("qkv_format", ["sbhd", "bshd"])
def test_kda_gated_rmsnorm_kernel_matches_reference(qkv_format):
    torch.manual_seed(0)
    seq_dim = 0 if qkv_format == "sbhd" else 1
    head_dim, eps = 64, 1e-5
    o = torch.randn(BATCH, SEQ_LEN, HEADS * head_dim, dtype=DTYPE, device="cuda", requires_grad=True)
    g_shape = (SEQ_LEN, BATCH, HEADS * head_dim) if seq_dim == 0 else o.shape
    g = torch.randn(g_shape, dtype=DTYPE, device="cuda", requires_grad=True)
    weight = torch.rand(head_dim, dtype=torch.float32, device="cuda", requires_grad=True) + 0.5

    out = kda_block_kernels.gated_rmsnorm(o, g, weight, eps, seq_dim, HEADS)
    ref_o = o.transpose(0, 1) if seq_dim == 0 else o
    ref = _gated_rmsnorm(ref_o, g, weight, eps, head_dim)
    assert out.shape == g.shape
    torch.testing.assert_close(out, ref)

    dy = torch.randn_like(out)
    out.backward(dy)
    grads = [t.grad.clone() for t in (o, g, weight)]
    for t in (o, g, weight):
        t.grad = None
    ref.backward(dy)
    torch.testing.assert_close(grads[0], o.grad)
    torch.testing.assert_close(grads[1], g.grad)
    torch.testing.assert_close(grads[2], weight.grad, rtol=1e-3, atol=1e-2)


@pytest.mark.skipif(not kda_block_kernels.HAVE_TRITON, reason="Triton is required")
def test_kimi_linear_attention_fused_glue_matches_fallback():
    torch.manual_seed(0)
    attention = KimiLinearAttention(HIDDEN, HEADS, head_dim=64, params_dtype=DTYPE)
    _use_reference_kda(attention)
    assert attention.fused_glue
    x = _input()
    out = attention(x)
    out.float().pow(2).mean().backward()
    grads = [x.grad.clone()] + [p.grad.clone() for p in attention.parameters()]
    x.grad = None
    attention.zero_grad()

    attention.fused_glue = False
    ref = attention(x)
    ref.float().pow(2).mean().backward()
    torch.testing.assert_close(out, ref)
    ref_grads = [x.grad] + [p.grad for p in attention.parameters()]
    for got, expected in zip(grads, ref_grads):
        torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


def test_mla_nope_output_gate_matches_reference():
    torch.manual_seed(0)
    mla = MultiLatentAttention(
        HIDDEN, HEADS, params_dtype=DTYPE, rope=False, output_gate=True, **MLA_KWARGS
    )
    x = _input()
    out = mla(x)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    _finite_grads(mla)

    nope, rope, v_dim = (
        MLA_KWARGS[k] for k in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
    )
    eps = mla.q_up_proj.eps
    xf = x.detach()
    q_lora, kv_lora = MLA_KWARGS["q_lora_rank"], MLA_KWARGS["kv_lora_rank"]
    q_latent, latent, k_pos = torch.split(
        xf @ mla.qkv_down_proj.weight.T, [q_lora, kv_lora, rope], dim=-1
    )
    q = _rmsnorm(q_latent, mla.q_up_proj.layer_norm_weight, eps)
    q = (q @ mla.q_up_proj.weight.T).view(SEQ_LEN, BATCH, HEADS, nope + rope)
    kv = _rmsnorm(latent, mla.kv_up_proj.layer_norm_weight, eps) @ mla.kv_up_proj.weight.T
    k_nope, v = torch.split(kv.view(SEQ_LEN, BATCH, HEADS, nope + v_dim), [nope, v_dim], dim=-1)
    k = torch.cat([k_nope, k_pos.unsqueeze(-2).expand(-1, -1, HEADS, -1)], dim=-1)
    q, k, v = (t.permute(1, 2, 0, 3).float() for t in (q, k, v))
    ctx = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    ctx = ctx.permute(2, 0, 1, 3).reshape(SEQ_LEN, BATCH, HEADS * v_dim).to(DTYPE)
    ctx = ctx * torch.sigmoid(xf @ mla.gate_proj.weight.T)
    ref = ctx @ mla.out_proj.weight.T
    torch.testing.assert_close(out, ref, rtol=0.05, atol=0.05)


def test_sigmoid_gate_matches_reference():
    from transformer_engine.pytorch.triton import sigmoid_gate as sg

    if not sg.HAVE_TRITON:
        pytest.skip("requires Triton")
    torch.manual_seed(0)
    x = torch.randn(SEQ_LEN, BATCH, HIDDEN, dtype=DTYPE, device="cuda", requires_grad=True)
    gate = torch.randn(SEQ_LEN, BATCH, HIDDEN, dtype=DTYPE, device="cuda", requires_grad=True)
    dy = torch.randn(SEQ_LEN, BATCH, HIDDEN, dtype=DTYPE, device="cuda")

    out = sg.sigmoid_gate(x, gate)
    out.backward(dy)
    ref = (x.detach().float() * torch.sigmoid(gate.detach().float())).to(DTYPE)
    torch.testing.assert_close(out, ref)
    sgf = torch.sigmoid(gate.detach().float())
    torch.testing.assert_close(x.grad, (dy.float() * sgf).to(DTYPE))
    torch.testing.assert_close(
        gate.grad, (dy.float() * x.detach().float() * sgf * (1 - sgf)).to(DTYPE)
    )


@pytest.mark.parametrize("shared", [False, True], ids=["no_shared", "shared"])
def test_kimi_moe_matches_dense_reference(shared):
    torch.manual_seed(0)
    num_experts, topk, latent, ffn = 4, 2, 128, 96
    moe = KimiK3MoE(
        HIDDEN,
        ffn,
        num_experts,
        topk=topk,
        latent_size=latent,
        shared_expert_ffn_hidden_size=128 if shared else None,
        params_dtype=DTYPE,
    )
    x = _input()
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    _finite_grads(moe)

    tokens = x.detach().reshape(-1, HIDDEN)
    routed = moe.routed_experts
    assert routed.qb_histogram.sum().item() == tokens.shape[0] * num_experts
    with torch.no_grad():
        probs, _ = routed._route(routed.gate(tokens).float())
    assert (probs > 0).sum(dim=1).eq(topk).all()
    assert routed.qb_histogram.sum().item() == tokens.shape[0] * num_experts

    z = tokens @ moe.down_proj.weight.T
    fc1, _, fc2 = routed.experts
    ref = torch.zeros_like(z)
    for e in range(num_experts):
        w1 = deinterleave_glu_tensor(getattr(fc1, f"weight{e}"), 32)
        gate_part, lin_part = (z @ w1.t()).chunk(2, dim=-1)
        act = _situ_glu(gate_part, lin_part).to(DTYPE) * probs[:, e : e + 1].to(DTYPE)
        ref += act @ getattr(fc2, f"weight{e}").t()
    ref = (
        _rmsnorm(ref, moe.post_expert_norm.weight, moe.post_expert_norm.eps) @ moe.up_proj.weight.T
    )
    if shared:
        ref = ref + moe.shared_expert(tokens)
    torch.testing.assert_close(out.reshape(-1, HIDDEN), ref, rtol=0.05, atol=0.05)

    moe.update_expert_bias()
    assert torch.isfinite(moe.expert_bias).all()
    assert moe.expert_bias.mean().abs().item() < 1e-5
    assert routed.qb_histogram.sum().item() == 0
    assert routed.qb_bin_bounds[0] <= moe.expert_bias.min() - 1
    assert routed.qb_bin_bounds[1] >= moe.expert_bias.max() + 1


def test_expert_bias_from_qb_histogram_matches_quantile():
    torch.manual_seed(0)
    num_experts, topk, num_bins, tokens = 8, 2, 1000, 20000
    required = torch.randn(num_experts, tokens, device="cuda") * 0.2 + torch.linspace(
        -0.5, 0.5, num_experts, device="cuda"
    ).unsqueeze(1)
    bounds = torch.tensor([-1.0, 1.0], device="cuda")
    histogram = torch.stack(
        [torch.histc(required[e], bins=num_bins, min=-1.0, max=1.0) for e in range(num_experts)]
    ).to(torch.int32)
    old_bias = torch.zeros(num_experts, device="cuda")
    bias = expert_bias_from_qb_histogram(histogram, bounds, old_bias, topk)
    expected = torch.quantile(required, topk / num_experts, dim=1)
    expected = expected - expected.mean()
    torch.testing.assert_close(bias, expected, rtol=0, atol=3.0 / num_bins)

    histogram[0].zero_()
    old_bias = old_bias + 0.3
    bias = expert_bias_from_qb_histogram(histogram, bounds, old_bias, topk)
    raw = torch.quantile(required, topk / num_experts, dim=1)
    raw[0] = old_bias[0]
    torch.testing.assert_close(bias, raw - raw.mean(), rtol=0, atol=3.0 / num_bins)


def test_kimi_layer_stack_attn_res():
    torch.manual_seed(0)
    layers = torch.nn.ModuleList(
        [
            KimiK3Layer(
                HIDDEN,
                HEADS,
                layer_number=i + 1,
                attention_type="mla" if i == 1 else "kda",
                ffn_hidden_size=128 if i == 0 else None,
                num_experts=None if i == 0 else 4,
                moe_ffn_hidden_size=None if i == 0 else 64,
                attn_res_block_size=2,
                params_dtype=DTYPE,
                **(MLA_KWARGS if i == 1 else dict(head_dim=64)),
                **(
                    {}
                    if i == 0
                    else dict(topk=2, latent_size=128, shared_expert_ffn_hidden_size=64)
                ),
            )
            for i in range(3)
        ]
    )
    _use_reference_kda(layers)
    assert [layer.commits_block_residual for layer in layers] == [True, False, True]

    x = _input()
    hidden, blocks = x, None
    for layer in layers:
        hidden, blocks = layer(hidden, block_residuals=blocks)
        assert hidden.shape == x.shape
    assert blocks.shape == (SEQ_LEN, BATCH, 2, HIDDEN)
    hidden.float().pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()
    _finite_grads(layers)
    assert layers[0].attention_res_proj is None
    for layer in layers[1:]:
        layer.mlp.update_expert_bias()
        assert torch.isfinite(layer.mlp.expert_bias).all()


def test_mla_in_place_rope_matches_clone(monkeypatch):
    """Rotating q in place inside MultiLatentAttention must equal the out-of-place path."""
    import transformer_engine.pytorch.models.deepseek_v3.multi_latent_attention as mla_module

    torch.manual_seed(0)
    mla = MultiLatentAttention(HIDDEN, HEADS, params_dtype=DTYPE, **MLA_KWARGS)
    x = _input()
    out = mla(x)
    out.sum().backward()
    grads = [x.grad.clone()] + [p.grad.clone() for p in mla.parameters()]

    original = mla_module.apply_mla_rope_q
    monkeypatch.setattr(
        mla_module, "apply_mla_rope_q", lambda *a, **k: original(*a, **{**k, "in_place": False})
    )
    x.grad = None
    mla.zero_grad()
    ref = mla(x)
    ref.sum().backward()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    for got, want in zip([x.grad] + [p.grad for p in mla.parameters()], grads):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
