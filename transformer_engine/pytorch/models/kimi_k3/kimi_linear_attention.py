# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Kimi Delta Attention block as used in Kimi-K3."""

import math
from typing import Optional, Union

import torch

from transformer_engine.pytorch.module import Linear
from transformer_engine.pytorch.attention import KimiDeltaAttention

__all__ = ["KimiLinearAttention"]


def _causal_depthwise_conv1d(x: torch.Tensor, weight: torch.Tensor, seq_dim: int) -> torch.Tensor:
    """SiLU(causal depthwise conv over the sequence) in float32, cast back to ``x.dtype``.

    ``x`` is ``[s, b, c]`` or ``[b, s, c]``; ``weight`` is ``[c, kernel_size]``.
    """
    kernel_size = weight.shape[-1]
    x_bcs = torch.nn.functional.pad(x.float().movedim(seq_dim, -1), (kernel_size - 1, 0))
    out = torch.nn.functional.conv1d(x_bcs, weight.unsqueeze(1), groups=weight.shape[0])
    return torch.nn.functional.silu(out).movedim(-1, seq_dim).to(x.dtype)


class KimiLinearAttention(torch.nn.Module):
    """
    Kimi Delta Attention (KDA) block as used in Kimi-K3 and Kimi Linear.

    Queries, keys and values are projected, passed through a causal depthwise
    short convolution with SiLU, and fed to the
    :class:`transformer_engine.pytorch.KimiDeltaAttention` recurrence with
    in-kernel L2 normalization of queries and keys, a per-channel decay gate
    ``gate_lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))`` and a
    per-head write strength ``sigmoid(b(x))``. The output is RMS-normalized
    per head, gated by ``sigmoid(g(x))`` and projected back to the hidden size.

    The recurrence runs through the cuDNN frontend KDA kernel (Blackwell+).
    Tensor parallelism is not supported yet.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    num_attention_heads : int
                         number of attention heads.
    head_dim : int, default = 128
              per-head size of queries, keys and values (64 or 128).
    conv_kernel_size : int, default = 4
                      window of the causal short convolution.
    gate_lower_bound : float, default = -5.0
                      lower bound of the log-space decay gate.
    layernorm_epsilon : float, default = 1e-5
                       epsilon of the per-head output RMSNorm.
    qkv_format : str, default = "sbhd"
                layout of the input/output tensors, ``sbhd`` or ``bshd``.
    params_dtype : torch.dtype, optional
                  dtype of the projection parameters. The convolution weights,
                  ``A_log``, ``dt_bias`` and the output norm weight are float32.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int = 128,
        conv_kernel_size: int = 4,
        gate_lower_bound: float = -5.0,
        layernorm_epsilon: float = 1e-5,
        qkv_format: str = "sbhd",
        params_dtype: Optional[torch.dtype] = None,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()
        assert qkv_format in ("sbhd", "bshd"), "KimiLinearAttention supports sbhd/bshd formats."

        self.qkv_format = qkv_format
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.gate_lower_bound = gate_lower_bound
        self.layernorm_epsilon = layernorm_epsilon
        projection_size = num_attention_heads * head_dim

        common = {"bias": False, "params_dtype": params_dtype, "device": device}
        self.qkv_proj = Linear(hidden_size, 3 * projection_size, **common)
        self.conv_weight = torch.nn.Parameter(
            torch.empty(3 * projection_size, conv_kernel_size, dtype=torch.float32, device=device)
        )
        self.f_a_proj = Linear(hidden_size, head_dim, **common)
        self.f_b_proj = Linear(head_dim, projection_size, **common)
        self.A_log = torch.nn.Parameter(
            torch.empty(num_attention_heads, dtype=torch.float32, device=device)
        )
        self.dt_bias = torch.nn.Parameter(
            torch.empty(num_attention_heads, head_dim, dtype=torch.float32, device=device)
        )
        self.b_proj = Linear(hidden_size, num_attention_heads, **common)
        self.g_proj = Linear(hidden_size, projection_size, **common)
        self.o_norm_weight = torch.nn.Parameter(
            torch.ones(head_dim, dtype=torch.float32, device=device)
        )
        self.out_proj = Linear(projection_size, hidden_size, **common)
        self.core_attention = KimiDeltaAttention(
            num_attention_heads, head_dim, qkv_format=qkv_format
        )
        self.reset_recurrence_parameters()

    @torch.no_grad()
    def reset_recurrence_parameters(self) -> None:
        """Initialize the convolution weights, ``A_log`` and ``dt_bias``."""
        bound = 1.0 / math.sqrt(self.conv_weight.shape[-1])
        self.conv_weight.uniform_(-bound, bound)
        self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1.0, 16.0).log())
        self.dt_bias.uniform_(-1.0, 1.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor
                       input of shape ``[sq, b, h]`` (sbhd) or ``[b, sq, h]`` (bshd).
        """
        seq_dim = 0 if self.qkv_format == "sbhd" else 1
        heads, head_dim = self.num_attention_heads, self.head_dim
        token_shape = hidden_states.shape[:-1]

        qkv = _causal_depthwise_conv1d(self.qkv_proj(hidden_states), self.conv_weight, seq_dim)
        q, k, v = (t.view(*token_shape, heads, head_dim) for t in qkv.chunk(3, dim=-1))
        g = self.f_b_proj(self.f_a_proj(hidden_states)).view(*token_shape, heads, head_dim)
        beta = self.b_proj(hidden_states)

        out = self.core_attention(
            q,
            k,
            v,
            g,
            beta,
            a_log=self.A_log,
            dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            gate_lower_bound=self.gate_lower_bound,
        )

        out = out.view(*token_shape, heads, head_dim).float()
        out = out * torch.rsqrt(out.pow(2).mean(dim=-1, keepdim=True) + self.layernorm_epsilon)
        out = out * self.o_norm_weight
        gate = self.g_proj(hidden_states).view(*token_shape, heads, head_dim).float()
        out = (out * torch.sigmoid(gate)).to(hidden_states.dtype)
        return self.out_proj(out.reshape(*token_shape, heads * head_dim))
