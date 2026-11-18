# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Kimi-K3 transformer layer."""

from typing import Optional, Tuple, Union

import torch

from transformer_engine.pytorch.module import RMSNorm
from transformer_engine.pytorch.models.deepseek_v3.multi_latent_attention import (
    MultiLatentAttention,
)
from transformer_engine.pytorch.models.deepseek_v3.moe import make_glu_mlp
from transformer_engine.pytorch.models.kimi_k3.kimi_linear_attention import KimiLinearAttention
from transformer_engine.pytorch.models.kimi_k3.moe import KimiK3MoE

__all__ = ["KimiK3Layer", "attn_res_pool"]


def attn_res_pool(
    prefix_sum: torch.Tensor,
    block_residuals: Optional[torch.Tensor],
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Attention Residuals (AttnRes) softmax pooling.

    Pools the committed block residuals ``[..., K, D]`` and the running
    ``prefix_sum`` ``[..., D]`` with weights ``softmax_k(RMSNorm(v_k) . (norm_weight * proj_weight))``,
    computed in float32. With no block residuals the pooling is the identity.
    """
    if block_residuals is None:
        return prefix_sum
    values = torch.cat([block_residuals, prefix_sum.unsqueeze(-2)], dim=-2).float()
    keys = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + eps)
    scores = (keys * (norm_weight.float() * proj_weight.float())).sum(dim=-1)
    probs = torch.softmax(scores, dim=-1)
    return (probs.unsqueeze(-1) * values).sum(dim=-2).to(prefix_sum.dtype)


class KimiK3Layer(torch.nn.Module):
    """
    A full Kimi-K3 transformer layer.

    Attention is either :class:`KimiLinearAttention` (KDA) or NoPE
    :class:`MultiLatentAttention` with an output gate; the MLP is either a
    dense SiTU-GLU MLP (first layer) or :class:`KimiK3MoE`. The residual stream
    is the Attention Residuals (AttnRes) highway: every
    ``attn_res_block_size`` layers the running prefix sum is committed as a
    block residual and the stream restarts at the attention output; both
    sub-layers read a softmax pooling of the committed blocks and the prefix
    sum instead of the plain residual.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    num_attention_heads : int
                         number of attention heads.
    layer_number : int
                  1-based index of the layer in the model; decides whether this
                  layer commits a block residual.
    attention_type : str, default = "kda"
                    ``"kda"`` or ``"mla"``.
    ffn_hidden_size : int, optional
                     ffn size of the dense MLP (used when ``num_experts`` is ``None``).
    num_experts : int, optional
                 number of routed experts; ``None`` makes this a dense layer.
    moe_ffn_hidden_size : int, optional
                         ffn size of each routed expert (required with MoE).
    attn_res_block_size : int, default = 12
                         number of layers per AttnRes block.
    layernorm_epsilon : float, default = 1e-5
                       epsilon of the RMSNorms and AttnRes pooling.
    **kwargs
             kwargs forwarded to :class:`KimiLinearAttention` (``head_dim``,
             ``conv_kernel_size``, ``gate_lower_bound``), NoPE
             :class:`MultiLatentAttention` (``q_lora_rank``, ``kv_lora_rank``,
             ``qk_nope_head_dim``, ``qk_rope_head_dim``, ``v_head_dim``,
             ``attention_dropout``, ``attn_mask_type``, ``softmax_scale``,
             ``tp_group``, ``tp_size``) and :class:`KimiK3MoE` (``topk``,
             ``latent_size``, ``shared_expert_ffn_hidden_size``,
             ``routed_scaling_factor``, ``expert_bias_update``, ``qb_num_bins``,
             EP options); ``qkv_format`` applies to both attention types.
    """

    _KDA_KWARGS = frozenset({"head_dim", "conv_kernel_size", "gate_lower_bound", "qkv_format"})
    _MLA_KWARGS = frozenset(
        {
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "attention_dropout",
            "attn_mask_type",
            "softmax_scale",
            "qkv_format",
            "tp_group",
            "tp_size",
        }
    )
    _MOE_KWARGS = frozenset(
        {
            "topk",
            "latent_size",
            "shared_expert_ffn_hidden_size",
            "routed_scaling_factor",
            "expert_bias_update",
            "qb_num_bins",
            "ep_group",
            "ep_max_tokens_per_rank",
        }
    )

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        layer_number: int,
        attention_type: str = "kda",
        ffn_hidden_size: Optional[int] = None,
        num_experts: Optional[int] = None,
        moe_ffn_hidden_size: Optional[int] = None,
        attn_res_block_size: int = 12,
        layernorm_epsilon: float = 1e-5,
        params_dtype: Optional[torch.dtype] = None,
        device: Union[torch.device, str] = "cuda",
        **kwargs,
    ) -> None:
        super().__init__()
        if attention_type not in ("kda", "mla"):
            raise ValueError("attention_type must be 'kda' or 'mla'.")
        if layer_number < 1:
            raise ValueError("layer_number is 1-based.")
        attn_kwargs_names = self._KDA_KWARGS if attention_type == "kda" else self._MLA_KWARGS
        unknown = set(kwargs) - attn_kwargs_names - self._MOE_KWARGS
        if unknown:
            raise TypeError(f"Unexpected keyword arguments: {sorted(unknown)}")
        attn_kwargs = {k: v for k, v in kwargs.items() if k in attn_kwargs_names}
        moe_kwargs = {k: v for k, v in kwargs.items() if k in self._MOE_KWARGS}

        dtype = params_dtype if params_dtype is not None else torch.get_default_dtype()
        self.layer_number = layer_number
        self.attention_type = attention_type
        self.commits_block_residual = (layer_number - 1) % attn_res_block_size == 0
        self.layernorm_epsilon = layernorm_epsilon

        norm = {"eps": layernorm_epsilon, "device": device, "dtype": params_dtype}
        self.input_layernorm = RMSNorm(hidden_size, **norm)
        self.post_attention_layernorm = RMSNorm(hidden_size, **norm)
        # The first layer has no block residuals to pool before attention.
        self.attention_res_norm_weight = None
        self.attention_res_proj = None
        if layer_number > 1:
            self.attention_res_norm_weight = torch.nn.Parameter(
                torch.ones(hidden_size, dtype=dtype, device=device)
            )
            self.attention_res_proj = torch.nn.Linear(
                hidden_size, 1, bias=False, dtype=dtype, device=device
            )
        self.mlp_res_norm_weight = torch.nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device)
        )
        self.mlp_res_proj = torch.nn.Linear(hidden_size, 1, bias=False, dtype=dtype, device=device)

        if attention_type == "kda":
            self.self_attention = KimiLinearAttention(
                hidden_size,
                num_attention_heads,
                layernorm_epsilon=layernorm_epsilon,
                params_dtype=params_dtype,
                device=device,
                **attn_kwargs,
            )
        else:
            self.self_attention = MultiLatentAttention(
                hidden_size,
                num_attention_heads,
                rope=False,
                output_gate=True,
                params_dtype=params_dtype,
                device=device,
                **attn_kwargs,
            )

        if num_experts is None:
            assert ffn_hidden_size is not None, "Dense layers require ffn_hidden_size."
            self.mlp = make_glu_mlp(
                hidden_size, ffn_hidden_size, dtype, device, activation="situglu"
            )
        else:
            assert moe_ffn_hidden_size is not None, "MoE layers require moe_ffn_hidden_size."
            self.mlp = KimiK3MoE(
                hidden_size,
                moe_ffn_hidden_size,
                num_experts,
                layernorm_epsilon=layernorm_epsilon,
                params_dtype=params_dtype,
                device=device,
                **moe_kwargs,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        block_residuals: Optional[torch.Tensor] = None,
        ep_buffer=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor
                       AttnRes prefix sum of shape ``[sq, b, h]`` (sbhd) or ``[b, sq, h]`` (bshd).
        attention_mask : torch.Tensor, optional
                        boolean attention mask (MLA layers only).
        block_residuals : torch.Tensor, optional
                         committed block residuals of shape ``[..., K, h]`` from the
                         previous layer; ``None`` at the first layer.
        ep_buffer : EpBuffer, optional
                   forwarded to :class:`KimiK3MoE` (MoE layers with EP).

        Returns
        -------
        hidden_states : torch.Tensor
                       updated prefix sum, input of the next layer.
        block_residuals : torch.Tensor, optional
                         block residuals to pass to the next layer.
        """
        prefix_sum = hidden_states
        if self.attention_res_proj is None:
            assert block_residuals is None, "The first layer takes no block residuals."
            attention_input = prefix_sum
        else:
            attention_input = attn_res_pool(
                prefix_sum,
                block_residuals,
                self.attention_res_proj.weight[0],
                self.attention_res_norm_weight,
                self.layernorm_epsilon,
            )
        if self.commits_block_residual:
            new_block = prefix_sum.unsqueeze(-2)
            block_residuals = (
                new_block
                if block_residuals is None
                else torch.cat([block_residuals, new_block], dim=-2)
            )

        attention_input = self.input_layernorm(attention_input)
        if self.attention_type == "kda":
            attention_out = self.self_attention(attention_input)
        else:
            attention_out = self.self_attention(attention_input, attention_mask=attention_mask)
        prefix_sum = attention_out if self.commits_block_residual else prefix_sum + attention_out

        mlp_input = attn_res_pool(
            prefix_sum,
            block_residuals,
            self.mlp_res_proj.weight[0],
            self.mlp_res_norm_weight,
            self.layernorm_epsilon,
        )
        mlp_input = self.post_attention_layernorm(mlp_input)
        if ep_buffer is not None:
            mlp_out = self.mlp(mlp_input, ep_buffer=ep_buffer)
        else:
            mlp_out = self.mlp(mlp_input)
        return prefix_sum + mlp_out, block_residuals
