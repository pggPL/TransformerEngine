# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Kimi-K3 latent MoE block."""

from typing import Optional, Union

import torch

from transformer_engine.pytorch.module import Linear, RMSNorm
from transformer_engine.pytorch.models.deepseek_v3.moe import DeepSeekV3MoE, make_glu_mlp

__all__ = ["KimiK3MoE"]


class KimiK3MoE(torch.nn.Module):
    """
    Kimi-K3 latent Mixture-of-Experts block.

    The router reads the full hidden state while the routed experts run on a
    down-projected latent: ``up_proj(RMSNorm(experts(down_proj(x))))``, with
    SiTU-GLU experts, a sigmoid router with quantile-balanced expert bias and
    an optional dense SiTU-GLU shared expert on the full hidden state. Routing,
    experts and expert parallelism are those of :class:`DeepSeekV3MoE`.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    moe_ffn_hidden_size : int
                         ffn size of each routed expert.
    num_experts : int
                 total number of routed experts.
    topk : int, default = 16
          number of experts per token.
    latent_size : int, optional
                 input/output size of the routed experts; default ``hidden_size // 2``.
    shared_expert_ffn_hidden_size : int, optional
                                   ffn size of the shared expert; ``None``
                                   disables the shared expert.
    routed_scaling_factor : float, default = 1.0
                           scaling applied to the routing probabilities.
    expert_bias_update : str, default = "quantile"
                        bias update rule of :meth:`update_expert_bias`
                        (see :class:`DeepSeekV3MoE`).
    qb_num_bins : int, default = 1000
                 histogram bins of quantile balancing.
    layernorm_epsilon : float, default = 1e-5
                       epsilon of the post-expert RMSNorm.
    single_grouped_weight : bool, default = False
                           one grouped weight parameter per expert GEMM
                           (see :class:`DeepSeekV3MoE`).
    params_dtype : torch.dtype, optional
                  dtype of module parameters.
    ep_group : ProcessGroup, optional
              expert-parallel process group; enables the NCCL EP path.
    ep_max_tokens_per_rank : int, optional
                            max local tokens per forward (required with EP).
    """

    def __init__(
        self,
        hidden_size: int,
        moe_ffn_hidden_size: int,
        num_experts: int,
        topk: int = 16,
        latent_size: Optional[int] = None,
        shared_expert_ffn_hidden_size: Optional[int] = None,
        routed_scaling_factor: float = 1.0,
        expert_bias_update: str = "quantile",
        qb_num_bins: int = 1000,
        layernorm_epsilon: float = 1e-5,
        single_grouped_weight: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        device: Union[torch.device, str] = "cuda",
        ep_group: Optional[torch.distributed.ProcessGroup] = None,
        ep_max_tokens_per_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        dtype = params_dtype if params_dtype is not None else torch.get_default_dtype()
        latent_size = latent_size if latent_size is not None else hidden_size // 2
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        common = {"bias": False, "params_dtype": params_dtype, "device": device}
        self.down_proj = Linear(hidden_size, latent_size, **common)
        self.routed_experts = DeepSeekV3MoE(
            latent_size,
            moe_ffn_hidden_size,
            num_experts,
            topk=topk,
            routed_scaling_factor=routed_scaling_factor,
            expert_bias_update=expert_bias_update,
            qb_num_bins=qb_num_bins,
            activation="situglu",
            router_input_size=hidden_size,
            single_grouped_weight=single_grouped_weight,
            params_dtype=params_dtype,
            device=device,
            ep_group=ep_group,
            ep_max_tokens_per_rank=ep_max_tokens_per_rank,
        )
        self.post_expert_norm = RMSNorm(
            latent_size, eps=layernorm_epsilon, device=device, dtype=params_dtype
        )
        self.up_proj = Linear(latent_size, hidden_size, **common)

        self.shared_expert = None
        if shared_expert_ffn_hidden_size is not None:
            self.shared_expert = make_glu_mlp(
                hidden_size, shared_expert_ffn_hidden_size, dtype, device, activation="situglu"
            )

    @property
    def expert_bias(self) -> torch.Tensor:
        """Routing bias of the routed experts."""
        return self.routed_experts.expert_bias

    def make_ep_buffer(self, device: Optional[torch.device] = None):
        """See :meth:`DeepSeekV3MoE.make_ep_buffer`."""
        return self.routed_experts.make_ep_buffer(device)

    def update_expert_bias(self) -> None:
        """See :meth:`DeepSeekV3MoE.update_expert_bias`."""
        self.routed_experts.update_expert_bias()

    def forward(self, hidden_states: torch.Tensor, ep_buffer=None) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor
                       input of shape ``[..., hidden_size]``.
        ep_buffer : EpBuffer, optional
                   buffer from :meth:`make_ep_buffer` (EP only).
        """
        latent = self.down_proj(hidden_states)
        routed = self.routed_experts(latent, ep_buffer=ep_buffer, router_input=hidden_states)
        out = self.up_proj(self.post_expert_norm(routed))
        if self.shared_expert is not None:
            out = out + self.shared_expert(hidden_states)
        return out
