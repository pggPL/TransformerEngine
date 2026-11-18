# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DeepSeekV3 MoE block: sigmoid router with aux-loss-free bias, shared +
routed experts."""

from typing import Optional, Union

import torch

import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.router import fused_topk_with_score_function
from transformer_engine.pytorch.permutation import (
    moe_permute_and_pad_with_probs,
    moe_permute_with_probs,
    moe_unpermute,
)
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    get_align_size_for_quantization,
)

__all__ = ["DeepSeekV3MoE", "make_glu_mlp", "expert_bias_from_qb_histogram"]


_EP_ALIGNMENT = 256
_FUSED_MLP_ROWS = 256


_ACTIVATIONS = {
    "swiglu": (te_ops.SwiGLU, te_ops.ScaledSwiGLU),
    "situglu": (te_ops.SiTUGLU, te_ops.ScaledSiTUGLU),
}


def make_glu_mlp(
    hidden_size, ffn_hidden_size, dtype, device, num_experts=None, activation="swiglu"
):
    """Dense GLU MLP, or a grouped one (probs applied inside the activation) per expert.

    The grouped variant fuses into a single CuTe grouped MLP on supported hardware.
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}")
    dense_act, grouped_act = _ACTIVATIONS[activation]
    common = {"bias": False, "dtype": dtype, "device": device}
    if num_experts is None:
        return te_ops.Sequential(
            te_ops.Linear(hidden_size, 2 * ffn_hidden_size, **common),
            dense_act(),
            te_ops.Linear(ffn_hidden_size, hidden_size, **common),
        )
    return te_ops.Sequential(
        te_ops.GroupedLinear(num_experts, hidden_size, 2 * ffn_hidden_size, **common),
        grouped_act(glu_interleave_size=32),
        te_ops.GroupedLinear(num_experts, ffn_hidden_size, hidden_size, **common),
    )


@torch.no_grad()
def expert_bias_from_qb_histogram(
    histogram: torch.Tensor, bin_bounds: torch.Tensor, expert_bias: torch.Tensor, topk: int
) -> torch.Tensor:
    """Quantile-balancing bias from a required-bias histogram.

    ``histogram[e, b]`` counts tokens whose required bias for expert ``e``
    (top-k cutoff score minus the expert's score) fell into uniform bin ``b``
    of ``[bin_bounds[0], bin_bounds[1])``. The new bias of each expert is the
    ``topk / num_experts`` quantile of its required bias, linearly
    interpolated within the bin, then mean-centered. Experts without counts
    keep their current bias.
    """
    num_experts, num_bins = histogram.shape
    lower, upper = bin_bounds[0], bin_bounds[1]
    width = (upper - lower) / num_bins
    counts = histogram.to(torch.int64)
    total = counts.sum(dim=1)
    rank = torch.ceil(total.double() * topk / num_experts).to(torch.int64).clamp_min(1)
    cdf = counts.cumsum(dim=1)
    bin_idx = (cdf < rank[:, None]).sum(dim=1).clamp_max(num_bins - 1)
    below = torch.where(
        bin_idx > 0, cdf.gather(1, (bin_idx - 1).clamp_min(0)[:, None]).squeeze(1), 0
    )
    in_bin = counts.gather(1, bin_idx[:, None]).squeeze(1).clamp_min(1)
    frac = (rank - below).double() / in_bin.double()
    quantile = lower.double() + (bin_idx.double() + frac) * width.double()
    new_bias = torch.where(total > 0, quantile.float(), expert_bias)
    return new_bias - new_bias.mean()


class DeepSeekV3MoE(torch.nn.Module):
    """
    DeepSeekV3 Mixture-of-Experts block.

    Each token is scored by a sigmoid router with a non-trainable expert bias
    updated by ``update_expert_bias()`` (aux-loss-free load balancing, either
    the sign-of-error rule of DeepSeek-V3 or the quantile balancing of
    Kimi-K3) and, optionally, group-limited routing: experts are split into ``num_groups``
    groups, the top ``group_topk`` groups are selected by their summed scores,
    and the final ``topk`` experts are chosen only from those groups. Selected
    tokens run through the routed experts, a GLU MLP shared across experts
    as a grouped GEMM, with the routing probability applied inside the MLP. An
    optional shared expert (dense GLU MLP) is added to every token. The router
    may read a different tensor than the experts (``router_input_size`` and the
    ``router_input`` argument of :meth:`forward`), as in latent MoE. On
    hardware that supports it the expert MLP runs as a single fused
    grouped-GEMM kernel. MXFP8 and NVFP4 pad each expert's token count to a
    multiple of 256, including unfused execution; FP8 block scaling uses 128.

    Without ``ep_group`` all experts live on the local device. With
    ``ep_group`` the experts are split across the group and tokens are
    exchanged over NCCL; this requires ``ep_bootstrap`` to be called once per
    process before constructing the module, and bfloat16 inputs.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    moe_ffn_hidden_size : int
                         ffn size of each routed expert.
    num_experts : int
                 total number of routed experts.
    topk : int, default = 8
          number of experts per token.
    num_groups : int, optional
                number of expert groups for node-limited routing; requires ``group_topk``.
    group_topk : int, optional
                number of groups each token is limited to; requires ``num_groups``.
    routed_scaling_factor : float, default = 2.5
                           scaling applied to the routing probabilities.
    shared_expert_ffn_hidden_size : int, optional
                                   ffn size of the shared expert; ``None``
                                   disables the shared expert.
    expert_bias_update : str, default = "sign"
                        ``"sign"``: :meth:`update_expert_bias` moves each bias by
                        ``expert_bias_update_rate`` against the sign of its load
                        error. ``"quantile"``: the router accumulates a
                        required-bias histogram and :meth:`update_expert_bias`
                        replaces the bias with its ``topk / num_experts``
                        quantile (quantile balancing). Not supported with
                        grouped routing.
    expert_bias_update_rate : float, default = 1e-3
                             step size of the ``"sign"`` bias update.
    qb_num_bins : int, default = 1000
                 number of histogram bins of the ``"quantile"`` update.
    activation : str, default = "swiglu"
                GLU activation of routed and shared experts, ``"swiglu"`` or
                ``"situglu"``.
    router_input_size : int, optional
                       size of the router input when it differs from
                       ``hidden_size``; requires passing ``router_input`` to
                       :meth:`forward`.
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
        topk: int = 8,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        routed_scaling_factor: float = 2.5,
        shared_expert_ffn_hidden_size: Optional[int] = None,
        expert_bias_update: str = "sign",
        expert_bias_update_rate: float = 1e-3,
        qb_num_bins: int = 1000,
        activation: str = "swiglu",
        router_input_size: Optional[int] = None,
        params_dtype: Optional[torch.dtype] = None,
        device: Union[torch.device, str] = "cuda",
        ep_group: Optional[torch.distributed.ProcessGroup] = None,
        ep_max_tokens_per_rank: Optional[int] = None,
    ) -> None:
        super().__init__()

        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if not 1 <= topk <= num_experts:
            raise ValueError("topk must be in [1, num_experts].")
        if (num_groups is None) != (group_topk is None):
            raise ValueError("num_groups and group_topk must be provided together.")
        if num_groups is not None:
            if num_groups <= 0 or num_experts % num_groups != 0:
                raise ValueError("num_groups must be positive and divide num_experts.")
            if not 1 <= group_topk <= num_groups:
                raise ValueError("group_topk must be in [1, num_groups].")
            if topk % group_topk != 0:
                raise ValueError("topk must be divisible by group_topk.")
            if topk // group_topk > num_experts // num_groups:
                raise ValueError("topk per group must not exceed the number of experts per group.")
        if expert_bias_update not in ("sign", "quantile"):
            raise ValueError("expert_bias_update must be 'sign' or 'quantile'.")
        if expert_bias_update == "quantile" and num_groups is not None:
            raise ValueError("Quantile balancing does not support grouped routing.")

        dtype = params_dtype if params_dtype is not None else torch.get_default_dtype()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.topk = topk
        self.num_groups = num_groups
        self.group_topk = group_topk
        self.routed_scaling_factor = routed_scaling_factor
        self.expert_bias_update = expert_bias_update
        self.expert_bias_update_rate = expert_bias_update_rate
        self.router_input_size = router_input_size or hidden_size

        self.gate = torch.nn.Linear(
            self.router_input_size, num_experts, bias=False, dtype=dtype, device=device
        )
        self.register_buffer(
            "expert_bias", torch.zeros(num_experts, dtype=torch.float32, device=device)
        )
        self._last_tokens_per_expert: Optional[torch.Tensor] = None
        quantile = expert_bias_update == "quantile"
        self.register_buffer(
            "qb_histogram",
            (
                torch.zeros(num_experts, qb_num_bins, dtype=torch.int32, device=device)
                if quantile
                else None
            ),
            persistent=False,
        )
        self.register_buffer(
            "qb_bin_bounds",
            torch.tensor([-1.0, 1.0], dtype=torch.float32, device=device) if quantile else None,
        )

        self.ep_group = ep_group
        self.ep_size = 1 if ep_group is None else torch.distributed.get_world_size(ep_group)
        assert num_experts % self.ep_size == 0
        num_local_experts = num_experts // self.ep_size

        expert_mlp = make_glu_mlp(
            hidden_size,
            moe_ffn_hidden_size,
            dtype,
            device,
            num_experts=num_local_experts,
            activation=activation,
        )

        self._ep_buffer_kwargs = None
        if ep_group is not None:
            from transformer_engine.pytorch.ep import EpConfig, get_ep_drop_on_overflow

            assert ep_max_tokens_per_rank is not None, "EP requires ep_max_tokens_per_rank."
            drop_on_overflow = get_ep_drop_on_overflow()
            if drop_on_overflow is None:
                raise RuntimeError("EP requires ep_bootstrap before constructing DeepSeekV3MoE.")
            cap = self.ep_recv_capacity(
                self.ep_size, ep_max_tokens_per_rank, topk, num_local_experts
            )
            self._ep_buffer_kwargs = {
                "top_k": topk,
                "max_tokens_per_rank": ep_max_tokens_per_rank,
                "hidden_dim": hidden_size,
                "num_local_experts": num_local_experts,
                "recv_capacity_per_rank": cap,
                "alignment": _EP_ALIGNMENT,
            }
            config = EpConfig(
                ep_group=ep_group,
                drop_on_overflow=drop_on_overflow,
                **self._ep_buffer_kwargs,
            )
            dispatch = te_ops.MoeDispatch(config)
            combine = te_ops.MoeCombine(config)
            fc1, activation, fc2 = expert_mlp
            dispatch.set_extra_output_channel(0, "tokens_per_expert", output_to_caller=False)
            dispatch.set_extra_output_channel(1, "routing_weights", output_to_caller=False)
            fc1.set_extra_input_channel(0, "tokens_per_expert")
            activation.set_extra_input_channel(0, "routing_weights")
            fc2.set_extra_input_channel(0, "tokens_per_expert")
            self.experts = te_ops.Sequential(
                {"dispatch": dispatch, "0": fc1, "1": activation, "2": fc2, "combine": combine}
            )
        else:
            self.experts = expert_mlp

        self.shared_expert = None
        if shared_expert_ffn_hidden_size is not None:
            self.shared_expert = make_glu_mlp(
                hidden_size, shared_expert_ffn_hidden_size, dtype, device, activation=activation
            )

    @staticmethod
    def ep_recv_capacity(
        ep_size: int, max_tokens_per_rank: int, topk: int, num_local_experts: int
    ) -> int:
        """Worst-case receive capacity including per-expert alignment padding."""
        cap = ep_size * max_tokens_per_rank * topk
        cap += num_local_experts * _EP_ALIGNMENT
        return -(-cap // _FUSED_MLP_ROWS) * _FUSED_MLP_ROWS

    def _route(self, logits: torch.Tensor, topk_indices: Optional[torch.Tensor] = None):
        qb_kwargs = {}
        if self.qb_histogram is not None and self.training and torch.is_grad_enabled():
            qb_kwargs = {
                "qb_histogram": self.qb_histogram,
                "qb_bin_bounds": self.qb_bin_bounds,
                "qb_histogram_mode": "fused_atomic",
            }
        return fused_topk_with_score_function(
            logits=logits,
            topk=self.topk,
            use_pre_softmax=False,
            num_groups=self.num_groups,
            group_topk=self.group_topk,
            scaling_factor=self.routed_scaling_factor,
            score_function="sigmoid",
            expert_bias=self.expert_bias,
            topk_indices=topk_indices,
            **qb_kwargs,
        )

    def _forward_local(self, tokens: torch.Tensor, router_input: torch.Tensor) -> torch.Tensor:
        probs, routing_map = self._route(self.gate(router_input).float())
        tokens_per_expert = routing_map.sum(dim=0)
        self._last_tokens_per_expert = tokens_per_expert.detach()

        # Quantized grouped GEMMs need every expert's row count aligned.
        align = 1
        if FP8GlobalStateManager.is_fp8_enabled():
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            align = get_align_size_for_quantization(recipe)
            if recipe.mxfp8() or recipe.nvfp4():
                align = max(align, _FUSED_MLP_ROWS)
            elif recipe.float8_block_scaling():
                align = max(align, 128)
        if align > 1:
            permuted, permuted_probs, row_id_map, pad_offsets, tokens_per_expert = (
                moe_permute_and_pad_with_probs(tokens, probs, routing_map, tokens_per_expert, align)
            )
        else:
            permuted, permuted_probs, row_id_map = moe_permute_with_probs(
                tokens, probs, routing_map, num_out_tokens=tokens.shape[0] * self.topk
            )
            pad_offsets = None

        # The fused grouped MLP requires the total row count to be a multiple
        # of 128; rows beyond sum(tokens_per_expert) fall outside every group.
        num_rows = permuted.shape[0]
        pad = (-num_rows) % 128
        if pad:
            permuted = torch.nn.functional.pad(permuted, (0, 0, 0, pad))
            permuted_probs = torch.nn.functional.pad(permuted_probs, (0, pad))

        out = self.experts(
            permuted, tokens_per_expert, permuted_probs.to(tokens.dtype), tokens_per_expert
        )
        return moe_unpermute(
            out[:num_rows], row_id_map, restore_shape=tokens.shape, pad_offsets=pad_offsets
        )

    def make_ep_buffer(self, device: Optional[torch.device] = None):
        """EP buffer for this module's routing config. ``forward`` creates one per call unless
        given one; a buffer holds one call's routing state until its backward runs, so reuse
        it only across calls whose backward has completed."""
        from transformer_engine.pytorch.ep import EpBuffer

        assert self._ep_buffer_kwargs is not None, "make_ep_buffer requires ep_group."
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return EpBuffer(**self._ep_buffer_kwargs, device=device)

    def _forward_ep(
        self, tokens: torch.Tensor, router_input: torch.Tensor, ep_buffer=None
    ) -> torch.Tensor:
        assert tokens.dtype == torch.bfloat16, "The EP path requires bfloat16 inputs."
        buffer = ep_buffer if ep_buffer is not None else self.make_ep_buffer(tokens.device)
        topk_idx = torch.empty(
            (tokens.shape[0], self.topk), dtype=torch.int64, device=tokens.device
        )
        probs, topk_idx = self._route(self.gate(router_input).float(), topk_indices=topk_idx)
        flat_idx = topk_idx.flatten()
        self._last_tokens_per_expert = torch.zeros(
            self.num_experts, dtype=torch.long, device=tokens.device
        ).scatter_add_(0, flat_idx, torch.ones_like(flat_idx))
        topk_weights = probs.gather(1, topk_idx)

        return self.experts(
            tokens,
            topk_idx,
            topk_weights,
            topk_idx,
            op_kwargs={self.experts[0]: {"buffer": buffer}, self.experts[-1]: {"buffer": buffer}},
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        ep_buffer=None,
        router_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor
                       input of shape ``[..., hidden_size]``.
        ep_buffer : EpBuffer, optional
                   buffer from :meth:`make_ep_buffer` to reuse instead of creating one
                   per call (EP only).
        router_input : torch.Tensor, optional
                      router input of shape ``[..., router_input_size]`` with the same
                      leading dims as ``hidden_states``; defaults to ``hidden_states``.
        """
        tokens = hidden_states.reshape(-1, self.hidden_size)
        if router_input is None:
            router_input = tokens
        else:
            router_input = router_input.reshape(-1, self.router_input_size)
        if self.ep_group is not None:
            out = self._forward_ep(tokens, router_input, ep_buffer)
        else:
            out = self._forward_local(tokens, router_input)
        if self.shared_expert is not None:
            out = out + self.shared_expert(tokens)
        return out.view_as(hidden_states)

    @torch.no_grad()
    def update_expert_bias(self) -> None:
        """Aux-loss-free bias update from the routing statistics of the forward
        passes since the last call.

        ``"sign"`` uses the last forward's token counts per expert; with
        data/expert parallelism, all-reduce ``_last_tokens_per_expert`` across
        ranks before calling (or call on identically-routed ranks).
        ``"quantile"`` uses the required-bias histogram accumulated over every
        training forward since the last call (all-reduce ``qb_histogram``
        across ranks first), replaces the bias and resets the histogram and the
        bin bounds for the next period.
        """
        if self.expert_bias_update == "quantile":
            new_bias = expert_bias_from_qb_histogram(
                self.qb_histogram, self.qb_bin_bounds, self.expert_bias, self.topk
            )
            self.expert_bias.copy_(new_bias)
            self.qb_bin_bounds.copy_(torch.stack([new_bias.min() - 1.0, new_bias.max() + 1.0]))
            self.qb_histogram.zero_()
            return
        counts = self._last_tokens_per_expert
        if counts is None:
            return
        err = counts.float().mean() - counts.float()
        self.expert_bias += self.expert_bias_update_rate * torch.sign(err)
