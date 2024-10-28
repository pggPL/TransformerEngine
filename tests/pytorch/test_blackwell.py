# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
from dataclasses import dataclass
from typing import Optional

import torch
import pytest

from transformer_engine.pytorch.fp8 import (
    fp8_autocast,
    FP8GlobalStateManager,
)
from transformer_engine.pytorch.utils import (
    scaled_init_method_normal,
    is_bf16_compatible,
)
from transformer_engine.pytorch import Linear
from transformer_engine.common import recipe

# Only run FP8 tests on H100.
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()


def custom_amax_to_scale(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: torch.Tensor,
    recipe: recipe.DelayedScaling,
) -> torch.Tensor:
    """Custom func to test recipe."""
    sf = fp8_max / amax
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)

    return sf


def custom_amax_compute(amax_history: torch.Tensor) -> torch.Tensor:
    """Custom func to test recipe."""
    return torch.min(amax_history, dim=0).values


@dataclass
class ModelConfig:
    """Transformer model configuration"""

    num_layers: int
    seq_len: int
    batch_size: int
    hidden_size: int
    num_attention_heads: int
    kv_channels: Optional[int] = None

    def is_fp8_supported(self):
        if self.seq_len * self.batch_size % 16:
            return False
        if self.hidden_size % 16:
            return False
        return True


model_configs = {
    "126m": ModelConfig(12, 2048, 2, 768, 12),
}

fp8_recipes = [
    None,  # Handles non-FP8 case
    recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3),
    recipe.CurrentScaling(margin=0, fp8_format=recipe.Format.E4M3,),
]

param_types = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)

all_boolean = [True, False]
batch_sizes_with_zero = [0, 1, 2]

all_activations = ["gelu", "relu", "reglu", "geglu", "swiglu", "srelu"]
all_normalizations = ["LayerNorm", "RMSNorm"]


def _disable_wgrads(block):
    for p in block.parameters():
        p.requires_grad = False


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    FP8GlobalStateManager.reset()


def _test_sanity_common(block, dtype, config, fp8_recipe, skip_wgrad, skip_dgrad):
    if skip_dgrad and skip_wgrad:
        pytest.skip("No gradient computation; Skipping to avoid PyTorch RuntimeError.")

    te_inp = torch.randn(
        (config.seq_len, config.batch_size, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=not skip_dgrad,
    )

    if skip_wgrad:
        _disable_wgrads(block)

    use_fp8 = fp8_recipe is not None
    with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
        te_out = block(te_inp)
    if isinstance(te_out, tuple):
        te_out = te_out[0]
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("skip_wgrad", [False])
@pytest.mark.parametrize("skip_dgrad", [False])
@pytest.mark.parametrize("dummy_mxfp8", all_boolean)
def test_sanity_linear(dtype, fp8_recipe, model, skip_wgrad, skip_dgrad, dummy_mxfp8):
    if dummy_mxfp8 and not isinstance(fp8_recipe, recipe.DelayedScaling):
        pytest.skip("Dummy mode only for Delayed Scaling.")
    if dummy_mxfp8:
        os.environ["_NVTE_MXFP8_GEMM_DEBUG"] = "1"

    config = model_configs[model]

    if fp8_recipe is not None:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if not config.is_fp8_supported():
            pytest.skip("Model config does not support FP8")

    sigma = 0.023
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block = Linear(
        config.hidden_size,
        config.hidden_size,
        init_method=output_layer_init_method,
        params_dtype=dtype,
        device="cuda",
    )
    _test_sanity_common(block, dtype, config, fp8_recipe, skip_wgrad, skip_dgrad)

    if dummy_mxfp8:
        del os.environ["_NVTE_MXFP8_GEMM_DEBUG"]
