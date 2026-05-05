# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_MOE_UNPERMUTE_PYTORCH
from transformer_engine.pytorch import moe_unpermute

# expert_out:    [num_out_tokens, hidden_size], expert-contiguous,
#                produced by GroupedLinear (or a grouped MLP).
# row_id_map:    returned by moe_permute.
# merging_probs: [num_tokens, num_experts]; routing probabilities used to
#                weight the per-expert contributions to each token. Provide
#                for top-k routing; pass None for top-1.
tokens_out = moe_unpermute(
    expert_out,
    row_id_map,
    merging_probs=merging_probs,
)

# tokens_out: [num_tokens, hidden_size], in the original token order
# END_MOE_UNPERMUTE_PYTORCH
