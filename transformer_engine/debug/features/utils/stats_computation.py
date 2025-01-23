# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

# Computing statistics happen in 2 passes.
# In the first pass, the HELPER_STATS on each node are computed.
# Then they are all-gathered on every node. In the second pass STATS
# are computed from HELPER_STATS.

def _compute_dynamic_range_top(tensor):
    tensor_abs = tensor.abs()
    tensor_abs = tensor_abs[tensor_abs != 0]
    amax = tensor_abs.max().float()
    if not amax.all():
        amax = torch.tensor(1, device=tensor.device).to(torch.float)
    return torch.log2(amax)

def _compute_dynamic_range_bottom(tensor):
    tensor_abs = tensor.abs()
    tensor_abs = tensor_abs[tensor_abs != 0]
    if tensor_abs.any():
        amin = tensor_abs.min().float()
    else:
        amin = torch.tensor(1, device=tensor.device).to(torch.float)
    return torch.log2(amin)

def compute_std(variances, numels, sums):
    # We use Welford algorithm for numerically stable distributed variance computation.
    mean = torch.sum(sums) / torch.sum(numels)
    means = sums / numels
    var = torch.sum(numels * (variances - torch.pow((means - mean), 2))) / torch.sum(numels)
    return torch.sqrt(var)


HELPER_STATS = {
    "min": torch.min,
    "max": torch.max,
    "sum": torch.sum,
    "numel": lambda x: x.numel(),
    "l1_norm": lambda x: torch.norm(x, p=1),
    "l2_norm_square": lambda x: torch.sum(x ** 2),
    "variance": torch.var,
    "amax": lambda x: x.abs().max(),
    "dynamic_range_top": _compute_dynamic_range_top,
    "dynamic_range_bottom": _compute_dynamic_range_bottom,
    "underflows_num": lambda x: (x == 0).sum(),
    "overflows_num": lambda x: (x == 255).sum(),
}

STATS = {
    "min": (["min"], torch.min),
    "max": (["max"], torch.max),
    "mean": (["numel", "sum"], lambda numels, sums: sum(sums) / sum(numels)),
    "std": (["variance", "numel", "sum"], compute_std), 
    "l1_norm": (["l1_norm"], torch.sum),
    "l2_norm": (["l2_norm_square"], lambda x: torch.sqrt(torch.sum(x))),
    "cur_amax": (["amax"], torch.max),
    "dynamic_range": (["dynamic_range_top", "dynamic_range_bottom"], lambda x, y: torch.max(x) - torch.min(y)),
    "underflows": (["underflows_num", "numel"], lambda x, n: 100 * torch.sum(x) / torch.sum(n)),
    "overflows":(["overflows_num", "numel"], lambda x, n: 100 * torch.sum(x) / torch.sum(n)),
}