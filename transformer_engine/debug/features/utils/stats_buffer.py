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

from collections import defaultdict

import torch

from transformer_engine.debug.features.utils.stats_computation import HELPER_STATS, STATS
from nvdlfw_inspect.utils import gather_along_first_dim
from nvdlfw_inspect.logging import MetricLogger

# Buffer used for LogTensorStats and LogFp8TensorStats features.
# Buffer are fed with tensors, they compute necessary helper stats and save them.
# When log() is called, they gather helper stats, compute final stats and log them.

# for example sum of elements and number of elements on each of the nodes are helper stats
# and the mean of all the elements on all nodes is final stat we will log.


class _Buffer:
    def __init__(self, layer_name, tensor_name, stat, reduction_group):
        self.layer_name = layer_name
        self.tensor_name = tensor_name
        self.reduction_group = reduction_group
        self.stat = stat
        self.helper_stats_to_compute = STATS[self.stat][0]
        self.helper_stats_buffer = torch.zeros(len(self.helper_stats_to_compute), dtype=torch.float32).cuda()

        # in case of data parallelism it is possible that layer will not be run on one node
        # modified is set to True if node is run
        # we do not take not run nodes into account
        self.modified = torch.tensor([False], dtype=torch.bool).cuda()
        self.iteration = None
        self.skip_reduction = False
       
    def _get_helper_stat_idx(self, stat_name):
        return self.helper_stats_to_compute.index(stat_name)

    def _update_helper_stat(self, helper_stat_name, value):
        idx = self._get_helper_stat_idx(helper_stat_name)
        self.helper_stats_buffer[idx] = value
        self.modified[0] = True

    def _reset_before_next_step(self):
        self.modified[0] = False
    
    def _gather_helper_stats(self):
        if self.skip_reduction:
            return self.helper_stats_buffer.unsqueeze(0)
        mask = gather_along_first_dim(
                self.modified,
                process_group=self.reduction_group)[0]
        gathered_helper_stats_buffer, _ = gather_along_first_dim(
            self.helper_stats_buffer.unsqueeze(0), 
            process_group=self.reduction_group)
        return gathered_helper_stats_buffer[mask.to(bool)]

    def _get_helper_stats(self, gathered_helper_stats):
        output = []
        for name in self.helper_stats_to_compute:
            idx = self._get_helper_stat_idx(name)
            output.append(gathered_helper_stats[:, idx])
        return output
    
    def feed(self, tensor, iteration):
        self.iteration = iteration
        
        for helper_stat_name in self.helper_stats_to_compute:
            fn = HELPER_STATS[helper_stat_name]
            self._update_helper_stat(helper_stat_name, fn(tensor))
    
    def log(self):
        gathered_helper_stats = self._gather_helper_stats()
        if not self.modified[0]:
            return
        # combine stats and log
        # args is a tuple of helper starts, gathered only for nodes, which were run.
        args = self._get_helper_stats(gathered_helper_stats)
        combiner = STATS[self.stat][1]
        stat = combiner(*args)
        MetricLogger.log_scalar(f"{self.layer_name}_{self.tensor_name}_{self.stat}", stat.float(), self.iteration)
        self._reset_before_next_step()
        return stat


class StatsBuffers:
    def __init__(self):
        self.buffers = {} # (layer_name, tensor_name) -> buffer
        self.reduction_group_to_buffer = defaultdict(list)
    
    def reset(self):
        self.buffers = {} # (layer_name, tensor_name) -> buffer
        self.reduction_group_to_buffer = defaultdict(list)


    def try_add_buffer(self, layer_name, tensor_name, stat, options, reduction_group):
        if (layer_name, tensor_name, stat, options) in self.buffers.keys():
            return
        buffer = _Buffer(layer_name, tensor_name, stat, reduction_group)
        self.buffers[(layer_name, tensor_name, stat, options)] = buffer
        self.reduction_group_to_buffer[reduction_group].append((buffer, tensor_name))

    def feed(self, layer_name, tensor_name, stats, options, tensor, iteration, skip_reduciton):
        for stat in stats:
            buffer = self.buffers[(layer_name, tensor_name, stat, options)]
            buffer.feed(tensor, iteration)
            buffer.skip_reduction = skip_reduciton

    def log_stats(self, forward):
        output = {}
        for reduction_group, buffers in self.reduction_group_to_buffer.items():
            changed_buffers = [
                (i, buffer, tensor_name) 
                for i, (buffer, tensor_name) in enumerate(buffers) 
                if gather_along_first_dim(buffer.modified.unsqueeze(0), process_group=reduction_group)[0].any()
            ]
            for i, buffer, tensor_name in changed_buffers:
                if (forward and tensor_name in ["activation", "weight"]) or (not forward and tensor_name == "gradient"):
                    stat = buffer.log()
                    output[(buffer.layer_name, buffer.tensor_name, buffer.stat, buffer.iteration)] = stat # for testing purpouses
        
        return output 

STATS_BUFFERS = StatsBuffers()