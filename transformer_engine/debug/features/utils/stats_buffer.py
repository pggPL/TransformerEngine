# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from transformer_engine.debug.features.utils.stats_computation import (
    STATS,
    DEPENDENCIES,
    stats_to_num,
)
from nvdlfw_inspect.utils import gather_along_first_dim
from nvdlfw_inspect.logging import MetricLogger

# Buffer used for LogTensorStats and LogFp8TensorStats features.
# Buffer are fed with tensors, they compute necessary stats and save them.
# When log() is called, they gather stats from all nodes, compute combined final stats and log them.


class _Buffer:
    def __init__(self, layer_name, tensor_name, stats, reduction_group):
        self.layer_name = layer_name
        self.tensor_name = tensor_name
        self.reduction_group = reduction_group
        self.stats_to_log = stats

        self.stats_to_compute = set()
        for stat in stats:
            self.stats_to_compute = self.stats_to_compute | DEPENDENCIES[stat]

        self._buffer = torch.zeros(len(STATS), dtype=torch.float32).cuda()
        self._new_buffer = self._buffer.clone()
        self._tmp_buffer = self._buffer.clone()

        # in case of data parallelism it is possible that layer will not be run on one node
        # modified is set to True if node is run
        # we do not take not run nodes into account
        self.modified = torch.tensor([False], dtype=torch.bool).cuda()
        self.iteration = None
        self.skip_reduction = False

    def _reset_before_next_step(self):
        self.modified[0] = False

    def _gather_helper_stats(self):
        if self.skip_reduction:
            return self._buffer.unsqueeze(0)
        mask = gather_along_first_dim(self.modified, process_group=self.reduction_group)[0]
        gathered__buffer, _ = gather_along_first_dim(
            self._buffer.unsqueeze(0), process_group=self.reduction_group
        )
        return gathered__buffer[mask.to(bool)]

    def feed(self, tensor, iteration):
        # feed() is used to add tensor for computing the statistics.
        # Because of the microbatching, feed() can be used multiple
        # times for one log().
        #
        # Ability to combine result for already processed tensors with
        # results new tensor are the main reason for such a design of this class.

        self.iteration = iteration

        # save stats for tensor to tmp buffer
        for stat_name in self.stats_to_compute:
            fn, _ = STATS[stat_name]
            self._tmp_buffer[stats_to_num[stat_name]] = fn(tensor)

        # [num_buffers, num_stats]
        buffers = torch.cat((self._buffer.unsqueeze(0), self._tmp_buffer.unsqueeze(0)), dim=0)

        for stat_name in self.stats_to_compute:
            fn, combinator = STATS[stat_name]
            if self.modified:
                self._new_buffer[stats_to_num[stat_name]] = combinator(buffers)
            else:
                fn = STATS[stat_name][0]
                self._new_buffer[stats_to_num[stat_name]] = fn(tensor)

        self._buffer.copy_(self._new_buffer)

        self.modified[0] = True

    def log(self):
        # [num_active_nodes, num_stats]
        gathered_helper_stats = self._gather_helper_stats()

        if not self.modified[0]:
            return
        output = {}
        for stat_name in self.stats_to_log:
            combiner = STATS[stat_name][1]
            stat_value = combiner(gathered_helper_stats)

            MetricLogger.log_scalar(
                f"{self.layer_name}_{self.tensor_name}_{stat_name}", stat_value, self.iteration
            )
            output[(self.layer_name, self.tensor_name, stat_name, self.iteration)] = (
                stat_value  # for debuggin purpouses
            )
        self._reset_before_next_step()
        return output


class StatsBuffers:
    def __init__(self):
        self.buffers = {}  # (layer_name, tensor_name) -> buffer
        self.reduction_group_to_buffer = defaultdict(list)

    def reset(self):
        self.buffers = {}  # (layer_name, tensor_name) -> buffer
        self.reduction_group_to_buffer = defaultdict(list)

    def try_add_buffer(self, layer_name, tensor_name, stats, options, reduction_group):
        if (layer_name, tensor_name, options) in self.buffers.keys():
            return
        buffer = _Buffer(layer_name, tensor_name, stats, reduction_group)
        self.buffers[(layer_name, tensor_name, options)] = buffer
        self.reduction_group_to_buffer[reduction_group].append((buffer))

    def feed(self, layer_name, tensor_name, options, tensor, iteration, skip_reduciton):
        buffer = self.buffers[(layer_name, tensor_name, options)]
        buffer.feed(tensor, iteration)
        buffer.skip_reduction = skip_reduciton

    def log_stats(self):
        output = {}
        for reduction_group, buffers in self.reduction_group_to_buffer.items():
            changed_buffers = [
                (i, buffer)
                for i, buffer in enumerate(buffers)
                if gather_along_first_dim(
                    buffer.modified.unsqueeze(0), process_group=reduction_group
                )[0].any()
            ]
            for _, buffer in changed_buffers:
                stats = buffer.log()
                output = output | stats

        return output


STATS_BUFFERS = StatsBuffers()
