#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from multiprocessing.context import BaseContext
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
)

import attr
from habitat_baselines.rl.ver.environment_worker import (
    EnvironmentWorker,
    EnvironmentWorkerProcess,
    _create_worker_configs,
)
from habitat_baselines.rl.ver.worker_common import WorkerBase, WorkerQueues

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True, auto_detect=True)
class ILEnvironmentWorkerProcess(EnvironmentWorkerProcess):
    teacher_label: Optional[str] = None

    def _step_env(self, action):
        with self.timer.avg_time("step env"):
            obs, reward, done, info = self.env.step(
                self._last_obs[self.teacher_label].item()
            )
            # ^only line different from the original EnvironmentWorkerProcess._step_env
            self._step_id += 1

            if not math.isfinite(reward):
                reward = -1.0

        with self.timer.avg_time("reset env"):
            if done:
                self._episode_id += 1
                self._step_id = 0
                if self.auto_reset_done:
                    obs = self.env.reset()

        return obs, reward, done, info


class ILEnvironmentWorker(EnvironmentWorker):
    def __init__(
        self,
        mp_ctx: BaseContext,
        env_idx: int,
        env_config,
        auto_reset_done,
        queues: WorkerQueues,
    ):
        teacher_label = None
        obs_trans_conf = env_config.habitat_baselines.rl.policy.obs_transforms
        if hasattr(env_config.habitat_baselines.rl.policy, "obs_transforms"):
            for obs_transform_config in obs_trans_conf.values():
                if hasattr(obs_transform_config, "teacher_label"):
                    teacher_label = obs_transform_config.teacher_label
                    break
        assert teacher_label is not None, "teacher_label not found in config"
        WorkerBase.__init__(
            self,
            mp_ctx,
            ILEnvironmentWorkerProcess,
            env_idx,
            env_config,
            auto_reset_done,
            queues,
            teacher_label=teacher_label,
        )
        self.env_worker_queue = queues.environments[env_idx]


def _construct_il_environment_workers_impl(
    configs,
    auto_reset_done,
    mp_ctx: BaseContext,
    queues: WorkerQueues,
):
    num_environments = len(configs)
    workers = []
    for i in range(num_environments):
        w = ILEnvironmentWorker(mp_ctx, i, configs[i], auto_reset_done, queues)
        workers.append(w)

    return workers


def construct_il_environment_workers(
    config: "DictConfig",
    mp_ctx: BaseContext,
    worker_queues: WorkerQueues,
) -> List[EnvironmentWorker]:
    configs = _create_worker_configs(config)

    return _construct_il_environment_workers_impl(configs, True, mp_ctx, worker_queues)
