from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import DistanceToGoal, Success

from ovon.measurements.imagenav import AngleSuccess, AngleToGoal

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_measure
class ImageNavReward(Measure):
    cls_uuid: str = "imagenav_reward"

    def __init__(self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config
        self._previous_dtg: Optional[float] = None
        self._previous_atg: Optional[float] = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self,
        *args: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                Success.cls_uuid,
                DistanceToGoal.cls_uuid,
                AngleToGoal.cls_uuid,
                AngleSuccess.cls_uuid,
            ],
        )
        self._metric = None
        self._previous_dtg = None
        self._previous_atg = None
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        # success reward
        success = task.measurements.measures[Success.cls_uuid].get_metric()
        success_reward = self._config.success_reward if success else 0.0

        # distance-to-goal reward
        dtg = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        if self._previous_dtg is None:
            self._previous_dtg = dtg
        add_dtg = self._config.use_dtg_reward
        dtg_reward = self._previous_dtg - dtg if add_dtg else 0.0
        self._previous_dtg = dtg

        # angle-to-goal reward
        atg = task.measurements.measures[AngleToGoal.cls_uuid].get_metric()
        add_atg = self._config.use_atg_reward
        if self._config.use_atg_fix:
            if dtg > self._config.atg_reward_distance:
                atg = np.pi
        else:
            if dtg > self._config.atg_reward_distance:
                add_atg = False
        if self._previous_atg is None:
            self._previous_atg = atg
        angle_reward = self._previous_atg - atg if add_atg else 0.0
        self._previous_atg = atg

        # angle success reward
        angle_success = task.measurements.measures[AngleSuccess.cls_uuid].get_metric()
        angle_success_reward = (
            self._config.angle_success_reward if angle_success else 0.0
        )

        # slack penalty
        slack_penalty = self._config.slack_penalty

        # reward
        self._metric = (
            success_reward
            + dtg_reward
            + angle_reward
            + angle_success_reward
            + slack_penalty
        )
