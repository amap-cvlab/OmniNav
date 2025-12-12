from typing import TYPE_CHECKING, Any

import numpy as np
import quaternion
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationEpisode, Success
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_coeff,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_measure
class AngleToGoal(Measure):
    """The measure calculates an angle towards the goal. Note: this measure is
    only valid for single goal tasks (e.g., ImageNav)
    """

    cls_uuid: str = "angle_to_goal"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        super().__init__()
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = None
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(self, episode: NavigationEpisode, *args: Any, **kwargs: Any):
        current_angle = self._sim.get_agent_state().rotation
        if not isinstance(current_angle, quaternion.quaternion):
            current_angle = quaternion_from_coeff(current_angle)

        goal_angle = episode.goals[0].rotation
        if not isinstance(goal_angle, quaternion.quaternion):
            goal_angle = quaternion_from_coeff(goal_angle)

        self._metric = angle_between_quaternions(current_angle, goal_angle)


@registry.register_measure
class AngleSuccess(Measure):
    """Weather or not the agent is within an angle tolerance."""

    cls_uuid: str = "angle_success"

    def __init__(self, config: "DictConfig", *args: Any, **kwargs: Any):
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [Success.cls_uuid, AngleToGoal.cls_uuid]
        )
        self.update_metric(task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        success = task.measurements.measures[Success.cls_uuid].get_metric()
        angle_to_goal = task.measurements.measures[AngleToGoal.cls_uuid].get_metric()

        if success and np.rad2deg(angle_to_goal) < self._config.success_angle:
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class AgentPosition(Measure):
    """The measure calculates current position of agent"""

    cls_uuid: str = "agent_position"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        super().__init__()
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = None
        self.update_metric(*args, **kwargs)  # type: ignore

    def update_metric(self, *args: Any, **kwargs: Any):
        self._metric = self._sim.get_agent_state().position


@registry.register_measure
class AgentRotation(Measure):
    """The measure calculates current position of agent"""

    cls_uuid: str = "agent_rotation"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        super().__init__()
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = None
        self.update_metric(*args, **kwargs)  # type: ignore

    def update_metric(self, *args: Any, **kwargs: Any):
        self._metric = self._sim.get_agent_state().rotation
