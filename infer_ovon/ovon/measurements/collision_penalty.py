from dataclasses import dataclass
from typing import Any

from habitat import EmbodiedTask, Measure, Simulator, registry
from habitat.config.default_structured_configs import MeasurementConfig
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@registry.register_measure
class CollisionPenalty(Measure):
    """
    Returns a penalty value if the robot has collided.
    """

    cls_uuid: str = "collision_penalty"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._collision_penalty = config.collision_penalty
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, ["collisions"])
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        collisions = task.measurements.measures["collisions"].get_metric()
        collided = collisions is not None and collisions["is_collision"]
        if collided:
            self._metric = -self._collision_penalty
        else:
            self._metric = 0


@dataclass
class CollisionPenaltyMeasurementConfig(MeasurementConfig):
    type: str = CollisionPenalty.__name__
    collision_penalty: float = 0.003


cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.measurements.{CollisionPenalty.cls_uuid}",
    group="habitat/task/measurements",
    name=f"{CollisionPenalty.cls_uuid}",
    node=CollisionPenaltyMeasurementConfig,
)
