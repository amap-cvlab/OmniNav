from dataclasses import dataclass, field
from typing import Any, List

from habitat import EmbodiedTask, Measure, registry
from habitat.config.default_structured_configs import MeasurementConfig
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@registry.register_measure
class SumReward(Measure):
    """
    Sums various reward measures.
    """

    cls_uuid: str = "sum_reward"

    def __init__(self, config: "DictConfig", *args: Any, **kwargs: Any):
        self._config = config
        self._reward_terms = config.reward_terms
        self._reward_coefficients = [float(i) for i in config.reward_coefficients]
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, self._reward_terms)
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        self._metric = 0
        for term, coefficient in zip(self._reward_terms, self._reward_coefficients):
            self._metric += coefficient * task.measurements.measures[term].get_metric()


@dataclass
class SumRewardMeasurementConfig(MeasurementConfig):
    type: str = SumReward.__name__
    reward_terms: List[str] = field(
        # available options are "disk" and "tensorboard"
        default_factory=list
    )
    reward_coefficients: List[str] = field(
        # available options are "disk" and "tensorboard"
        default_factory=list
    )


cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.measurements.{SumReward.cls_uuid}",
    group="habitat/task/measurements",
    name=f"{SumReward.cls_uuid}",
    node=SumRewardMeasurementConfig,
)
