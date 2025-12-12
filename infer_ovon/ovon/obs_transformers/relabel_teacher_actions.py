import copy
from dataclasses import dataclass
from typing import Dict

import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.config.default_structured_configs import ObsTransformConfig
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@baseline_registry.register_obs_transformer()
class RelabelTeacherActions(ObservationTransformer):
    """Renames the entry corresponding to the given key string within the observations
    dict to 'teacher_actions'"""

    TEACHER_LABEL: str = "teacher_label"

    def __init__(self, teacher_label: str):
        super().__init__()
        self.teacher_label = teacher_label

    def transform_observation_space(self, observation_space: spaces.Dict, **kwargs):
        assert (
            self.teacher_label in observation_space.spaces
        ), f"Teacher action key {self.teacher_label} not in observation space!"
        observation_space = copy.deepcopy(observation_space)
        observation_space.spaces[self.TEACHER_LABEL] = observation_space.spaces.pop(
            self.teacher_label
        )
        return observation_space

    @classmethod
    def from_config(cls, config: DictConfig):
        return cls(config.teacher_label)

    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        observations[self.TEACHER_LABEL] = observations.pop(self.teacher_label)
        return observations


@dataclass
class RelabelTeacherActionsConfig(ObsTransformConfig):
    type: str = RelabelTeacherActions.__name__
    teacher_label: str = ""


cs = ConfigStore.instance()

cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.relabel_teacher_actions",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="relabel_teacher_actions",
    node=RelabelTeacherActionsConfig,
)
