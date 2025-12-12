import copy
from dataclasses import dataclass
from typing import Dict

import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.config.default_structured_configs import (
    ObsTransformConfig,
)
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from ovon.task.sensors import ClipImageGoalSensor, ImageGoalRotationSensor


@baseline_registry.register_obs_transformer()
class RelabelImageGoal(ObservationTransformer):
    """Renames ImageGoalRotationSensor to ClipImageGoalSensor"""

    def transform_observation_space(self, observation_space: spaces.Dict, **kwargs):
        assert ImageGoalRotationSensor.cls_uuid in observation_space.spaces
        observation_space = copy.deepcopy(observation_space)
        observation_space.spaces[ClipImageGoalSensor.cls_uuid] = (
            observation_space.spaces.pop(ImageGoalRotationSensor.cls_uuid)
        )
        return observation_space

    @classmethod
    def from_config(cls, config: DictConfig):
        return cls()

    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        observations[ClipImageGoalSensor.cls_uuid] = observations.pop(
            ImageGoalRotationSensor.cls_uuid
        )
        return observations


@dataclass
class RelabelImageGoalConfig(ObsTransformConfig):
    type: str = RelabelImageGoal.__name__


cs = ConfigStore.instance()

cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.relabel_image_goal",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="relabel_image_goal",
    node=RelabelImageGoalConfig,
)
