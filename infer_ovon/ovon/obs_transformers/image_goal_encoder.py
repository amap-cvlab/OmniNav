from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.config.default_structured_configs import (
    ObsTransformConfig,
)
from habitat_baselines.utils.common import inference_mode
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from ovon.models.ovrl_policy import ResNetCLIPGoalEncoder
from ovon.task.sensors import ClipImageGoalSensor, CurrentEpisodeUUIDSensor


@baseline_registry.register_obs_transformer()
class CLIPImageGoalEncoder(ObservationTransformer):
    """Renames the entry corresponding to the given key string within the observations
    dict to 'teacher_actions'"""

    def __init__(
        self,
        backbone: str,
        inference_worker_idx: int = 0,
        clip_model: str = "RN50",
    ):
        super().__init__()
        dummy_obs_space = spaces.Dict(
            {
                "rgb": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(224, 224, 3),
                    dtype=np.float32,
                ),
                ClipImageGoalSensor.cls_uuid: spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )
        self._encoder = ResNetCLIPGoalEncoder(
            dummy_obs_space,
            backbone_type="resnet50_avgattnpool",
            clip_model=clip_model,
        ).cuda()
        self.inference_worker_idx = inference_worker_idx
        print(
            "Initializing obs transforms for worker: {}...........".format(
                self.inference_worker_idx
            )
        )
        self._counter = 0
        self._episodes = {}
        self._max_steps = 250
        self._episode_embeddings = {}
        self._episode_step_counter = {}

    def transform_observation_space(self, observation_space: spaces.Dict, **kwargs):
        assert "rgb" in observation_space.spaces, "CLIP needs rgb observation!"
        observation_space.spaces[ClipImageGoalSensor.cls_uuid] = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1024,),
            dtype=np.float32,
        )
        observation_space.spaces.pop(CurrentEpisodeUUIDSensor.cls_uuid)
        return observation_space

    @classmethod
    def from_config(cls, config: DictConfig):
        return cls(
            backbone=config.backbone,
            inference_worker_idx=config.inference_worker_idx,
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        clip_image_goal = None

        clip_image_goals = []
        obs_batch = []
        episode_ids = []
        all_eps = []
        for idx, episode_id in enumerate(
            observations[CurrentEpisodeUUIDSensor.cls_uuid]
        ):
            episode_id = episode_id.item()
            self._episodes[episode_id] = 1
            if episode_id not in self._episode_embeddings:
                obs_batch.append(observations[ClipImageGoalSensor.cls_uuid][idx])
                episode_ids.append(episode_id)
            all_eps.append(episode_id)

        if len(obs_batch) > 0:
            with inference_mode():
                clip_image_goal = self._encoder(
                    {
                        ClipImageGoalSensor.cls_uuid: torch.stack(
                            obs_batch, dim=0
                        ).cuda(),
                    }
                )

            for idx, episode_id in enumerate(episode_ids):
                self._episode_embeddings[episode_id] = clip_image_goal[idx]
                self._episode_step_counter[episode_id] = 1

        for idx, episode_id in enumerate(
            observations[CurrentEpisodeUUIDSensor.cls_uuid]
        ):
            episode_id = episode_id.item()
            clip_image_goals.append(self._episode_embeddings[episode_id])

        observations[ClipImageGoalSensor.cls_uuid] = torch.stack(
            clip_image_goals, dim=0
        )
        observations.pop(CurrentEpisodeUUIDSensor.cls_uuid)

        del_eps = []
        episode_ids = list(self._episode_embeddings.keys())

        for episode_id in episode_ids:
            self._episode_step_counter[episode_id] += 1
            if self._episode_step_counter[episode_id] > self._max_steps:
                del_eps.append(episode_id)
                del self._episode_embeddings[episode_id]
                del self._episode_step_counter[episode_id]

        self._counter += 1

        if self._counter % 1000 == 0:
            print(
                "[Worker: {}] Cache: {}/{}".format(
                    self.inference_worker_idx,
                    len(self._episode_embeddings.keys()),
                    len(self._episodes.keys()),
                )
            )

        return observations


@dataclass
class CLIPImageGoalEncoderConfig(ObsTransformConfig):
    type: str = CLIPImageGoalEncoder.__name__
    backbone: str = "resnet50_clip_attnpool"
    inference_worker_idx: int = 0


cs = ConfigStore.instance()

cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.clip_imagegoal_encoder",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="clip_imagegoal_encoder",
    node=CLIPImageGoalEncoderConfig,
)
