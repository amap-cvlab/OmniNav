from typing import Optional

from habitat_baselines import RolloutStorage
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from torch import nn


class RolloutStorageNo2D(RolloutStorage):
    """RolloutStorage variant that will store visual features extracted using a given
    visual encoder instead of raw images to save space."""

    buffers: TensorDict
    visual_encoder: nn.Module

    def __init__(
        self, visual_encoder, initial_obs: Optional[TensorDict] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Remove any 2D observations from the rollout storage buffer
        delete_keys = []
        for sensor in self.buffers["observations"]:
            if self.buffers["observations"][sensor].dim() >= 4:  # NCHW -> 4 dims
                delete_keys.append(sensor)
        for k in delete_keys:
            del self.buffers["observations"][k]
        self.visual_encoder = visual_encoder
        if initial_obs is not None:
            self.buffers["observations"][0] = self.filter_obs(initial_obs)

    def filter_obs(self, obs: TensorDict) -> TensorDict:
        filtered_obs = TensorDict()
        for sensor in obs:
            # Filter out 2D observations
            if obs[sensor].dim() < 4:
                filtered_obs[sensor] = obs[sensor]
        # Extract visual features from 2D observations
        filtered_obs[PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY] = (
            self.visual_encoder(obs)
        )
        return filtered_obs

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
    ):
        if next_observations is not None:
            filtered_next_observations = self.filter_obs(next_observations)
        else:
            filtered_next_observations = None
        super().insert(  # noqa
            filtered_next_observations,
            next_recurrent_hidden_states,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            next_masks,
            buffer_index,
        )
