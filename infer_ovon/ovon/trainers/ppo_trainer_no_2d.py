import numpy as np
import torch
from gym import spaces
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet

from ovon.utils.rollout_storage_no_2d import RolloutStorageNo2D


@baseline_registry.register_trainer(name="ddppo_no_2d")
@baseline_registry.register_trainer(name="ppo_no_2d")
class PPONo2DTrainer(PPOTrainer):
    def _init_train(self, *args, **kwargs):
        super()._init_train(*args, **kwargs)
        # Hacky overwriting of existing RolloutStorage with a new one
        ppo_cfg = self.config.habitat_baselines.rl.ppo
        action_shape = self.rollouts.buffers["actions"].shape[2:]
        discrete_actions = self.rollouts.buffers["actions"].dtype == torch.long
        batch = self.rollouts.buffers["observations"][0]

        obs_space = spaces.Dict(
            {
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY: spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=self._encoder.output_shape,
                    dtype=np.float32,
                ),
                **self.obs_space.spaces,
            }
        )

        self.rollouts = RolloutStorageNo2D(
            self.actor_critic.net.visual_encoder,
            batch,
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)
