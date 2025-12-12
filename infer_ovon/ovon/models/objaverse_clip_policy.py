from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from gym import spaces
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions
from torch import nn as nn

from ovon.models.encoders.clip_encoder import ResNetCLIPEncoder
from ovon.task.sensors import ClipObjectGoalSensor


@baseline_registry.register_policy
class ObjaverseCLIPPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        depth_ckpt: str = "",
        **kwargs,
    ):
        if policy_config is not None:
            discrete_actions = policy_config.action_distribution_type == "categorical"
            self.action_distribution_type = policy_config.action_distribution_type
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        super().__init__(
            ObjaverseCLIPNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                discrete_actions=discrete_actions,
                depth_ckpt=depth_ckpt,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # Exclude cameras for rendering from the observation space.
        ignore_names: List[str] = []
        for agent_config in config.habitat.simulator.agents.values():
            ignore_names.extend(
                agent_config.sim_sensors[k].uuid
                for k in config.habitat_baselines.video_render_views
                if k in agent_config.sim_sensors
            )
        filtered_obs = spaces.Dict(
            OrderedDict(
                ((k, v) for k, v in observation_space.items() if k not in ignore_names)
            )
        )
        try:
            """TODO: Eventually eliminate this check. Safeguard for loading
            configs from checkpoints created before commit where depth_ckpt was
            added to OVONPolicyConfig
            """
            depth_ckpt = config.habitat_baselines.rl.policy.depth_ckpt
        except:
            depth_ckpt = ""
        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            policy_config=config.habitat_baselines.rl.policy,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            depth_ckpt=depth_ckpt,
        )


class ObjaverseCLIPNet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        discrete_actions: bool = True,
        clip_model: str = "RN50",
        depth_ckpt: str = "",
        raw_compressor_hid_out_dims: Tuple[int, int] = (2048, 256),
        combine_hid_out_dims: Tuple[int, int] = (256, 64),
        goal_compressor_out_dims=256,
    ):
        super().__init__()
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self.goal_compressor_out_dims = goal_compressor_out_dims
        self.combiner_dims = self.goal_compressor_out_dims
        self._n_prev_action = 32
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(num_actions, self._n_prev_action)
        rnn_input_size = self._n_prev_action  # test
        rnn_input_size_info = {"prev_action": self._n_prev_action}

        self.visual_encoder = ResNetCLIPEncoder(
            observation_space,
            backbone_type="none",
            clip_model=clip_model,
            depth_ckpt=depth_ckpt,
        )
        self.visual_encoder_out_dims = self.visual_encoder.output_shape

        self.visual_compressor = nn.Sequential(
            nn.Conv2d(
                self.visual_encoder_out_dims[0],
                raw_compressor_hid_out_dims[0],
                1,
            ),
            nn.ReLU(),
            nn.Conv2d(*raw_compressor_hid_out_dims[0:2], 1),
            nn.ReLU(),
        )
        self.combiner_dims += raw_compressor_hid_out_dims[-1]

        self.combiner = nn.Sequential(
            nn.Conv2d(
                self.combiner_dims,
                combine_hid_out_dims[0],
                1,
            ),
            nn.ReLU(),
            nn.Conv2d(*combine_hid_out_dims[0:2], 1),
        )
        combined_embedding = (
            self.visual_encoder_out_dims[-1] ** 2 * combine_hid_out_dims[1]
        )
        rnn_input_size += combined_embedding
        rnn_input_size_info["combined_embedding"] = combined_embedding
        print("Obs space: {}".format(observation_space.spaces))

        clip_embedding = 1024 if clip_model == "RN50" else 768
        self.goal_compressor = nn.Linear(clip_embedding, self.goal_compressor_out_dims)

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[EpisodicGPSSensor.cls_uuid].shape[
                0
            ]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            rnn_input_size_info["gps_embedding"] = 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[0] == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32
            rnn_input_size_info["compass_embedding"] = 32

        self._hidden_size = hidden_size

        print("RNN input size info: ")
        total = 0
        for k, v in rnn_input_size_info.items():
            print(f"  {k}: {v}")
            total += v
        if total - rnn_input_size != 0:
            print(f"  UNACCOUNTED: {total - rnn_input_size}")
        total_str = f"  Total RNN input size: {total}"
        print("  " + "-" * (len(total_str) - 2))
        print(total_str)

        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def compress_goal(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        return self.goal_compressor(
            observations[ClipObjectGoalSensor.cls_uuid].type(torch.float32)
        ).view(-1, self.goal_compressor_out_dims, 1, 1)

    def distribute(self, obs):
        return obs.expand(
            -1,
            -1,
            self.visual_encoder_out_dims[-2],
            self.visual_encoder_out_dims[-1],
        )

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        aux_loss_state = {}
        # We CANNOT use observations.get() here because
        # self.visual_encoder(observations) is an expensive operation. Therefore,
        # we need `# noqa: SIM401`
        if (  # noqa: SIM401
            PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in observations
        ):
            visual_feats = observations[
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
            ]
        else:
            visual_feats = self.visual_encoder(observations)

        embs = [
            self.distribute(self.compress_goal(observations)),
            self.visual_compressor(visual_feats),
        ]
        combined = self.combiner(torch.cat(embs, dim=1))
        combined = combined.reshape(combined.shape[0], -1)

        aux_loss_state["perception_embed"] = combined
        x.append(combined)

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(self.compass_embedding(compass_observations.squeeze(dim=1)))

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid]))

        prev_actions = prev_actions.squeeze(-1)
        start_token = torch.zeros_like(prev_actions)
        # The mask means the previous action will be zero, an extra dummy action
        prev_actions = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token)
        )

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state
