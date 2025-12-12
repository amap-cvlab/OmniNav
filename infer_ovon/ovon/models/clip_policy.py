from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions
from torch import nn as nn

from ovon.models.encoders.cma_xattn import CrossModalAttention
from ovon.models.encoders.cross_attention import CrossAttention
from ovon.models.encoders.make_encoder import make_encoder
from ovon.task.sensors import ClipObjectGoalSensor

if TYPE_CHECKING:
    from omegaconf import DictConfig


class FusionType:
    _possible_fusions = [
        "concat",
        "cross_attention",
        "cross_attention_cma",
        "cross_attention_concat",
        "concat_late_fusion",
    ]

    def __init__(self, fusion_type: str):
        assert fusion_type in self._possible_fusions
        self._fusion_type = fusion_type

    @property
    def concat(self):
        return "concat" in self._fusion_type

    @property
    def xattn(self):
        return "cross_attention" in self._fusion_type

    @property
    def late_fusion(self):
        return "late_fusion" in self._fusion_type

    @property
    def cma(self):
        return "cma" in self._fusion_type


@baseline_registry.register_policy
class PointNavResNetCLIPPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        backbone: str = "clip_avgpool",
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        depth_ckpt: str = "",
        fusion_type: str = "cross_attention",
        attn_heads: int = 3,
        use_vis_query: bool = True,
        use_residual: bool = True,
        residual_vision: bool = True,
        unfreeze_xattn: bool = False,
        rgb_only: bool = True,
        use_prev_action: bool = True,
        use_odom: bool = False,
        **kwargs,
    ):
        self.unfreeze_xattn = unfreeze_xattn
        if policy_config is not None:
            discrete_actions = policy_config.action_distribution_type == "categorical"
            self.action_distribution_type = policy_config.action_distribution_type
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        super().__init__(
            OVONNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                discrete_actions=discrete_actions,
                depth_ckpt=depth_ckpt,
                fusion_type=fusion_type,
                attn_heads=attn_heads,
                use_vis_query=use_vis_query,
                use_residual=use_residual,
                residual_vision=residual_vision,
                rgb_only=rgb_only,
                use_prev_action=use_prev_action,
                use_odom=use_odom,
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
        # If training resnet encoder from scratch, assert train_encoder=True
        if config.habitat_baselines.rl.policy.backbone == "resnet":
            assert config.habitat_baselines.rl.ddppo.train_encoder, (
                "When training resnet encoder from scratch, "
                "train_encoder must be set to True."
            )

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

        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            backbone=config.habitat_baselines.rl.policy.backbone,
            policy_config=config.habitat_baselines.rl.policy,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            depth_ckpt=config.habitat_baselines.rl.policy.depth_ckpt,
            fusion_type=config.habitat_baselines.rl.policy.fusion_type,
            attn_heads=config.habitat_baselines.rl.policy.attn_heads,
            use_vis_query=config.habitat_baselines.rl.policy.use_vis_query,
            use_residual=config.habitat_baselines.rl.policy.use_residual,
            residual_vision=config.habitat_baselines.rl.policy.residual_vision,
            unfreeze_xattn=config.habitat_baselines.rl.policy.unfreeze_xattn,
            rgb_only=config.habitat_baselines.rl.policy.rgb_only,
            use_prev_action=config.habitat_baselines.rl.policy.get(
                "use_prev_action", True
            ),
            use_odom=config.habitat_baselines.rl.policy.get("use_odom", False),
            **kwargs,
        )

    def freeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(False)
        for param in self.net.visual_fc.parameters():
            param.requires_grad_(False)

    def unfreeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(True)
        for param in self.net.visual_fc.parameters():
            param.requires_grad_(True)

    def freeze_state_encoder(self):
        state_encoder_params = []
        blacklisted_layers = [
            "visual_encoder",
            "action_distribution",
            "critic",
            "visual_fc",
            "gps_embedding",
            "compass_embedding",
            "cross_attention",
            "prev_action_embedding",
        ]
        if self.unfreeze_xattn:
            blacklisted_layers.remove("cross_attention")

        whitelisted_names = []
        for name, param in self.net.named_parameters():
            is_blacklisted = False
            for layer in blacklisted_layers:
                if layer in name:
                    is_blacklisted = True
                    break
            if not is_blacklisted:
                state_encoder_params.append(param)
                whitelisted_names.append(name)

        for param in state_encoder_params:
            param.requires_grad_(False)
        print("freze", whitelisted_names)

    def unfreeze_state_encoder(self):
        state_encoder_params = []
        blacklisted_layers = [
            "visual_encoder",
            "action_distribution",
            "critic",
            "visual_fc",
            "gps_embedding",
            "compass_embedding",
            "cross_attention",
            "prev_action_embedding",
        ]
        if self.unfreeze_xattn:
            blacklisted_layers.remove("cross_attention")

        whitelisted_names = []
        for name, param in self.net.named_parameters():
            is_blacklisted = False
            for layer in blacklisted_layers:
                if layer in name:
                    is_blacklisted = True
                    break
            if not is_blacklisted:
                state_encoder_params.append(param)
                whitelisted_names.append(name)

        for param in state_encoder_params:
            param.requires_grad_(True)
        print("unf", whitelisted_names)

    def freeze_new_params(self):
        state_encoder_params = []
        whitelisted_layers = [
            "gps_embedding",
            "compass_embedding",
            "cross_attention",
            "prev_action_embedding",
        ]
        if self.unfreeze_xattn:
            whitelisted_layers.remove("cross_attention")

        whitelisted_names = []
        for name, param in self.net.named_parameters():
            for layer in whitelisted_layers:
                if layer in name:
                    state_encoder_params.append(param)
                    whitelisted_names.append(name)
                    break

        for param in state_encoder_params:
            param.requires_grad_(False)
        print("fnew", whitelisted_names)

    def freeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(False)

    def unfreeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(True)


class OVONNet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone: str,
        discrete_actions: bool = True,
        fusion_type: str = "concat",
        clip_embedding_size: int = 1024,  # Target category CLIP embedding size
        attn_heads: int = 3,
        use_vis_query: bool = False,
        use_residual: bool = True,
        residual_vision: bool = False,
        rgb_only: bool = True,
        use_prev_action: bool = True,
        use_odom: bool = False,
        *args,
        **kwargs,
    ):
        print("Observation space info:")
        for k, v in observation_space.spaces.items():
            print(f"  {k}: {v}")

        super().__init__()
        self.discrete_actions = discrete_actions
        self._fusion_type = FusionType(fusion_type)
        self._hidden_size = hidden_size
        self._rgb_only = rgb_only

        self._use_prev_action = not (rgb_only or not use_prev_action)
        self._use_odom = not (rgb_only or not use_odom)

        # Embedding layer for previous action
        self._n_prev_action = 32
        rnn_input_size_info = {}
        if self._use_prev_action:
            rnn_input_size_info["prev_action"] = self._n_prev_action
            if discrete_actions:
                self.prev_action_embedding = nn.Embedding(
                    action_space.n + 1, self._n_prev_action
                )
            else:
                num_actions = get_num_actions(action_space)
                self.prev_action_embedding = nn.Linear(num_actions, self._n_prev_action)

        # Visual encoder
        self.visual_encoder = make_encoder(backbone, observation_space)
        if backbone in ["clip_attnpool", "siglip"]:
            self.visual_fc = nn.Identity()
            if backbone == "clip_attnpool":
                clip_embedding_size = 1024
            else:
                clip_embedding_size = 768
            visual_feats_size = clip_embedding_size
        else:
            if backbone == "resnet":
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
                    nn.ReLU(True),
                )
            else:
                self.visual_fc = nn.Sequential(
                    nn.Linear(self.visual_encoder.output_size, hidden_size),
                    nn.ReLU(True),
                )
            visual_feats_size = hidden_size

        # Optional Compass embedding layer
        if (
            EpisodicCompassSensor.cls_uuid in observation_space.spaces
            and self._use_odom
        ):
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[0] == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size_info["compass_embedding"] = 32

        # Optional GPS embedding layer
        if EpisodicGPSSensor.cls_uuid in observation_space.spaces and self._use_odom:
            input_gps_dim = observation_space.spaces[EpisodicGPSSensor.cls_uuid].shape[
                0
            ]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size_info["gps_embedding"] = 32

        # Optional cross-attention layer
        if self._fusion_type.concat:
            rnn_input_size_info["visual_feats"] = visual_feats_size
        elif self._fusion_type.xattn:
            if backbone in ["clip_attnpool", "siglip"]:
                if backbone == "clip_attnpool":
                    embed_dim = 1024
                else:
                    embed_dim = 768
                assert clip_embedding_size == embed_dim
                assert visual_feats_size == embed_dim
            else:
                embed_dim = None
            if self._fusion_type.cma:
                self.cross_attention = CrossModalAttention(
                    text_embedding_dim=clip_embedding_size,
                    rgb_embedding_dim=visual_feats_size,
                    hidden_size=512,
                )
            else:
                self.cross_attention = CrossAttention(
                    x1_dim=clip_embedding_size,
                    x2_dim=visual_feats_size,
                    num_heads=attn_heads,
                    use_vis_query=use_vis_query,
                    use_residual=use_residual,
                    residual_vision=residual_vision,
                    embed_dim=embed_dim,
                )
            rnn_input_size_info["visual_feats"] = self.cross_attention.output_size
        else:
            raise NotImplementedError(f"Unknown fusion type: {fusion_type}")

        assert ClipObjectGoalSensor.cls_uuid in observation_space.spaces
        if not self._fusion_type.late_fusion and not self._fusion_type.xattn:
            rnn_input_size_info["clip_goal"] = clip_embedding_size

        # Report the type and sizes of the inputs to the RNN
        self.rnn_input_size = sum(rnn_input_size_info.values())
        print("RNN input size info: ")
        for k, v in rnn_input_size_info.items():
            print(f"  {k}: {v}")
        total_str = f"  Total RNN input size: {self.rnn_input_size}"
        print("  " + "-" * (len(total_str) - 2) + "\n" + total_str)

        self.rnn_type = rnn_type
        self._num_recurrent_layers = num_recurrent_layers
        self.state_encoder = self.build_state_encoder()

        print(
            "State encoder parameters: ",
            sum(p.numel() for p in self.state_encoder.parameters()),
        )

        if self._fusion_type.late_fusion:
            self.late_fusion_fc = nn.Sequential(
                nn.Linear(clip_embedding_size, hidden_size),
                nn.ReLU(True),
            )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def build_state_encoder(self):
        return build_rnn_state_encoder(
            self.rnn_input_size,
            self._hidden_size,
            rnn_type=self.rnn_type,
            num_layers=self._num_recurrent_layers,
        )

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
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

        visual_feats = self.visual_fc(visual_feats)
        object_goal = observations[ClipObjectGoalSensor.cls_uuid]

        if self._fusion_type.xattn:
            visual_feats = self.cross_attention(object_goal, visual_feats)

        x = [visual_feats]

        if self._fusion_type.concat and not self._fusion_type.late_fusion:
            x.append(object_goal)

        if EpisodicCompassSensor.cls_uuid in observations and self._use_odom:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(self.compass_embedding(compass_observations.squeeze(dim=1)))

        if EpisodicGPSSensor.cls_uuid in observations and self._use_odom:
            x.append(self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid]))

        if self._use_prev_action:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            # The mask means the previous action will be zero, an extra dummy action
            prev_actions = self.prev_action_embedding(
                torch.where(masks[:, -1:].view(-1), prev_actions + 1, start_token)
            )

            x.append(prev_actions)

        out = torch.cat(x, dim=1)

        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )

        if self._fusion_type.late_fusion:
            out = (out + visual_feats) * self.late_fusion_fc(object_goal)

        aux_loss_state = {
            "rnn_output": out,
            "perception_embed": visual_feats,
        }

        return out, rnn_hidden_states, aux_loss_state
