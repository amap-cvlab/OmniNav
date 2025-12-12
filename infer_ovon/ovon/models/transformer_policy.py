from typing import Dict, Optional, Tuple

import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import NetPolicy
from omegaconf import DictConfig

from ovon.models.clip_policy import OVONNet, PointNavResNetCLIPPolicy
from ovon.models.transformer_encoder import TransformerEncoder


@baseline_registry.register_policy
class OVONTransformerPolicy(PointNavResNetCLIPPolicy):
    is_transformer = True

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        transformer_config,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        backbone: str = "clip_avgpool",
        policy_config: DictConfig = None,
        aux_loss_config: Optional[DictConfig] = None,
        depth_ckpt: str = "",
        fusion_type: str = "concat",
        attn_heads: int = 3,
        use_vis_query: bool = False,
        use_residual: bool = True,
        residual_vision: bool = False,
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

        NetPolicy.__init__(
            self,
            OVONTransformerNet(
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
                transformer_config=transformer_config,
                rgb_only=rgb_only,
                use_prev_action=use_prev_action,
                use_odom=use_odom,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(cls, config: DictConfig, *args, **kwargs):
        tf_cfg = config.habitat_baselines.rl.policy.transformer_config
        return super().from_config(config, transformer_config=tf_cfg, *args, **kwargs)

    @property
    def num_recurrent_layers(self):
        return self.net.state_encoder.n_layers

    @property
    def num_heads(self):
        return self.net.state_encoder.n_head

    @property
    def max_context_length(self):
        return self.net.state_encoder.max_context_length

    @property
    def recurrent_hidden_size(self):
        return self.net.state_encoder.n_embed


class OVONTransformerNet(OVONNet):
    """Same as OVONNet but uses transformer instead of LSTM."""

    def __init__(self, transformer_config, *args, **kwargs):
        self.transformer_config = transformer_config
        super().__init__(*args, **kwargs)

    @property
    def output_size(self):
        return self.transformer_config.n_hidden

    def build_state_encoder(self):
        state_encoder = TransformerEncoder(
            self.rnn_input_size, config=self.transformer_config
        )
        return state_encoder

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if "step_id" in observations:
            if rnn_build_seq_info is None:
                # Means online inference. Update should already have "episode_ids" key.
                rnn_build_seq_info = {}
            rnn_build_seq_info["step_id"] = observations["step_id"]
        return super().forward(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            rnn_build_seq_info,
        )
