import torch
from habitat import logger
from habitat.config import read_write
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet  # noqa: F401.
from omegaconf import DictConfig, open_dict

from ovon.algos.dagger import DAgger, DAggerPolicy, DDPDAgger
from ovon.trainers.ver_transformer_trainer import VERTransformerTrainer

try:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
except AttributeError:
    pass


@baseline_registry.register_trainer(name="ver_dagger")
@baseline_registry.register_trainer(name="ver_il")
class VERDAggerTrainer(VERTransformerTrainer):
    def __init__(self, config: DictConfig):
        with open_dict(config.habitat_baselines.rl.policy):  # allow new keys
            with read_write(config.habitat_baselines.rl.policy):  # allow write
                if hasattr(config.habitat_baselines.rl.policy, "original_name"):
                    original_name = config.habitat_baselines.rl.policy.original_name
                else:
                    original_name = config.habitat_baselines.rl.policy.name
                config.habitat_baselines.rl.policy["original_name"] = (
                    # add new key "original_name"
                    original_name
                )
                config.habitat_baselines.rl.policy["teacher_forcing"] = (
                    # add new key "teacher_forcing"
                    config.habitat_baselines.trainer_name
                    == "ver_il"
                )
                config.habitat_baselines.rl.policy.name = DAggerPolicy.__name__
        super().__init__(config)

    def _setup_actor_critic_agent(self, ppo_cfg: "DictConfig") -> None:
        r"""Same as PPOTrainer._setup_actor_critic_agent but mixes the policy class with
        DAgger mixin so that evaluate_actions induces the correct gradients, and allows
        the usage of DAgger agent instead of DDPPO or PPO. Also, the critic is gone, so
        we don't need to reset it.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.habitat_baselines.log_file)

        policy = baseline_registry.get_policy(
            self.config.habitat_baselines.rl.policy.name
        )
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.actor_critic = policy.from_config(
            self.config,
            observation_space,
            self.policy_action_space,
            orig_action_space=self.orig_policy_action_space,
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if (
            self.config.habitat_baselines.rl.ddppo.pretrained_encoder
            or self.config.habitat_baselines.rl.ddppo.pretrained
        ):
            pretrained_state = torch.load(
                self.config.habitat_baselines.rl.ddppo.pretrained_weights,
                map_location="cpu",
            )

        if self.config.habitat_baselines.rl.ddppo.pretrained:
            self.actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.habitat_baselines.rl.ddppo.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.habitat_baselines.rl.ddppo.train_encoder and hasattr(
            self.actor_critic, "net"
        ):
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        self.agent = (DDPDAgger if self._is_distributed else DAgger).from_config(
            self.actor_critic, ppo_cfg
        )
