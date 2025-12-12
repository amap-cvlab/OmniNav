from typing import TYPE_CHECKING

import torch
from habitat.utils import profiling_wrapper
from habitat_baselines import VERTrainer
from habitat_baselines.rl.ver.task_enums import ReportWorkerTasks
from habitat_baselines.rl.ver.timing import Timing
from habitat_baselines.utils.common import cosine_decay, inference_mode
from torch.optim.lr_scheduler import LambdaLR

from ovon.algos.ppo import DDPPO, PPO
from ovon.utils.lr_scheduler import PIRLNavLRScheduler
from ovon.utils.visualize.viz import overlay_frame

try:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
except AttributeError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig

import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch
import tqdm
from habitat import logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_num_actions,
    inference_mode,
    is_continuous_action_space,
)
from omegaconf import OmegaConf
from torch import nn

from ovon.measurements.nav import FailureModeMeasure, OVONObjectGoalID
from ovon.utils.utils import load_pickle

if TYPE_CHECKING:
    from omegaconf import DictConfig


@baseline_registry.register_trainer(name="ver_pirlnav")
class VERPIRLNavTrainer(VERTrainer):
    def _setup_actor_critic_agent(self, ppo_cfg: "DictConfig") -> None:
        r"""Sets up actor critic and agent for PPO.

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
            try:
                self.actor_critic.load_state_dict(
                    {  # type: ignore
                        k[len("actor_critic.") :]: v
                        for k, v in pretrained_state["state_dict"].items()
                    }
                )
            except:
                print("Fail to load pretrained weights. Trying again.")
                merged_dict = self.actor_critic.state_dict()
                merged_dict.update(
                    {  # type: ignore
                        k[len("actor_critic.") :]: v
                        for k, v in pretrained_state["state_dict"].items()
                        if k[len("actor_critic.") :] in merged_dict
                    }
                )
                self.actor_critic.load_state_dict(merged_dict)
                print("Successfully loaded pretrained weights.")
        elif self.config.habitat_baselines.rl.ddppo.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.habitat_baselines.rl.ddppo.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        policy_cfg = self.config.habitat_baselines.rl.policy
        if hasattr(policy_cfg, "finetune") and policy_cfg.finetune.enabled:
            self.actor_critic.freeze_visual_encoders()
            self.actor_critic.freeze_state_encoder()
            self.actor_critic.freeze_actor()
            self.actor_critic.freeze_new_params()

        if self.config.habitat_baselines.rl.ddppo.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = (DDPPO if self._is_distributed else PPO).from_config(
            self.actor_critic, ppo_cfg
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training VER using PIRLNav LR scheduler.

        Returns:
            None
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        self.num_steps_done = 0
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config = resume_state["config"]

            requeue_stats = resume_state["requeue_stats"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]

        self._init_train(resume_state)

        count_checkpoints = 0
        policy_cfg = self.config.habitat_baselines.rl.policy

        if hasattr(policy_cfg, "finetune") and policy_cfg.finetune.enabled:
            lr_scheduler = PIRLNavLRScheduler(
                optimizer=self.agent.optimizer,
                agent=self.agent,
                num_updates=self.config.habitat_baselines.num_updates,
                base_lr=self.config.habitat_baselines.rl.ppo.lr,
                finetuning_lr=policy_cfg.finetune.lr,
                ppo_eps=self.config.habitat_baselines.rl.ppo.eps,
                start_actor_update_at=policy_cfg.finetune.start_actor_update_at,
                start_actor_warmup_at=policy_cfg.finetune.start_actor_warmup_at,
                start_critic_update_at=policy_cfg.finetune.start_critic_update_at,
                start_critic_warmup_at=policy_cfg.finetune.start_critic_warmup_at,
            )
        else:
            lr_scheduler = LambdaLR(
                optimizer=self.agent.optimizer,
                lr_lambda=lambda x: cosine_decay(self.percent_done()),
            )

        if resume_state is not None:
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self._last_checkpoint_percent = requeue_stats["_last_checkpoint_percent"]
            count_checkpoints = requeue_stats["count_checkpoints"]

        ppo_cfg = self.config.habitat_baselines.rl.ppo
        if self.ver_config.overlap_rollouts_and_learn:
            self.preemption_decider.start_rollout()

        while not self.is_done():
            profiling_wrapper.on_start_step()

            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * (1 - self.percent_done())

            if rank0_only() and self._should_save_resume_state():
                requeue_stats = dict(
                    count_checkpoints=count_checkpoints,
                    num_steps_done=self.num_steps_done,
                    num_updates_done=self.num_updates_done,
                    _last_checkpoint_percent=self._last_checkpoint_percent,
                    report_worker_state=self.report_worker.state_dict(),
                )
                resume_state = dict(
                    state_dict=self.agent.state_dict(),
                    optim_state=self.agent.optimizer.state_dict(),
                    lr_sched_state=lr_scheduler.state_dict(),
                    config=self.config,
                    requeue_stats=requeue_stats,
                )

                save_resume_state(
                    resume_state,
                    self.config,
                )

            if EXIT.is_set():
                profiling_wrapper.range_pop()  # train update
                [w.close() for w in self._all_workers]
                [w.join() for w in self._all_workers]

                requeue_job()
                break

            with inference_mode():
                if not self.ver_config.overlap_rollouts_and_learn:
                    self.preemption_decider.start_rollout()
                    while not self.rollouts.rollout_done:
                        self._inference_worker_impl.try_one_step()

                    self._inference_worker_impl.finish_rollout()

                self._iw_sync.rollout_done.wait()
                self._iw_sync.rollout_done.clear()

                if self._iw_sync.all_workers.n_waiting > 0:
                    raise RuntimeError(
                        f"{self._iw_sync.all_workers.n_waiting} inference worker(s)"
                        " is(are) still waiting on the IW barrier. Likely they never"
                        " waited on it.\n"
                    )

                self.rollouts.after_rollout()

                if self.ver_config.overlap_rollouts_and_learn:
                    with self.timer.avg_time("overlap_transfers"):
                        self.learning_rollouts.copy(self.rollouts)

                self.preemption_decider.end_rollout(self.rollouts.num_steps_to_collect)

                self.queues.report.put(
                    (
                        ReportWorkerTasks.num_steps_collected,
                        int(self.rollouts.num_steps_collected),
                    )
                )

                if self.ver_config.overlap_rollouts_and_learn:
                    with self.timer.avg_time("overlap_transfers"):
                        self.rollouts.after_update()
                        self._iw_sync.should_start_next.set()
                        self.preemption_decider.start_rollout()

            losses = self._update_agent()
            lrs = {}
            for i, param_group in enumerate(self.agent.optimizer.param_groups):
                lrs["lr_{}".format(i)] = param_group["lr"]

            learner_metrics = {
                **losses,
                **lrs,
            }

            self.preemption_decider.learner_time(self._learning_time)

            self.queues.report.put_many(
                (
                    (
                        ReportWorkerTasks.learner_timing,
                        self.timer,
                    ),
                    (
                        ReportWorkerTasks.learner_update,
                        learner_metrics,
                    ),
                )
            )
            self.timer = Timing()

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()  # type: ignore

            self.num_steps_done = int(self.report_worker.num_steps_done)

            self.num_updates_done += 1
            # checkpoint model
            if rank0_only() and self.should_checkpoint():
                self.save_checkpoint(
                    f"ckpt.{count_checkpoints}.pth",
                    dict(
                        step=self.num_steps_done,
                        wall_time=self.report_worker.time_taken,
                    ),
                )
                count_checkpoints += 1

        self.window_episode_stats = self.report_worker.get_window_episode_stats()

        [w.close() for w in self._all_workers]
        [w.join() for w in self._all_workers]

        if self._is_distributed:
            torch.distributed.barrier()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        if self.config.habitat_baselines.eval.should_load_ckpt:
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict["config"] = None

        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])

        ppo_cfg = config.habitat_baselines.rl.ppo

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if (
            len(config.habitat_baselines.video_render_views) > 0
            and len(self.config.habitat_baselines.eval.video_option) > 0
        ):
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            render_view_uuids = [
                agent_sensors[render_view].uuid
                for render_view in config.habitat_baselines.video_render_views
                if render_view in agent_sensors
            ]
            assert len(render_view_uuids) > 0, (
                "Missing render sensors in agent config: "
                f"{config.habitat_baselines.video_render_views}."
            )
            with read_write(config):
                for render_view_uuid in render_view_uuids:
                    if render_view_uuid not in config.habitat.gym.obs_keys:
                        config.habitat.gym.obs_keys.append(render_view_uuid)
                config.habitat.simulator.debug_render = True

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        self._init_envs(config, is_eval=True)

        action_space = self.envs.action_spaces[0]
        self.policy_action_space = action_space
        self.orig_policy_action_space = self.envs.orig_action_spaces[0]
        if is_continuous_action_space(action_space):
            # Assume NONE of the actions are discrete
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = (1,)
            discrete_actions = True

        self._setup_actor_critic_agent(ppo_cfg)

        if self.agent.actor_critic.should_load_agent_state:
            if "actor_critic.critic.fc.weight" not in ckpt_dict["state_dict"]:
                print("Critic weights not found in checkpoint. Using default weights")
                merged_dict = self.agent.state_dict()
                merged_dict.update(ckpt_dict["state_dict"])
                self.agent.load_state_dict(merged_dict)
            else:
                self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            self.config.habitat_baselines.num_environments,
            self.actor_critic.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[Any, Any] = (
            {}
        )  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        rgb_frames = [
            [] for _ in range(self.config.habitat_baselines.num_environments)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self.actor_critic.eval()
        num_successes = 0
        num_total = 0
        json_dict = {}
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and self.envs.num_envs > 0
        ):
            current_episodes_info = self.envs.current_episodes()

            with inference_mode():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(self.policy_action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self.policy_action_space.low,
                        self.policy_action_space.high,
                    )
                    for a in actions.cpu()
                ]
            else:
                step_data = [a.item() for a in actions.cpu()]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            policy_info = self.actor_critic.get_policy_info(infos, dones)
            for i in range(len(policy_info)):
                infos[i].update(policy_info[i])
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                target_obj = None
                # We would like to write the object name on the frame too
                # if we are doing objectnav
                if ObjectGoalSensor.cls_uuid in observations[i]:
                    obj_id = observations[i][ObjectGoalSensor.cls_uuid][0]
                    id_to_name = [
                        "chair",
                        "bed",
                        "potted_plant",
                        "toilet",
                        "tv",
                        "couch",
                    ]
                elif OVONObjectGoalID.cls_uuid in infos[i]:
                    obj_id = infos[i][OVONObjectGoalID.cls_uuid]
                    if not hasattr(self, "objectgoal_vocab"):
                        cache = load_pickle(
                            "data/clip_embeddings/ovon_stretch_final_cache.pkl"
                        )
                        self.objectgoal_vocab = sorted(list(cache.keys()))
                    id_to_name = self.objectgoal_vocab
                else:
                    id_to_name = None
                if id_to_name is not None:
                    target_obj = id_to_name[obj_id]
                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    if not not_done_masks[i].item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()}, infos[i]
                        )
                    overlay_dict = self._extract_scalars_from_info(infos[i])
                    assert target_obj is not None
                    overlay_dict["target"] = target_obj
                    frame = overlay_frame(frame, overlay_dict)
                    rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(self._extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if episode_stats["success"] == 1:
                        num_successes += 1
                    num_total += 1
                    print(
                        f"Success rate: {num_successes/num_total*100:.2f}% "
                        f"({num_successes} out of {num_total})"
                    )
                    # Add data about the episode to json dict
                    if os.environ.get("OVON_EPISODES_JSON", "") != "":
                        assert target_obj is not None
                        ep_json_dict = {
                            "scene_id": current_episodes_info[i].scene_id,
                            "episode_id": current_episodes_info[i].episode_id,
                            "target": target_obj,
                        }
                        ep_json_dict.update(self._extract_scalars_from_info(infos[i]))

                        for key, val in infos[i].items():
                            if FailureModeMeasure.cls_uuid in key:
                                ep_json_dict[key] = val
                        logger.info(ep_json_dict)
                        hash_key_str = (
                            f"{current_episodes_info[i].scene_id}"
                            f"_{current_episodes_info[i].episode_id}"
                        )
                        if hash_key_str in json_dict:
                            hash_key_str += "_1"
                        json_dict[hash_key_str] = ep_json_dict

                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        video_metrics = self._extract_scalars_from_info(infos[i])
                        if target_obj is not None:
                            video_metrics[OVONObjectGoalID.cls_uuid] = target_obj

                        generate_video(
                            video_option=self.config.habitat_baselines.eval.video_option,
                            video_dir=self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes_info[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=video_metrics,
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values()]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        print("OVON json: {}".format(os.environ.get("OVON_EPISODES_JSON", None)))

        if os.environ.get("OVON_EPISODES_JSON", "") != "":
            import json

            json_file = os.environ["OVON_EPISODES_JSON"]
            with open(json_file, "w") as f:
                json.dump(json_dict, f)

        self.envs.close()
