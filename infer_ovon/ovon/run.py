#!/usr/bin/env python3

import argparse
import glob
import os
import os.path as osp

import torch
from habitat import get_config
from habitat.config import read_write
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from omegaconf import OmegaConf  # keep this import for print debugging

from ovon.config import ClipObjectGoalSensorConfig, HabitatConfigPlugin


def register_plugins():
    register_hydra_plugin(HabitatConfigPlugin)


def main():
    """Builds upon the habitat_baselines.run.main() function to add more flags
    for convenience."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--run-type",
        "-r",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        "-e",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Saves files to $JUNK directory and ignores resume state.",
    )
    parser.add_argument(
        "--single-env",
        "-s",
        action="store_true",
        help="Sets num_environments=1.",
    )
    parser.add_argument(
        "--debug-datapath",
        "-p",
        action="store_true",
        help="Uses faster-to-load $OVON_DEBUG_DATAPATH episode dataset for debugging.",
    )
    parser.add_argument(
        "--blind",
        "-b",
        action="store_true",
        help="If set, no cameras will be used.",
    )
    parser.add_argument(
        "--checkpoint-config",
        "-c",
        action="store_true",
        help=(
            "If set, checkpoint's config will be used, but overrides WILL be "
            "applied. Does nothing when training; meant for using ckpt config + "
            "overrides for eval."
        ),
    )
    parser.add_argument(
        "--text-goals",
        "-t",
        action="store_true",
        help="If set, only CLIP text goals will be used for evaluation.",
    )
    parser.add_argument(
        "--eval-analysis",
        "-v",
        action="store_true",
        help="If set, add semantic sensor for evaluation.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    # Register custom hydra plugin
    register_plugins()

    config = get_config(args.exp_config, args.opts)

    if args.run_type == "eval" and args.checkpoint_config:
        config = merge_config(config, args.opts)

    with read_write(config):
        edit_config(config, args)

    execute_exp(config, args.run_type)


def merge_config(config, opts):
    """There might be a better way to do this with Hydra... do I know it? No.
    1. Locate a checkpoint using the config's eval checkpoint path
    2. Load that checkpoint's config to replicate training config
    3. Save this config to a temporary file
    4. Use the path to the temporary file and the given override opts

    This is the only way to add overrides in eval that also use whatever
    overrides were used in training.
    """
    # 1. Locate a checkpoint using the config
    checkpoint_path = config.habitat_baselines.eval_ckpt_path_dir
    if osp.isdir(checkpoint_path):
        ckpt_files = glob.glob(osp.join(checkpoint_path, "*.pth"))
        assert len(ckpt_files) > 0, f"No checkpoints found in {checkpoint_path}!"
        checkpoint_path = ckpt_files[0]
    elif not osp.isfile(checkpoint_path):
        raise ValueError(f"Checkpoint path {checkpoint_path} is not a file!")

    # 2. Load the config from the checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_config = ckpt["config"]

    # 3. Save the given config to a temporary file
    randstr = str(torch.randint(0, 100000, (1,)).item())
    tmp_config_path = f"/tmp/ovon_config_{randstr}.yaml"
    OmegaConf.save(ckpt_config, tmp_config_path)

    # 4. Use the path to the temporary file as the config path and use the
    # given opts to override the config
    config = get_config(tmp_config_path, opts)
    os.remove(tmp_config_path)

    # Set load_resume_state_config to False so we don't load the checkpoint's
    # config again and lose the overrides
    with read_write(config):
        config.habitat_baselines.load_resume_state_config = False

    return config


def edit_config(config, args):
    if args.debug:
        assert osp.isdir(os.environ["JUNK"]), (
            "Environment variable directory $JUNK does not exist "
            f"(Current value: {os.environ['JUNK']})"
        )

        # Remove resume state in junk if training, so we don't resume from it
        resume_state_path = osp.join(os.environ["JUNK"], ".habitat-resume-state.pth")
        if args.run_type == "train" and osp.isfile(resume_state_path):
            print(
                "Removing junk resume state file:",
                osp.abspath(resume_state_path),
            )
            os.remove(resume_state_path)

        config.habitat_baselines.tensorboard_dir = os.environ["JUNK"]
        config.habitat_baselines.video_dir = os.environ["JUNK"]
        config.habitat_baselines.checkpoint_folder = os.environ["JUNK"]
        config.habitat_baselines.log_file = osp.join(os.environ["JUNK"], "junk.log")
        config.habitat_baselines.load_resume_state_config = False

    if args.debug_datapath:
        # Only load one scene for faster debugging
        scenes = "1UnKg1rAb8A" if args.run_type == "train" else "4ok3usBNeis"
        config.habitat.dataset.content_scenes = [scenes]

    if args.single_env:
        config.habitat_baselines.num_environments = 1

    # Remove all cameras if running blind (e.g., evaluating frontier explorer)
    if args.blind:
        for k in ["depth_sensor", "rgb_sensor"]:
            if k in config.habitat.simulator.agents.main_agent.sim_sensors:
                config.habitat.simulator.agents.main_agent.sim_sensors.pop(k)
        from habitat.config.default_structured_configs import (
            HabitatSimDepthSensorConfig,
        )

        # Camera required to load in a scene; use dummy 1x1 depth camera
        config.habitat.simulator.agents.main_agent.sim_sensors.update(
            {"depth_sensor": HabitatSimDepthSensorConfig(height=1, width=1)}
        )
        if hasattr(config.habitat_baselines.rl.policy, "obs_transforms"):
            config.habitat_baselines.rl.policy.obs_transforms = {}

    if args.eval_analysis:
        from habitat.config.default_structured_configs import (
            HabitatSimSemanticSensorConfig,
        )

        config.habitat.simulator.agents.main_agent.sim_sensors.update(
            {"semantic_sensor": HabitatSimSemanticSensorConfig(height=640, width=360)}
        )

    if (
        args.text_goals
        and args.run_type == "eval"
        and hasattr(config.habitat.task.lab_sensors, "clip_imagegoal_sensor")
    ):
        config.habitat.task.lab_sensors.pop("clip_imagegoal_sensor")
        if hasattr(config.habitat.task.lab_sensors, "clip_goal_selector_sensor"):
            config.habitat.task.lab_sensors.pop("clip_goal_selector_sensor")
        if not hasattr(config.habitat.task.lab_sensors, "clip_objectgoal_sensor"):
            config.habitat.task.lab_sensors.update(
                {"clip_objectgoal_sensor": ClipObjectGoalSensorConfig()}
            )

    if args.run_type == "train":
        for measure_name in ["frontier_exploration_map", "top_down_map"]:
            if hasattr(config.habitat.task.measurements, measure_name):
                print(
                    f"[run.py]: Removing {measure_name} measurement from "
                    "config to speed up training."
                )
                config.habitat.task.measurements.pop(measure_name)
    elif len(config.habitat_baselines.eval.video_option) == 0:
        # For eval, remove the objnav_explorer teacher
        if hasattr(config.habitat.task.lab_sensors, "objnav_explorer"):
            print("[run.py]: Removing objnav_explorer sensor from config for eval.")
            config.habitat.task.lab_sensors.pop("objnav_explorer")

        # Remove relabeler too
        if hasattr(
            config.habitat_baselines.rl.policy.obs_transforms, "relabel_teacher_actions"
        ):
            print("[run.py]: Removing relabel_teacher_actions from config for eval.")
            config.habitat_baselines.rl.policy.obs_transforms.pop(
                "relabel_teacher_actions"
            )


if __name__ == "__main__":
    main()
