import numpy as np
import argparse
import torch
import habitat
import habitat.config
import habitat.config.default
from habitat.datasets import make_dataset
from habitat import Env

from ovon.dataset import OVONDatasetV1
# from ovon.task.simulator import OVONHabitatConfig
from omegaconf import DictConfig, OmegaConf
from ovon.config import OVONDistanceToGoalConfig
from habitat import Env

from agent.waypoint_agent_ovon import evaluate_agent_ovon


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value, got {value!r}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--split-num",
        type=int,
        required=True,
        help="chunks of evluation"
    )

    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
        help="chunks ID of evluation"

    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="location of model weights"

    )

    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="location to save results"

    )
    parser.add_argument(
        "--flow-match",
        type=str_to_bool,
        default=False,
        help="whether to use the Flow Matching waypoint inference branch",
    )
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(
        exp_config: str, split_num: str, split_id: str, model_path: str,
        result_path: str, flow_match: bool = False) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """

    config_all = habitat.get_config(exp_config)
    config = config_all.habitat
    dataset = habitat.make_dataset(config.dataset.type, config=config.dataset)

    np.random.seed(42)
    dataset_split = dataset.get_splits(split_num)[split_id]
    with torch.no_grad():
        evaluate_agent_ovon(
            config,
            split_id,
            dataset_split,
            model_path,
            result_path,
            flow_match_enabled=flow_match,
        )


if __name__ == "__main__":
    main()
