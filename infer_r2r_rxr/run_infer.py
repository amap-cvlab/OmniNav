#!/usr/bin/env python3
import numpy as np
import argparse
import torch
from habitat.datasets import make_dataset
from VLN_CE.vlnce_baselines.config.default import get_config
from agent.waypoint_agent import evaluate_agent


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config",
        type=str,
        required=False,
        default="",
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--split-num",
        type=int,
        required=False,
        default=1,
        help="chunks of evluation"
    )
    
    parser.add_argument(
        "--split-id",
        type=int,
        required=False,
        default=1,
        help="chunks ID of evluation"

    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default="",
        help="location of model weights"

    )

    parser.add_argument(
        "--result-path",
        type=str,
        required=False,
        default="",
        help="location to save results"

    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, split_num: str, split_id: str, model_path: str, result_path: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """

    config = get_config(exp_config, opts)
            
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    
    np.random.seed(42)
    dataset_split = dataset.get_splits(split_num)[split_id]
    with torch.no_grad():
        evaluate_agent(config, split_id, dataset_split, model_path, result_path)


if __name__ == "__main__":
    main()
