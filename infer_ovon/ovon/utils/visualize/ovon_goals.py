import glob
import os
import random
from collections import defaultdict
from typing import Dict, List

import habitat_sim
from habitat.utils.render_wrapper import append_text_to_image
from habitat.utils.visualizations.utils import observations_to_image
from habitat_sim import bindings as hsim
from habitat_sim.agent.agent import AgentConfiguration, AgentState
from numpy import ndarray
from tqdm import tqdm

from ovon.utils.utils import load_dataset, save_image, write_json

SCENE_ROOT = "data/scene_datasets/"


def _compute_frame_coverage(obs: List[Dict[str, ndarray]], oid: int):
    def _single(obs, oid):
        mask = obs["semantic_sensor"] == oid
        return mask.sum() / mask.size

    if isinstance(obs, list):
        return [_single(o, oid) for o in obs]
    if isinstance(obs, dict):
        return _single(obs, oid)
    else:
        raise TypeError("argument `obs` must be either a list or a dict.")


def save_observations(file_path, out_path, object_goal_meta):
    scene_id = file_path.split("/")[-1]
    output_path = os.path.join(out_path, "{}".format(scene_id.split(".")[0]))

    data = load_dataset(file_path)
    sim = _config_sim(os.path.join(SCENE_ROOT, data["episodes"][0]["scene_id"]))
    print("Output path: {}".format(output_path))

    objects = [o for o in sim.semantic_scene.objects]

    os.makedirs(output_path, exist_ok=True)

    total_saved = 0
    for goals_key, goals in tqdm(data["goals_by_category"].items()):
        object_category = goals_key.split("_")[1]
        object_goal_meta[object_category].append(
            os.path.join(scene_id.split(".")[0], "{}.png".format(object_category))
        )

        max_coverage = 0
        observation = None
        for goal in goals:
            semantic_id = [o.semantic_id for o in objects if o.id == goal["object_id"]][
                0
            ]

            for view_point in goal["view_points"]:
                position = view_point["agent_state"]["position"]
                rotation = view_point["agent_state"]["rotation"]

                obs = sim.agents[0].set_state(
                    AgentState(position=position, rotation=rotation)
                )

                for act in ["look_down", "look_up", "look_up"]:
                    obs = sim.step(act)
                    cov = _compute_frame_coverage(obs, semantic_id)

                    if cov > max_coverage:
                        max_coverage = cov
                        observation = obs

        if observation is not None:
            observation = observations_to_image(
                {"rgb": observation["color_sensor"]}, {}
            )
            observation = append_text_to_image(
                observation,
                text=["object_category: {}".format(object_category)],
                font_size=0.75,
            )
            save_image(
                observation, os.path.join(output_path, "{}.png".format(object_category))
            )
            total_saved += 1
    print("Total observations saved: {}".format(total_saved))


def _config_sim(scene: str):
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_dataset_config_file = (
        "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    )
    sim_cfg.scene_id = scene

    sensor_specs = []
    for name, sensor_type in zip(
        ["color", "depth", "semantic"],
        [
            habitat_sim.SensorType.COLOR,
            habitat_sim.SensorType.DEPTH,
            habitat_sim.SensorType.SEMANTIC,
        ],
    ):
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = f"{name}_sensor"
        sensor_spec.sensor_type = sensor_type
        sensor_spec.resolution = [512, 512]
        sensor_spec.position = [0.0, 1.41, 0.0]
        sensor_spec.hfov = 100
        sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(sensor_spec)

    # create agent specifications
    agent_cfg = AgentConfiguration(
        height=1.41,
        radius=0.17,
        sensor_specifications=sensor_specs,
        action_space={
            "look_up": habitat_sim.ActionSpec(
                "look_up",
                habitat_sim.ActuationSpec(amount=30),
            ),
            "look_down": habitat_sim.ActionSpec(
                "look_down",
                habitat_sim.ActuationSpec(amount=30),
            ),
        },
    )

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    # set the navmesh
    assert sim.pathfinder.is_loaded, "pathfinder is not loaded!"
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_height = 1.41
    navmesh_settings.agent_radius = 0.17
    navmesh_settings.agent_max_climb = 0.10
    navmesh_settings.cell_height = 0.05
    navmesh_success = sim.recompute_navmesh(
        sim.pathfinder, navmesh_settings, include_static_objects=False
    )
    assert navmesh_success, "Failed to build the navmesh!"
    return sim


def save_object_goals_for_dataset(path, output_path):
    files = glob.glob(os.path.join(path, "*.json.gz"))
    object_goal_meta = defaultdict(list)
    for file in files:
        print("Validating {}".format(file))
        save_observations(file, output_path, object_goal_meta)

    object_goals = []
    for object_category, paths in object_goal_meta.items():
        object_goals.append(
            {
                "object_category": object_category,
                "object_goals": random.sample(paths, min(len(paths), 3)),
            }
        )

    write_json(object_goals, os.path.join(output_path, "object_goals.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()

    if args.path.endswith(".json.gz"):
        print("Validate episodeL: {}".format(args.path))
        save_observations(args.path, args.output_path)
    else:
        save_object_goals_for_dataset(args.path, args.output_path)
