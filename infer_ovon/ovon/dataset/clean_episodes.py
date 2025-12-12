import copy
import glob
import os
from typing import Sequence

import habitat_sim
import numpy as np
from habitat_sim import bindings as hsim
from habitat_sim.agent.agent import AgentConfiguration
from tqdm import tqdm

from ovon.utils.utils import load_dataset, write_dataset

SCENE_ROOT = "data/scene_datasets/"


def validate_episodes(file_path, output_path):
    scene_id = file_path.split("/")[-1]
    os.makedirs(output_path, exist_ok=True)

    output_path = os.path.join(output_path, scene_id)

    data = load_dataset(file_path)
    sim = _config_sim(os.path.join(SCENE_ROOT, data["episodes"][0]["scene_id"]))
    print("Output path: {}".format(output_path))

    valid_episodes = []
    for ep in tqdm(data["episodes"]):
        try:
            scene_id = ep["scene_id"]
            goal_key = "{}_{}".format(scene_id.split("/")[-1], ep["object_category"])

            goals = copy.deepcopy(data["goals_by_category"][goal_key])

            for child in ep["children_object_categories"]:
                goal_key = "{}_{}".format(scene_id.split("/")[-1], child)
                if goal_key not in data["goals_by_category"]:
                    continue
                goals.extend(data["goals_by_category"][goal_key])

            vps = [
                vp["agent_state"]["position"] for g in goals for vp in g["view_points"]
            ]

            start_position = np.array(ep["start_position"])

            # point should be not be isolated to a small poly island
            ISLAND_RADIUS_LIMIT = 1.5
            if sim.pathfinder.island_radius(start_position) < ISLAND_RADIUS_LIMIT:
                raise RuntimeError

            closest_goals = []
            geo_dist, closest_point = geodesic_distance(sim, start_position, vps)
            closest_goals.append((geo_dist, closest_point))

            geo_dists, goals_sorted = zip(
                *sorted(zip(closest_goals, goals), key=lambda x: x[0][0])
            )

            geo_dist, closest_pt = geo_dists[0]

            if not np.isfinite(geo_dist):
                raise RuntimeError

            if geo_dist < 1.0 or geo_dist > 30.0:
                raise RuntimeError

            # Check that the shortest path points are all on the same floor
            path = habitat_sim.ShortestPath()
            path.requested_start = start_position
            path.requested_end = closest_pt
            found_path = sim.pathfinder.find_path(path)

            if not found_path:
                raise RuntimeError

            heights = [p.tolist()[1] for p in path.points]
            h_delta = max(heights) - min(heights)

            if h_delta > 0.25:
                raise RuntimeError

            ep["info"]["geodesic_distance"] = geo_dist
            ep["info"]["euclidean_distance"] = np.linalg.norm(
                start_position - closest_pt
            )
            valid_episodes.append(ep)
        except Exception:
            # print("Error in episode: {} - {} - {}".format(ep["object_category"], ep["children_object_categories"], e))
            pass

    print(
        "Episodes pre and post filtering: {} / {}".format(
            len(data["episodes"]), len(valid_episodes)
        )
    )
    data["episodes"] = valid_episodes

    print("Writing cleaned episodes at: {}".format(output_path))
    write_dataset(data, output_path)


def _config_sim(scene: str):
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene

    sensor_specs = []
    for name, sensor_type in zip(
        ["color"],
        [
            habitat_sim.SensorType.COLOR,
        ],
    ):
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = f"{name}_sensor"
        sensor_spec.sensor_type = sensor_type
        sensor_spec.resolution = [1, 1]
        sensor_spec.position = [0.0, 1.0, 0.0]
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


def geodesic_distance(sim, position_a, position_b):
    path = habitat_sim.MultiGoalShortestPath()
    if isinstance(position_b[0], (Sequence, np.ndarray)):
        path.requested_ends = np.array(position_b, dtype=np.float32)
    else:
        path.requested_ends = np.array([np.array(position_b, dtype=np.float32)])
    path.requested_start = np.array(position_a, dtype=np.float32)
    sim.pathfinder.find_path(path)
    end_pt = path.points[-1] if len(path.points) else np.array([])
    return path.geodesic_distance, end_pt


def validate_dataset(path, output_path):
    files = glob.glob(os.path.join(path, "*.json.gz"))
    for file in files:
        print("Validating {}".format(file))
        validate_episodes(file, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()

    try:
        if args.path.endswith(".json.gz"):
            print("Validate episodeL: {}".format(args.path))
            validate_episodes(args.path, args.output_path)
        else:
            validate_dataset(args.path, args.output_path)
    except Exception as e:
        print(e)
