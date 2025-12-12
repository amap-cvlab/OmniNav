import argparse
import os

import cv2
import habitat
import habitat_sim
from habitat.config import read_write
from habitat.config.default import get_config
from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig
from habitat.utils.visualizations import maps

from ovon.utils.utils import (
    draw_bounding_box,
    draw_point,
    load_dataset,
)

SCENES_ROOT = "data/scene_datasets/hm3d"
MAP_RESOLUTION = 512


def get_objnav_config(scene):
    TASK_CFG = "config/tasks/objectnav_stretch_hm3d.yaml"
    SCENE_DATASET_CFG = os.path.join(
        SCENES_ROOT, "hm3d_annotated_basis.scene_dataset_config.json"
    )
    objnav_config = get_config(TASK_CFG)

    deviceId = 0

    with read_write(objnav_config):
        # TODO: find a better way to do it.
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor = (
            HabitatSimSemanticSensorConfig()
        )
        objnav_config.habitat.simulator.habitat_sim_v0.gpu_device_id = deviceId
        objnav_config.habitat.simulator.scene = scene
        objnav_config.habitat.simulator.scene_dataset = SCENE_DATASET_CFG
        objnav_config.habitat.simulator.habitat_sim_v0.enable_physics = True

        objnav_config.habitat.task.measurements = {}

    return objnav_config


def get_sim(objnav_config):
    sim = habitat.sims.make_sim("Sim-v0", config=objnav_config.habitat.simulator)

    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = (
        objnav_config.habitat.simulator.agents.main_agent.radius
    )
    navmesh_settings.agent_height = (
        objnav_config.habitat.simulator.agents.main_agent.height
    )
    navmesh_settings.agent_radius = 0.18
    navmesh_settings.agent_height = 0.88
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings, include_static_objects=True)

    return sim


def setup(scene):
    ovon_config = get_objnav_config(scene)
    sim = get_sim(ovon_config)
    return sim


def visualize_episodes(sim, episodes, goals, object_category):
    top_down_maps = []

    grouped_goal_heights = []
    grouped_goals = []

    for i in range(len(goals)):
        goal = goals[i]
        group_exists = False
        for idx, heights in enumerate(grouped_goal_heights):
            if abs(goal["position"][1] - heights[0]) <= 0.25:
                group_exists = True
                grouped_goal_heights[idx].append(goal["position"][1])
                grouped_goals[idx].append(goal)
                break
        if not group_exists:
            grouped_goal_heights.append([goal["position"][1]])
            grouped_goals.append([goal])

    print("Category: {}".format(object_category))
    print(
        "Grouped goals: {}, goals: {}".format(
            [len(g) for g in grouped_goals], len(goals)
        )
    )

    for grouped_goal in grouped_goals:
        ref_floor_height = grouped_goal[0]["position"][1]
        top_down_map = None
        goal_height = grouped_goal[0]["view_points"][0]["agent_state"]["position"][1]

        for goal in grouped_goal:
            if top_down_map is None:
                top_down_map = maps.get_topdown_map(
                    sim.pathfinder,
                    height=goal_height,
                    map_resolution=MAP_RESOLUTION,
                    draw_border=True,
                )

            top_down_map = draw_point(
                sim,
                top_down_map,
                goal["position"],
                maps.MAP_TARGET_POINT_INDICATOR,
                point_padding=6,
            )
            for view_point in goal["view_points"]:
                top_down_map = draw_point(
                    sim,
                    top_down_map,
                    view_point["agent_state"]["position"],
                    maps.MAP_VIEW_POINT_INDICATOR,
                )

            draw_bounding_box(sim, top_down_map, goal["object_id"], ref_floor_height)

        for episode in episodes:
            if episode["object_category"] != object_category:
                continue

            on_same_floor = False
            for goal in grouped_goal:
                if abs(episode["start_position"][1] - goal_height) <= 0.25:
                    on_same_floor = True
                    break

            if not on_same_floor:
                continue

            if top_down_map is None:
                top_down_map = maps.get_topdown_map(
                    sim.pathfinder,
                    height=episode["start_position"][1],
                    map_resolution=MAP_RESOLUTION,
                    draw_border=True,
                )

            draw_point(
                sim,
                top_down_map,
                episode["start_position"],
                maps.MAP_SOURCE_POINT_INDICATOR,
            )

        if top_down_map is None:
            continue

        top_down_map = maps.colorize_topdown_map(top_down_map)
        top_down_maps.append(top_down_map)

    print(
        "Grouped goals: {}, top down maps: {}".format(
            len(grouped_goals), len(top_down_maps)
        )
    )

    return top_down_maps


def save_visual(img, path):
    cv2.imwrite(path, img)


def visualize(episodes_path, output_path):
    dataset = load_dataset(episodes_path)

    os.makedirs(output_path, exist_ok=True)

    sim = setup(dataset["episodes"][0]["scene_id"])
    categories = dataset["goals_by_category"].keys()
    for category in categories:
        top_down_maps = visualize_episodes(
            sim,
            dataset["episodes"],
            dataset["goals_by_category"][category],
            object_category=category.split("_")[1],
        )
        for i, top_down_map in enumerate(top_down_maps):
            object_output_path = os.path.join(
                output_path, "{}_{}.png".format(category, i)
            )
            print(object_output_path, category)
            save_visual(top_down_map, object_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=str,
        required=True,
        help="Path to episode dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="output path of visualization",
    )
    args = parser.parse_args()
    visualize(args.episodes, args.output_path)
