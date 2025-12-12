import argparse
import os

import habitat
import habitat_sim
import numpy as np
from habitat import get_config, logger
from habitat.config import read_write
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
)
from habitat_sim.utils.common import quat_from_two_vectors
from numpy import ndarray
from tqdm import tqdm

from ovon.config import ClipObjectGoalSensorConfig, OVONDistanceToGoalConfig
from ovon.utils.visualize.viz import append_text_to_image


def _face_object(object_position: np.array, point: ndarray):
    EPS_ARRAY = np.array([1e-8, 0.0, 1e-8])
    cam_normal = (object_position - point) + EPS_ARRAY
    cam_normal[1] = 0
    cam_normal = cam_normal / np.linalg.norm(cam_normal)
    return quat_from_two_vectors(habitat_sim.geo.FRONT, cam_normal)


def make_videos(observations_list, output_dir, id):
    images_to_video(observations_list[0], output_dir=output_dir, video_name=id)


def get_nearest_goal(episode, env):
    min_dist = 1000.0
    sim = env.sim
    goal_key = "{}_{}".format(episode.scene_id.split("/")[-1], episode.object_category)
    goals = env._dataset.goals_by_category[goal_key]

    goal_location = None
    goal_rotation = None

    agent_position = sim.get_agent_state().position
    for goal in goals:
        for view_point in goal.view_points:
            position = view_point.agent_state.position

            dist = sim.geodesic_distance(agent_position, position)
            if min_dist > dist:
                min_dist = dist
                goal_location = position
                goal_rotation = view_point.agent_state.rotation
    return goal_location, goal_rotation


def generate_trajectories(cfg, video_dir="", num_episodes=1):
    os.makedirs(video_dir, exist_ok=True)
    with habitat.Env(cfg) as env:
        goal_radius = 0.1
        spl = 0
        total_success = 0.0
        total_episodes = 0.0
        scene_id = env._current_episode.scene_id.split("/")[-1].split(".")[0]

        logger.info("Total episodes: {}".format(len(env.episodes)))
        num_episodes = min(len(env.episodes), num_episodes)
        for episode_id in tqdm(range(num_episodes)):
            follower = ShortestPathFollower(env._sim, goal_radius, False)
            env.reset()
            success = 0
            episode = env.current_episode
            goal_position, goal_rotation = get_nearest_goal(episode, env)

            info = {}
            obs_list = []
            if goal_position is None:
                continue

            while not env.episode_over:
                best_action = follower.get_next_action(goal_position)

                if (
                    "distance_to_goal" in info.keys()
                    and info["distance_to_goal"] < 0.1
                    and best_action != HabitatSimActions.stop
                ):
                    best_action = HabitatSimActions.stop

                observations = env.step(best_action)

                if best_action == HabitatSimActions.stop:
                    position = env.sim.get_agent_state().position
                    observations = env.sim.get_observations_at(
                        position, goal_rotation, False
                    )

                info = env.get_metrics()
                frame = observations_to_image({"rgb": observations["rgb"]}, info)
                frame = append_text_to_image(
                    frame, "Go to {}".format(episode.object_category)
                )
                obs_list.append(frame)

                success = info["success"]

            print(info)
            total_success += success
            spl += info["spl"]
            total_episodes += 1

            make_videos([obs_list], video_dir, "{}_{}".format(scene_id, episode_id))
        print("Total episodes: {}".format(total_episodes))

        print("\n\nEpisode success: {}".format(total_success / total_episodes))
        print("SPL: {}, {}, {}".format(spl / total_episodes, spl, total_episodes))
        print(
            "Success: {}, {}, {}".format(
                total_success / total_episodes, total_success, total_episodes
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/episodes/sampled.json.gz")
    parser.add_argument("--video-dir", type=str, default="data/video_dir/")
    parser.add_argument("--num-episodes", type=int, default=2)
    args = parser.parse_args()

    objectnav_config = "config/tasks/objectnav_stretch_hm3d.yaml"
    config = get_config(objectnav_config)
    with read_write(config):
        config.habitat.dataset.type = "OVON-v1"
        config.habitat.dataset.split = "train"
        config.habitat.dataset.scenes_dir = "data/scene_datasets/"
        config.habitat.dataset.content_scenes = ["*"]
        config.habitat.dataset.data_path = args.data
        del config.habitat.task.lab_sensors["objectgoal_sensor"]
        config.habitat.task.lab_sensors["clip_objectgoal_sensor"] = (
            ClipObjectGoalSensorConfig()
        )
        config.habitat.task.measurements.distance_to_goal = OVONDistanceToGoalConfig()
        config.habitat.task.measurements.success.success_distance = 0.25

    generate_trajectories(
        config, video_dir=args.video_dir, num_episodes=args.num_episodes
    )


if __name__ == "__main__":
    main()
