import argparse
import gzip
import itertools
import multiprocessing
import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple, Union

import GPUtil
import habitat
import habitat_sim
import numpy as np
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode
from habitat_sim import bindings as hsim
from habitat_sim._ext.habitat_sim_bindings import SemanticObject
from habitat_sim.agent.agent import AgentConfiguration, AgentState
from habitat_sim.simulator import Simulator
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs
from numpy import ndarray

# from ovon.dataset.visualization import plot_area  # noqa:F401
# from ovon.dataset.visualization import save_candidate_imgs
from tqdm import tqdm

from ovon.dataset.pose_sampler import PoseSampler
from ovon.dataset.semantic_utils import (
    ObjectCategoryMapping,
    SceneRelationshipsMapping,
    get_hm3d_semantic_scenes,
)


class InstanceObjectGoalGenerator:
    ISLAND_RADIUS_LIMIT: float = 1.5

    semantic_spec_filepath: str
    img_size: Tuple[int, int]
    agent_height: float
    hfov: float
    agent_radius: float
    pose_sampler_args: Dict[str, Any]
    frame_cov_thresh: Tuple[float, float]
    goal_vp_cell_size: float
    goal_vp_max_dist: float
    start_poses_per_obj: float
    start_poses_tilt_angle: float
    start_distance_limits: Tuple[float, float]
    min_geo_to_euc_ratio: float
    start_retries: int
    cat_map: ObjectCategoryMapping
    relationships_mapping: SceneRelationshipsMapping

    def __init__(
        self,
        semantic_spec_filepath: str,
        img_size: Tuple[int, int],
        hfov: float,
        agent_height: float,
        agent_radius: float,
        sensor_height: float,
        pose_sampler_args: Dict[str, Any],
        mapping_file: str,
        categories: str,
        coverage_meta_file: str,
        relationship_meta_file: str,
        frame_cov_thresh: Tuple[float, float],
        goal_vp_cell_size: float = 0.1,
        goal_vp_max_dist: float = 1.0,
        start_poses_per_obj: int = 500,
        start_poses_tilt_angle: float = 30.0,
        start_distance_limits: Tuple[float, float] = (1.0, 30.0),
        min_geo_to_euc_ratio: float = 1.05,
        start_retries: int = 1000,
        sample_dense_viewpoints: bool = False,
        device_id: int = 0,
        total_tasks: int = 0,
        max_viewpoint_radius: float = 1.0,
    ) -> None:
        self.semantic_spec_filepath = semantic_spec_filepath
        self.img_size = img_size
        self.hfov = hfov
        self.agent_height = agent_height
        self.agent_radius = agent_radius
        self.sensor_height = sensor_height
        self.pose_sampler_args = pose_sampler_args
        self.frame_cov_thresh = frame_cov_thresh
        self.goal_vp_cell_size = goal_vp_cell_size
        self.goal_vp_max_dist = goal_vp_max_dist
        self.start_poses_per_obj = start_poses_per_obj
        self.start_poses_tilt_angle = start_poses_tilt_angle
        self.start_distance_limits = start_distance_limits
        self.min_geo_to_euc_ratio = min_geo_to_euc_ratio
        self.start_retries = start_retries
        self.sample_dense_viewpoints = sample_dense_viewpoints
        self.device_id = device_id
        self.cat_map = ObjectCategoryMapping(
            mapping_file=mapping_file,
            allowed_categories=categories,
            coverage_meta_file=coverage_meta_file,
            frame_coverage_threshold=frame_cov_thresh,
        )
        self.relationships_mapping = SceneRelationshipsMapping(relationship_meta_file)
        self.max_viewpoint_radius = max_viewpoint_radius

    def _config_sim(self, scene: str) -> Simulator:
        sim_cfg = hsim.SimulatorConfiguration()
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = self.device_id
        sim_cfg.scene_dataset_config_file = self.semantic_spec_filepath
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
            sensor_spec.resolution = [self.img_size[0], self.img_size[1]]
            sensor_spec.position = [0.0, self.sensor_height, 0.0]
            sensor_spec.hfov = self.hfov
            sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(sensor_spec)

        # create agent specifications
        agent_cfg = AgentConfiguration(
            height=self.agent_height,
            radius=self.agent_radius,
            sensor_specifications=sensor_specs,
            action_space={
                "look_up": habitat_sim.ActionSpec(
                    "look_up",
                    habitat_sim.ActuationSpec(amount=self.start_poses_tilt_angle),
                ),
                "look_down": habitat_sim.ActionSpec(
                    "look_down",
                    habitat_sim.ActuationSpec(amount=self.start_poses_tilt_angle),
                ),
            },
        )

        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

        # set the navmesh
        assert sim.pathfinder.is_loaded, "pathfinder is not loaded!"
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_height = self.agent_height
        navmesh_settings.agent_radius = self.agent_radius
        navmesh_success = sim.recompute_navmesh(
            sim.pathfinder, navmesh_settings, include_static_objects=False
        )
        assert navmesh_success, "Failed to build the navmesh!"
        return sim

    def _threshold_object_goals(
        self,
        frame_covs: List[float],
    ) -> List[bool]:
        keep_goal = []
        for fc in frame_covs:
            keep_goal.append(fc > self.frame_cov_thresh)
        return keep_goal

    def _make_object_viewpoints(self, sim: Simulator, obj: SemanticObject):
        object_position = obj.aabb.center
        eps = 1e-5
        x_len, _, z_len = obj.aabb.sizes / 2.0 + self.goal_vp_max_dist
        x_bxp = (
            np.arange(-x_len, x_len + eps, step=self.goal_vp_cell_size)
            + object_position[0]
        )
        z_bxp = (
            np.arange(-z_len, z_len + eps, step=self.goal_vp_cell_size)
            + object_position[2]
        )
        candiatate_poses = [
            np.array([x, object_position[1], z])
            for x, z in itertools.product(x_bxp, z_bxp)
        ]

        def _down_is_navigable(pt):
            pf = sim.pathfinder

            delta_y = 0.05
            max_steps = int(2 / delta_y)
            step = 0
            is_navigable = pf.is_navigable(pt, 2)
            while not is_navigable:
                pt[1] -= delta_y
                is_navigable = pf.is_navigable(pt)
                step += 1
                if step == max_steps:
                    return False
            return True

        def _face_object(object_position: np.array, point: ndarray):
            EPS_ARRAY = np.array([1e-8, 0.0, 1e-8])
            cam_normal = (object_position - point) + EPS_ARRAY
            cam_normal[1] = 0
            cam_normal = cam_normal / np.linalg.norm(cam_normal)
            return quat_from_two_vectors(habitat_sim.geo.FRONT, cam_normal)

        def _get_iou(pt):
            obb = habitat_sim.geo.OBB(obj.aabb)
            if obb.distance(pt) > self.goal_vp_max_dist:
                return -0.5, pt, None

            if not _down_is_navigable(pt):
                return -1.0, pt, None

            pt = np.array(sim.pathfinder.snap_point(pt))
            q = _face_object(object_position, pt)

            cov = 0
            sim.agents[0].set_state(AgentState(position=pt, rotation=q))
            for act in ["look_down", "look_up", "look_up"]:
                obs = sim.step(act)
                cov += self._compute_frame_coverage(obs, obj.semantic_id)

            return cov, pt, q

        candiatate_poses_ious = [_get_iou(pos) for pos in candiatate_poses]
        best_iou = (
            max(v[0] for v in candiatate_poses_ious)
            if len(candiatate_poses_ious) != 0
            else 0
        )
        if best_iou <= 0.0:
            return []

        view_locations = [
            {
                "agent_state": {
                    "position": pt.tolist(),
                    "rotation": quat_to_coeffs(q).tolist(),
                },
                "iou": iou,
            }
            for iou, pt, q in candiatate_poses_ious
            if iou > self.frame_cov_thresh
        ]
        view_locations = sorted(view_locations, reverse=True, key=lambda v: v["iou"])

        # # for debugging: shows top-down map of viewpoints.
        # plot_area(
        #     candiatate_poses_ious,
        #     [v["agent_state"]["position"] for v in view_locations],
        #     [object_position],
        #     obj.id,
        # )

        return view_locations

    def _sample_start_poses(
        self,
        sim: Simulator,
        goals: List,
    ) -> Tuple[List, List]:
        # viewpoint_locs = [vp["agent_state"]["position"] for vp in viewpoints]
        viewpoint_locs = [
            [vp["agent_state"]["position"] for vp in goal["view_points"]]
            for goal in goals
        ]
        start_positions = []
        start_rotations = []

        while len(start_positions) < self.start_poses_per_obj:
            for _ in range(self.start_retries):
                start_position = sim.pathfinder.get_random_navigable_point().astype(
                    np.float32
                )
                if (
                    start_position is None
                    or np.any(np.isnan(start_position))
                    or not sim.pathfinder.is_navigable(start_position)
                ):
                    raise RuntimeError("Unable to find valid starting location")

                # point should be not be isolated to a small poly island
                if (
                    sim.pathfinder.island_radius(start_position)
                    < self.ISLAND_RADIUS_LIMIT
                ):
                    continue

                closest_goals = []
                for vps in viewpoint_locs:
                    geo_dist, closest_point = self._geodesic_distance(
                        sim, start_position, vps
                    )
                    closest_goals.append((geo_dist, closest_point))

                geo_dists, goals_sorted = zip(
                    *sorted(zip(closest_goals, goals), key=lambda x: x[0][0])
                )

                geo_dist, closest_pt = geo_dists[0]

                if not np.isfinite(geo_dist):
                    continue

                if (
                    geo_dist < self.start_distance_limits[0]
                    or geo_dist > self.start_distance_limits[1]
                ):
                    continue

                dist_ratio = geo_dist / np.linalg.norm(start_position - closest_pt)
                if dist_ratio < self.min_geo_to_euc_ratio:
                    continue

                # aggressive _ratio_sample_rate (copied from PointNav)
                if np.random.rand() > (20 * (dist_ratio - 0.98) ** 2):
                    continue

                # Check if atleast one goal is on the same floor as the agent
                start_position_on_same_floor = False
                for view_point in viewpoint_locs:
                    gt = np.array(view_point)
                    start_position_on_same_floor = np.any(
                        np.abs(gt[:, 1] - start_position[1]) < 0.25
                    )
                    if start_position_on_same_floor:
                        break
                if not start_position_on_same_floor:
                    continue

                angle = np.random.uniform(0, 2 * np.pi)
                source_rotation = [
                    0,
                    np.sin(angle / 2),
                    0,
                    np.cos(angle / 2),
                ]  # Pick random starting rotation

                start_positions.append(start_position)
                start_rotations.append(source_rotation)
                break

            else:
                # no start pose found after n attempts
                return [], []

        return start_positions, start_rotations

    def _states_to_viewpoints(self, states):
        viewpoints = []

        for state, radius in states:
            if radius > self.max_viewpoint_radius:
                continue

            viewpoints.append(
                {
                    "agent_state": {
                        "position": state.position.tolist(),
                        "rotation": quat_to_coeffs(state.rotation).tolist(),
                    },
                    "iou": 0.0,
                }
            )
        return viewpoints

    def _make_goal(
        self,
        sim: Simulator,
        pose_sampler: PoseSampler,
        obj: SemanticObject,
        with_viewpoints: bool,
        with_start_poses: bool,
        prompt_goal: str = None,
    ):
        if with_start_poses:
            assert with_viewpoints

        states = pose_sampler.sample_agent_poses_radially(search_center=obj.aabb.center)
        observations = self._render_poses(sim, states)
        observations, states = self._can_see_object(observations, states, obj)
        if len(observations) == 0:
            return None

        frame_coverages = self._compute_frame_coverage(observations, obj.semantic_id)

        keep_goal = self._threshold_object_goals(frame_coverages)

        if sum(keep_goal) == 0:
            return None

        result = {
            "object_category": (
                self.cat_map[obj.category.name()]
                if prompt_goal is None
                else prompt_goal
            ),
            "object_id": obj.id,
            "position": obj.aabb.center.tolist(),
        }

        if not with_viewpoints:
            return result

        if self.sample_dense_viewpoints:
            goal_viewpoints = self._make_object_viewpoints(sim, obj)
            if len(goal_viewpoints) == 0:
                return None
            result["view_points"] = goal_viewpoints
        else:
            goal_viewpoints = self._states_to_viewpoints(states)
            result["view_points"] = goal_viewpoints

        if not with_start_poses:
            return result
        return result

    def make_object_goals(
        self,
        scene: str,
        with_viewpoints: bool,
        with_start_poses: bool,
    ) -> List[Dict[str, Any]]:
        sim = self._config_sim(scene)
        pose_sampler = PoseSampler(sim=sim, **self.pose_sampler_args)

        objects = [
            o
            for o in sim.semantic_scene.objects
            if self.cat_map[o.category.name()] is not None
        ]
        id_to_object = {obj.semantic_id: obj for obj in objects}

        scene_key = os.path.basename(scene).split(".")[0]
        print(scene_key)
        scene_relationships = self.relationships_mapping[scene_key]

        object_goals = defaultdict(list)
        results = []

        for reln_name, reln_instances_list in tqdm(
            scene_relationships.items(),
            total=len(scene_relationships.items()),
            dynamic_ncols=True,
        ):
            for instance in reln_instances_list:
                obj_id = instance["target_obj_semantic_id"]
                obj = id_to_object[obj_id]
                goal = self._make_goal(
                    sim,
                    pose_sampler,
                    obj,
                    with_viewpoints,
                    with_start_poses,
                    prompt_goal=reln_name,
                )
                if goal is not None and len(goal["view_points"]) > 0:
                    object_goals[reln_name].append(goal)
                    results.append((obj_id, reln_name, len(goal["view_points"])))

        all_goals = []
        for relationship_name, goals in object_goals.items():
            start_positions, start_rotations = self._sample_start_poses(sim, goals)
            if len(start_positions) == 0:
                print("Start poses none for: {}".format(relationship_name))
                continue

            all_goals.append(
                {
                    "object_goals": goals,
                    "start_positions": start_positions,
                    "start_rotations": start_rotations,
                }
            )
        sim.close()
        return all_goals

    @staticmethod
    def _render_poses(
        sim: Simulator, agent_states: List[AgentState]
    ) -> List[Dict[str, ndarray]]:
        obs = []
        for agent_state, _ in agent_states:
            sim.agents[0].set_state(agent_state, infer_sensor_states=False)
            obs.append(sim.get_sensor_observations())
        return obs

    @staticmethod
    def _can_see_object(
        observations: List[Dict[str, ndarray]],
        states: List[AgentState],
        obj: SemanticObject,
    ):
        """Keep observations and sim states that can see the object."""
        keep_o = []
        keep_s = []
        for o, s in zip(observations, states):
            if np.isin(obj.semantic_id, o["semantic_sensor"]):
                keep_o.append(o)
                keep_s.append(s)

        return keep_o, keep_s

    @staticmethod
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

    @staticmethod
    def _geodesic_distance(
        sim: Simulator,
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray],
    ) -> float:
        path = habitat_sim.MultiGoalShortestPath()
        if isinstance(position_b[0], (Sequence, np.ndarray)):
            path.requested_ends = np.array(position_b, dtype=np.float32)
        else:
            path.requested_ends = np.array([np.array(position_b, dtype=np.float32)])
        path.requested_start = np.array(position_a, dtype=np.float32)
        sim.pathfinder.find_path(path)
        end_pt = path.points[-1] if len(path.points) else np.array([])
        return path.geodesic_distance, end_pt

    @staticmethod
    def save_to_disk(episode_dataset, save_to: str):
        """
        TODO: pick a format for distribution and use. For now pickle it.
        TODO: Which observation modalities to save? Probably just RGB.
        """
        with gzip.open(save_to, "wt") as f:
            f.write(episode_dataset.to_json())

    @staticmethod
    def _create_episode(
        episode_id,
        scene_id,
        start_position,
        start_rotation,
        object_category,
        shortest_paths=None,
        info=None,
        scene_dataset_config="default",
    ):
        return ObjectGoalNavEpisode(
            episode_id=str(episode_id),
            goals=[],
            scene_id=scene_id,
            object_category=object_category,
            start_position=start_position,
            start_rotation=start_rotation,
            shortest_paths=shortest_paths,
            info=info,
            scene_dataset_config=scene_dataset_config,
        )

    def make_episodes(self, object_goals, scene):
        dataset = habitat.datasets.make_dataset("ObjectNav-v1")
        dataset.category_to_task_category_id = {}
        dataset.category_to_scene_annotation_category_id = {}

        goals_by_category = defaultdict(list)
        episode_count = 0
        print("Total number of object goals: {}".format(len(object_goals)))
        for goal in object_goals:
            object_goal = goal["object_goals"][0]
            scene_id = scene.split("/")[-1]
            goals_category_id = "{}_{}".format(scene_id, object_goal["object_category"])
            print(
                "Goal category: {} - viewpoints: {}".format(
                    goals_category_id, len(object_goal["view_points"])
                )
            )

            if len(goal["start_positions"]) == 0:
                print(
                    "Object: {} is ignored because there are no valid start positions"
                    .format(object_goal["img_ref"])
                )
                continue
            start_positions = goal["start_positions"]
            start_rotations = goal["start_rotations"]

            goals_by_category[goals_category_id].extend(goal["object_goals"])

            for start_position, start_rotation in zip(start_positions, start_rotations):
                episode = self._create_episode(
                    episode_id=episode_count,
                    scene_id=scene.replace("data/scene_datasets/", ""),
                    scene_dataset_config="./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
                    start_position=start_position,
                    start_rotation=start_rotation,
                    info={
                        "geodesic_distance": 0,
                        "euclidean_distance": 0,
                    },
                    object_category=object_goal["object_category"],
                )
                dataset.episodes.append(episode)
                episode_count += 1
        dataset.goals_by_category = goals_by_category
        return dataset


def make_episodes_for_scene(args):
    scene, outpath, device_id = args
    if isinstance(scene, tuple) and outpath is None:
        scene, outpath = scene

    iig_maker = InstanceObjectGoalGenerator(
        semantic_spec_filepath=(
            "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        ),
        img_size=(512, 512),
        hfov=90,
        agent_height=1.41,
        agent_radius=0.17,
        sensor_height=1.31,
        pose_sampler_args={
            "r_min": 0.5,
            "r_max": 2.0,
            "r_step": 0.5,
            "rot_deg_delta": 10.0,
            "h_min": 0.8,
            "h_max": 1.4,
            "sample_lookat_deg_delta": 5.0,
        },
        mapping_file="ovon/dataset/source_data/Mp3d_category_mapping.tsv",
        categories=None,
        coverage_meta_file="data/coverage_meta/train.pkl",
        relationship_meta_file=(
            "data/relationships/relationships2d_train_1_filtered.pickle"
        ),
        frame_cov_thresh=0.05,
        goal_vp_cell_size=0.1,
        goal_vp_max_dist=1.0,
        start_poses_per_obj=500,
        start_poses_tilt_angle=30.0,
        start_distance_limits=(1.0, 30.0),
        min_geo_to_euc_ratio=1.05,
        start_retries=2000,
        device_id=device_id,
    )

    object_goals = iig_maker.make_object_goals(
        scene=scene, with_viewpoints=True, with_start_poses=True
    )
    print("Scene: {}".format(scene))
    episode_dataset = iig_maker.make_episodes(object_goals, scene)

    scene_name = os.path.basename(scene).split(".")[0]
    save_to = os.path.join(outpath, f"{scene_name}.json.gz")
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    print("Total episodes: {}".format(len(episode_dataset.episodes)))
    iig_maker.save_to_disk(episode_dataset, save_to)


def make_episodes_for_split(
    split: str,
    outpath: str,
    num_scenes: str,
    tasks_per_gpu: int = 1,
    multiprocessing_enabled: bool = False,
):
    """Makes episodes for all scenes in a split"""
    scenes = sorted(
        list(get_hm3d_semantic_scenes("data/scene_datasets/hm3d", [split])[split]),
        reverse=True,
    )[:num_scenes]
    print(scenes)

    dataset = habitat.datasets.make_dataset("ObjectNav-v1")
    dataset.category_to_task_category_id = {}
    dataset.category_to_scene_annotation_category_id = {}

    os.makedirs(outpath.format(split), exist_ok=True)
    save_to = os.path.join(
        outpath.format(split).replace("content/", ""), f"{split}.json.gz"
    )
    InstanceObjectGoalGenerator.save_to_disk(dataset, save_to)

    if multiprocessing_enabled:
        gpus = len(GPUtil.getAvailable(limit=256))
        cpu_threads = gpus * 16
        deviceIds = GPUtil.getAvailable(
            order="memory", limit=1, maxLoad=1.0, maxMemory=1.0
        )
        print("In multiprocessing setup - cpu {}, GPU: {}".format(cpu_threads, gpus))

        items = []
        for i, s in enumerate(scenes):
            deviceId = deviceIds[0]
            if i < gpus * tasks_per_gpu or len(deviceIds) == 0:
                deviceId = i % gpus
            items.append((s, outpath.format(split), deviceId))

        mp_ctx = multiprocessing.get_context("forkserver")
        with (
            mp_ctx.Pool(cpu_threads) as pool,
            tqdm(total=len(scenes), position=0) as pbar,
        ):
            for _ in pool.imap_unordered(make_episodes_for_scene, items):
                pbar.update()
    else:
        for scene in tqdm(scenes, total=len(scenes), dynamic_ncols=True):
            make_episodes_for_scene(scene, outpath.format(split))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/datasets/ovon/hm3d/v1",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--tasks-per-gpu",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--multiprocessing-enabled",
        dest="multiprocessing_enabled",
        action="store_true",
    )

    args = parser.parse_args()
    outpath = os.path.join(args.output_path, "{}/content/".format(args.split))
    make_episodes_for_split(
        args.split,
        outpath,
        args.num_scenes,
        args.tasks_per_gpu,
        args.multiprocessing_enabled,
    )
