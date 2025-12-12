import argparse
import copy
import gzip
import itertools
import math
import multiprocessing
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple, Union


import habitat
import habitat_sim
import matplotlib.pyplot as plt
import numpy as np
from habitat_sim import bindings as hsim
from habitat_sim._ext.habitat_sim_bindings import SemanticObject
from habitat_sim.agent.agent import AgentConfiguration, AgentState
from habitat_sim.simulator import Simulator
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs
from numpy import ndarray
from sklearn.cluster import AgglomerativeClustering

# from ovon.dataset.visualization import plot_area  # noqa:F401
# from ovon.dataset.visualization import save_candidate_imgs
from tqdm import tqdm

from ovon.dataset.ovon_dataset import OVONEpisode
from ovon.dataset.pose_sampler import PoseSampler
from ovon.dataset.semantic_utils import (
    ObjectCategoryMapping,
    WordnetMapping,
    get_hm3d_semantic_scenes,
)
from ovon.utils.utils import load_json


try:
    import pandas as pd
    import seaborn as sns
    import trimesh
except ImportError:
    pass

class ObjectGoalGenerator:
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
    max_viewpoint_radius: float
    single_floor_threshold: float

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
        categories: List[str],
        coverage_meta_file: str,
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
        max_viewpoint_radius: float = 1.0,
        wordnet_mapping_file: str = None,
        single_floor_threshold: float = 0.25,
        verbose: bool = False,
        plot_folder: str = "data/visualizations/cluster_infos/",
        sample_start_poses_wrt_navmesh_clusters: bool = False,
        disable_euc_to_geo_ratio_check: bool = False,
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
        self.max_viewpoint_radius = max_viewpoint_radius
        self.single_floor_threshold = single_floor_threshold
        self.verbose = verbose
        self.plot_folder = plot_folder
        self.sample_start_poses_wrt_navmesh_clusters = (
            sample_start_poses_wrt_navmesh_clusters
        )
        self.disable_euc_to_geo_ratio_check = disable_euc_to_geo_ratio_check
        self.wordnet_map = WordnetMapping(mapping_file=wordnet_mapping_file)
        self.cat_map = ObjectCategoryMapping(
            mapping_file=mapping_file,
            allowed_categories=categories,
            coverage_meta_file=coverage_meta_file,
            frame_coverage_threshold=frame_cov_thresh,
        )
        print("All wordnet categories: {}", self.wordnet_map._mapping)

    def _config_sim(self, scene: str) -> Simulator:
        sim_cfg = hsim.SimulatorConfiguration()
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = self.device_id
        sim_cfg.scene_dataset_config_file = self.semantic_spec_filepath
        sim_cfg.scene_id = scene
        sim_cfg.allow_sliding = False

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
        navmesh_settings.agent_max_climb = 0.10
        navmesh_settings.cell_height = 0.05
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

    def _sample_start_poses_wrt_clusters(
        self,
        sim: Simulator,
        goals: List,
        cluster_centers: List[List[float]],
        distance_to_clusters: np.float32,
    ) -> Tuple[List, List]:
        # viewpoint_locs = [vp["agent_state"]["position"] for vp in viewpoints]
        viewpoint_locs = [
            [vp["agent_state"]["position"] for vp in goal["view_points"]]
            for goal in goals
        ]
        start_positions = []
        start_rotations = []

        # Filter out invalid clusters
        valid_mask = (distance_to_clusters >= self.start_distance_limits[0]) & (
            distance_to_clusters <= self.start_distance_limits[1]
        )
        target_positions = np.array(list(itertools.chain(*viewpoint_locs)))
        # Ensure that cluster is on same floor as atleast 1 object viewpoint
        for i, cluster_info in enumerate(cluster_centers):
            valid_mask[i] = valid_mask[i] & np.any(
                np.abs(cluster_info["center"][1] - target_positions[:, 1]) < 0.30
            )

        valid_clusters = []
        for i in range(len(cluster_centers)):
            if valid_mask[i].item():
                valid_clusters.append(cluster_centers[i])

        if len(valid_clusters) == 0:
            raise RuntimeError(
                f"No valid clusters: {len(valid_clusters)}/{len(cluster_centers)}"
            )

        # Divide episodes across clusters
        cluster_centers = valid_clusters
        NC = len(cluster_centers)
        episodes_per_cluster = np.zeros((len(cluster_centers),), dtype=np.int32)

        if NC <= self.start_poses_per_obj:
            # Case 1: There are more episodes than clusters
            ## Divide episodes equally across clusters
            episodes_per_cluster[:] = self.start_poses_per_obj // NC
            ## Place the residual episodes into random clusters
            residual_episodes = self.start_poses_per_obj % NC
            if residual_episodes > 0:
                random_order = np.random.permutation(NC)
                for i in random_order[:residual_episodes]:
                    episodes_per_cluster[i] += 1
        else:
            # Case 2: There are fewer episodes than clusters
            ## Sample one episode per cluster for a random subset of clusters.
            random_order = np.random.permutation(NC)
            for i in random_order[: self.start_poses_per_obj]:
                episodes_per_cluster[i] = 1

        pathfinder = sim.pathfinder
        for i, num_cluster_episodes in enumerate(episodes_per_cluster):
            episode_count = 0
            cluster_center = cluster_centers[i]["center"]
            cluster_radius = max(3 * cluster_centers[i]["stddev"], 2.0)

            while episode_count < num_cluster_episodes and num_cluster_episodes > 0:
                for _ in range(self.start_retries):
                    start_position = pathfinder.get_random_navigable_point_near(
                        cluster_center, cluster_radius
                    ).astype(np.float32)

                    if (
                        start_position is None
                        or np.any(np.isnan(start_position))
                        or not sim.pathfinder.is_navigable(start_position)
                    ):
                        print(f"Skipping cluster {cluster_center}")
                        num_cluster_episodes = 0
                        break

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

                    # Check that the shortest path points are all on the same floor
                    path = habitat_sim.ShortestPath()
                    path.requested_start = start_position
                    path.requested_end = closest_pt
                    found_path = sim.pathfinder.find_path(path)

                    if not found_path:
                        continue

                    heights = [p.tolist()[1] for p in path.points]
                    h_delta = max(heights) - min(heights)

                    if h_delta > self.single_floor_threshold:
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
                    episode_count += 1
                    break
        return start_positions, start_rotations

    def _sample_start_poses(
        self,
        sim: Simulator,
        goals: List,
    ) -> Tuple[List, List]:
        viewpoint_locs = [
            [vp["agent_state"]["position"] for vp in goal["view_points"]]
            for goal in goals
        ]
        start_positions = []
        start_rotations = []
        geodesic_dists = []
        euclidean_dists = []

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

                euc_dist = np.linalg.norm(start_position - closest_pt)

                if not self.disable_euc_to_geo_ratio_check:
                    dist_ratio = geo_dist / euc_dist
                    if dist_ratio < self.min_geo_to_euc_ratio:
                        continue

                    # aggressive _ratio_sample_rate (copied from PointNav)
                    if np.random.rand() > (20 * (dist_ratio - 0.98) ** 2):
                        continue

                # Check that the shortest path points are all on the same floor
                path = habitat_sim.ShortestPath()
                path.requested_start = start_position
                path.requested_end = closest_pt
                found_path = sim.pathfinder.find_path(path)

                if not found_path:
                    continue

                heights = [p.tolist()[1] for p in path.points]
                h_delta = max(heights) - min(heights)

                if h_delta > self.single_floor_threshold:
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
                geodesic_dists.append(geo_dist)
                euclidean_dists.append(float(euc_dist))
                break

            else:
                # no start pose found after n attempts
                return [], [], [], []

        return start_positions, start_rotations, geodesic_dists, euclidean_dists

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
                    "radius": radius,
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
    ):
        if with_start_poses:
            assert with_viewpoints

        states = pose_sampler.sample_agent_poses_radially(obj.aabb.center, obj)
        observations = self._render_poses(sim, states)
        observations, states = self._can_see_object(observations, states, obj)

        if len(observations) == 0:
            return None

        frame_coverages = self._compute_frame_coverage(observations, obj.semantic_id)

        keep_goal = self._threshold_object_goals(frame_coverages)

        if sum(keep_goal) == 0:
            return None

        result = {
            "object_category": self.cat_map[obj.category.name()],
            "object_id": obj.id,
            "position": obj.aabb.center.tolist(),
            "children_object_categories": [],
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

    def dense_sampling_trimesh(self, triangles, density=25.0, max_points=200000):
        # Create trimesh mesh from triangles
        t_vertices = triangles.reshape(-1, 3)
        t_faces = np.arange(0, t_vertices.shape[0]).reshape(-1, 3)
        t_mesh = trimesh.Trimesh(vertices=t_vertices, faces=t_faces)
        surface_area = t_mesh.area
        n_points = min(int(surface_area * density), max_points)
        t_pts, _ = trimesh.sample.sample_surface_even(t_mesh, n_points)
        return t_pts

    def _cluster_navmesh(
        self,
        sim: Simulator,
        goals_by_category: Dict[str, Any],
        scene_name: str,
    ):
        # Discover navmesh clusters
        navmesh_triangles = np.array(sim.pathfinder.build_navmesh_vertices())
        navmesh_pc = self.dense_sampling_trimesh(navmesh_triangles)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity="euclidean",
            distance_threshold=1.0,
        ).fit(navmesh_pc)

        labels = clustering.labels_
        n_clusters = clustering.n_clusters_

        cluster_infos = []
        for i in range(n_clusters):
            center = navmesh_pc[labels == i, :].mean(axis=0)
            if sim.pathfinder.is_navigable(center):
                center = np.array(sim.pathfinder.snap_point(center)).tolist()
                locs = navmesh_pc[labels == i, :].tolist()
                stddev = np.linalg.norm(np.std(locs, axis=0)).item()
                cluster_infos.append({"center": center, "locs": locs, "stddev": stddev})
        print(f"====> Calculated cluster infos. # clusters: {n_clusters}")

        # Calculate distances from goals to cluster centers
        goal_category_to_cluster_distances = {}
        for cat, data in goals_by_category.items():
            object_vps = []
            for inst_data in data:
                for view_point in inst_data["view_points"]:
                    object_vps.append(view_point["agent_state"]["position"])
            goal_distances = []
            for i, cluster_info in enumerate(cluster_infos):
                dist, _ = self._geodesic_distance(
                    sim, cluster_info["center"], object_vps
                )
                goal_distances.append(dist)
            goal_category_to_cluster_distances[cat] = goal_distances

        if self.verbose:
            os.makedirs(
                os.path.join(self.plot_folder, "split", scene_name),
                exist_ok=True,
            )
            # Plot distances for visualization
            plt.figure(figsize=(8, 8))
            hist_data = list(filter(math.isfinite, goal_distances))
            hist_data = pd.DataFrame.from_dict({"Geodesic distance": hist_data})
            sns.histplot(data=hist_data, x="Geodesic distance")
            plt.title(cat)
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.plot_folder, "split", scene_name, f"{cat}.png")
            )
        return goal_category_to_cluster_distances, cluster_infos

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

        print("Total objects post filtering: {}".format(len(objects)))

        object_goals = {}
        results = []
        for obj in tqdm(objects, total=len(objects), dynamic_ncols=True):
            goal = self._make_goal(
                sim, pose_sampler, obj, with_viewpoints, with_start_poses
            )
            if goal is not None and len(goal["view_points"]) > 0:
                if goal["object_category"] not in object_goals:
                    object_goals[goal["object_category"]] = []

                object_goals[goal["object_category"]].append(goal)
                results.append((obj.id, obj.category.name(), len(goal["view_points"])))

        if self.sample_start_poses_wrt_navmesh_clusters:
            (
                goal_category_to_cluster_distances,
                cluster_infos,
            ) = self._cluster_navmesh(sim, object_goals, scene)

        all_goals = []
        for object_category, goals in tqdm(object_goals.items()):
            obj_goals = copy.deepcopy(goals)
            # Merge viewpoints using wordnet mapping
            if self.wordnet_map[object_category] is not None:
                # Add goals from wordnet children categories
                children_object_categories = [
                    cat
                    for cat in self.wordnet_map[object_category]
                    if cat in object_goals
                ]
                for category in children_object_categories:
                    for goal in object_goals[category]:
                        if len(goal["view_points"]) > 0:
                            obj_goals.append(goal)

                # Populate wordnet children categories for each goal
                for goal in goals:
                    goal["children_object_categories"] = children_object_categories

            geodesic_distances, euclidean_distances = [], []
            if self.sample_start_poses_wrt_navmesh_clusters:
                (
                    start_positions,
                    start_rotations,
                ) = self._sample_start_poses_wrt_clusters(
                    sim,
                    obj_goals,
                    cluster_infos,
                    np.array(goal_category_to_cluster_distances[object_category]),
                )
            else:
                (
                    start_positions,
                    start_rotations,
                    geodesic_distances,
                    euclidean_distances,
                ) = self._sample_start_poses(
                    sim,
                    obj_goals,
                )

            if len(start_positions) == 0:
                print("Start poses none for: {}".format(object_category))
                continue

            all_goals.append(
                {
                    "object_goals": goals,
                    "start_positions": start_positions,
                    "start_rotations": start_rotations,
                    "geodesic_distances": geodesic_distances,
                    "euclidean_distances": euclidean_distances,
                }
            )
        sim.close()
        return all_goals

    @staticmethod
    def _render_poses(
        sim: Simulator, agent_states: List[AgentState]
    ) -> List[Dict[str, ndarray]]:
        obs = []
        for agent_state in agent_states:
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
        children_object_categories=None,
    ):
        return OVONEpisode(
            episode_id=str(episode_id),
            goals=[],
            scene_id=scene_id,
            object_category=object_category,
            start_position=start_position,
            start_rotation=start_rotation,
            shortest_paths=shortest_paths,
            info=info,
            scene_dataset_config=scene_dataset_config,
            children_object_categories=children_object_categories,
        )

    def make_episodes(
        self,
        object_goals: Dict,
        scene: str,
        episodes_per_object: int = -1,
        split: str = "train",
    ):
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
                "Goal category: {} - viewpoints: {}, episodes: {}".format(
                    goals_category_id,
                    sum([len(gg["view_points"]) for gg in goal["object_goals"]]),
                    len(goal["start_positions"]),
                )
            )

            goals_by_category[goals_category_id].extend(goal["object_goals"])

            start_positions = goal["start_positions"]
            start_rotations = goal["start_rotations"]
            geodesic_distances = goal["geodesic_distances"]
            euclidean_distances = goal["euclidean_distances"]

            episodes_for_object = []
            for start_position, start_rotation, geo_dist, euc_dist in zip(
                start_positions,
                start_rotations,
                geodesic_distances,
                euclidean_distances,
            ):
                episode = self._create_episode(
                    episode_id=episode_count,
                    scene_id=scene.replace("data/scene_datasets/", ""),
                    scene_dataset_config="./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
                    start_position=start_position,
                    start_rotation=start_rotation,
                    info={
                        "geodesic_distance": geo_dist,
                        "euclidean_distance": euc_dist,
                    },
                    object_category=object_goal["object_category"],
                    children_object_categories=object_goal[
                        "children_object_categories"
                    ],
                )
                episodes_for_object.append(episode)
                episode_count += 1

            if split != "train" and episodes_per_object > 0:
                episodes_for_object = random.sample(
                    episodes_for_object,
                    min(episodes_per_object, len(episodes_for_object)),
                )

            dataset.episodes.extend(episodes_for_object)

            # Clean up children object categories
            for o_g in goal["object_goals"]:
                del o_g["children_object_categories"]

        dataset.goals_by_category = goals_by_category
        return dataset


def make_episodes_for_scene(args):
    (
        scene,
        outpath,
        device_id,
        split,
        start_poses_per_object,
        episodes_per_object,
        disable_euc_to_geo_ratio_check,
        disable_wordnet_label,
    ) = args
    if isinstance(scene, tuple) and outpath is None:
        scene, outpath = scene

    scene_name = os.path.basename(scene).split(".")[0]
    print(
        "Processing scene: {}, output_path: {}".format(
            scene, os.path.join(outpath, "{}.json.gz".format(scene_name))
        )
    )
    if os.path.exists(os.path.join(outpath, "{}.json.gz".format(scene_name))):
        print("Skipping scene: {}".format(scene))
        return

    # Load OVON whitelisted categories
    categories = load_json("data/hm3d_meta/ovon_categories.json")
    wordnet_mapping_file = "data/wordnet/wordnet_mapping.json"
    if disable_wordnet_label:
        wordnet_mapping_file = None

    if split in ["val_unseen_easy", "val_unseen_hard"]:
        categories = load_json("data/hm3d_meta/ovon_val_splits.json")

    objectgoal_maker = ObjectGoalGenerator(
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
        categories=categories[split],
        coverage_meta_file="data/coverage_meta/{}.pkl".format(split.split("_")[0]),
        frame_cov_thresh=0.05,
        goal_vp_cell_size=0.25,
        goal_vp_max_dist=1.0,
        start_poses_per_obj=start_poses_per_object,
        start_poses_tilt_angle=30.0,
        start_distance_limits=(1.0, 30.0),
        min_geo_to_euc_ratio=1.05,
        start_retries=2000,
        max_viewpoint_radius=1.0,
        wordnet_mapping_file=wordnet_mapping_file,
        device_id=device_id,
        sample_dense_viewpoints=True,
        disable_euc_to_geo_ratio_check=disable_euc_to_geo_ratio_check,
    )

    object_goals = objectgoal_maker.make_object_goals(
        scene=scene, with_viewpoints=True, with_start_poses=True
    )
    print("Scene: {}".format(scene))
    episode_dataset = objectgoal_maker.make_episodes(
        object_goals,
        scene,
        episodes_per_object=episodes_per_object,
        split=split,
    )

    scene_name = os.path.basename(scene).split(".")[0]
    save_to = os.path.join(outpath, f"{scene_name}.json.gz")
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    print("Total episodes: {}".format(len(episode_dataset.episodes)))
    objectgoal_maker.save_to_disk(episode_dataset, save_to)


def make_episodes_for_split(
    scenes: List[str],
    split: str,
    outpath: str,
    tasks_per_gpu: int = 1,
    enable_multiprocessing: bool = False,
    start_poses_per_object: int = 2000,
    episodes_per_object: int = -1,
    disable_euc_to_geo_ratio_check: bool = False,
    disable_wordnet_mapping: bool = False,
):
    import GPUtil


    dataset = habitat.datasets.make_dataset("OVON-v1")

    os.makedirs(outpath.format(split), exist_ok=True)
    save_to = os.path.join(
        outpath.format(split).replace("content/", ""), f"{split}.json.gz"
    )
    ObjectGoalGenerator.save_to_disk(dataset, save_to)

    deviceIds = GPUtil.getAvailable(order="memory", limit=1, maxLoad=1.0, maxMemory=1.0)

    if enable_multiprocessing:
        gpus = len(GPUtil.getAvailable(limit=256))
        cpu_threads = gpus * 16
        print("In multiprocessing setup - cpu {}, GPU: {}".format(cpu_threads, gpus))

        items = []
        for i, s in enumerate(scenes):
            deviceId = deviceIds[0]
            if i < gpus * tasks_per_gpu or len(deviceIds) == 0:
                deviceId = i % gpus
            items.append(
                (
                    s,
                    outpath.format(split),
                    deviceId,
                    split,
                    start_poses_per_object,
                    episodes_per_object,
                    disable_euc_to_geo_ratio_check,
                )
            )

        mp_ctx = multiprocessing.get_context("forkserver")
        # fmt: off
        with mp_ctx.Pool(cpu_threads) as pool, tqdm(
            total=len(scenes), position=0
        ) as pbar:
            for _ in pool.imap_unordered(make_episodes_for_scene, items):
                pbar.update()
        # fmt: on
    else:
        for scene in tqdm(scenes, total=len(scenes), dynamic_ncols=True):
            make_episodes_for_scene(
                (
                    scene,
                    outpath.format(split),
                    deviceIds[0],
                    split,
                    start_poses_per_object,
                    episodes_per_object,
                    disable_euc_to_geo_ratio_check,
                    disable_wordnet_mapping,
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/datasets/ovon/hm3d/v1_stretch/",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--tasks-per-gpu",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--multiprocessing",
        dest="enable_multiprocessing",
        action="store_true",
    )
    parser.add_argument(
        "--start-poses-per-object",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--episodes-per-object",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--disable-euc-geo-ratio-check",
        action="store_true",
        dest="disable_euc_to_geo_ratio_check",
    )
    parser.add_argument(
        "--disable-wordnet-mapping",
        action="store_true",
        dest="disable_wordnet_mapping",
    )

    args = parser.parse_args()
    scenes = None
    if args.scene is not None:
        scene_id = args.scene.split(".")[0] + ".basis.glb"
        scenes = [scene_id]
    else:
        split = args.split.split("_")[0]
        scenes = list(
            get_hm3d_semantic_scenes("data/scene_datasets/hm3d", [split])[split]
        )
        scenes = sorted(scenes)

    if args.num_scenes > 0:
        scenes = scenes[: args.num_scenes]
    print(scenes)
    print(
        "Start poses per object: {}, Episodes per object: {}, Split: {}".format(
            args.start_poses_per_object, args.episodes_per_object, args.split
        )
    )

    outpath = os.path.join(args.output_path, "{}/content/".format(args.split))
    make_episodes_for_split(
        scenes,
        args.split,
        outpath,
        args.tasks_per_gpu,
        args.enable_multiprocessing,
        args.start_poses_per_object,
        args.episodes_per_object,
        args.disable_euc_to_geo_ratio_check,
        args.disable_wordnet_mapping,
    )
