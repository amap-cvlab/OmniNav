import argparse
import multiprocessing
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple

import GPUtil
import habitat_sim
import numpy as np
from habitat_sim import bindings as hsim
from habitat_sim._ext.habitat_sim_bindings import SemanticObject
from habitat_sim.agent.agent import Agent, AgentConfiguration
from habitat_sim.simulator import Simulator
from scipy.spatial import distance
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from ovon.dataset.pose_sampler import PoseSampler
from ovon.dataset.semantic_utils import (
    ObjectCategoryMapping,
)
from ovon.dataset.visualization import (
    get_best_viewpoint_with_posesampler,
    get_bounding_box,
)


class PromptGenerator:
    semantic_spec_filepath: str
    img_size: Tuple[int, int]
    agent_height: float
    hfov: float
    agent_radius: float
    pose_sampler_args: Dict[str, Any]
    frame_cov_thresh: Tuple[float, float]
    device_id: int
    start_poses_tilt_angle: float
    cat_map: ObjectCategoryMapping
    max_viewpoint_radius: float
    max_dist_between_objects: float
    outpath: str
    dim: int

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
        frame_cov_thresh: Tuple[float, float],
        device_id: int = 0,
        start_poses_tilt_angle=30.0,
        max_viewpoint_radius: float = 1.0,
        max_dist_between_objects: float = 0.5,
        dim: int = 2,
        outpath: str = "",
    ) -> None:
        self.semantic_spec_filepath = semantic_spec_filepath
        self.img_size = img_size
        self.hfov = hfov
        self.agent_height = agent_height
        self.agent_radius = agent_radius
        self.sensor_height = sensor_height
        self.pose_sampler_args = pose_sampler_args
        self.frame_cov_thresh = frame_cov_thresh
        self.device_id = device_id
        self.start_poses_tilt_angle = start_poses_tilt_angle
        self.max_viewpoint_radius = max_viewpoint_radius
        self.cat_map = ObjectCategoryMapping(
            mapping_file=mapping_file,
            allowed_categories=categories,
            coverage_meta_file=coverage_meta_file,
            frame_coverage_threshold=frame_cov_thresh,
        )
        self.max_dist_between_objects = max_dist_between_objects
        self.dim = dim
        self.outpath = outpath

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
            sensor_spec.uuid = f"{name}"
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

    def save_img(self, img, scene_key, img_ref):
        os.makedirs(
            os.path.dirname(self.outpath.format("images") + scene_key + "/"),
            exist_ok=True,
        )
        (ToPILImage()(img)).convert("RGB").save(
            os.path.join(self.outpath.format("images"), f"{scene_key}/{img_ref}.png")
        )

    def save_meta_file(self, relationships, scene_key):
        with open(
            os.path.join(self.outpath.format("relationships"), f"{scene_key}.pkl"),
            "wb",
        ) as handle:
            pickle.dump(relationships, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_html(
        self,
        file_name: str,
        relationships: List,
        visualised: bool = True,
        threshold: float = 0.05,
        dim: int = 2,
    ) -> None:
        html_head = """
        <html>
        <head>
            <meta charset="utf-8">
            <title>Objects Spatial Relationships</title>
        </head>
        """
        html_style = """
        <style>
            /* Three image containers (use 25% for four, and 50% for two, etc) */
            .column {
            float: left;
            width: 20.00%;
            padding: 5px;
            }

            /* Clear floats after image containers */
            .box {
            box-sizing: border-box;
            }
            .row {
            display: flex;
            }
        </style>
        """
        html_script = """
        <script>
            var li_relationships = []
            function addRelationships(cb) {
            if (cb.checked) {
                li_relationships.push(cb.id);
            }
            else {
                var index = li_relationships.indexOf(cb.id);
                if (index > -1) {
                    li_relationships.splice(index, 1);
                }
            }
            localStorage.setItem("relationships",li_relationships)
            }
        </script>
        """
        len(relationships)
        html_body = """<body>
            <h2> Visualising {cnt} Relationships </h2>
            """
        for i, info_dict in enumerate(relationships):
            scene = info_dict["scene"]
            name = info_dict["name"]
            img_ref = info_dict["img_ref"]
            cov_sum = np.sum(info_dict["cov"])

            if i % 5 == 0:
                html_body += """<div class="row">"""
            html_body += f"""
                        <input type="checkbox" id="{scene}_{img_ref}" name="{scene}_{name}" onclick=addRelationships(this);>
                        <div class="column">
                            <img src="./../../../images/relationships_{dim}d/{scene}/{img_ref}.png" alt="{img_ref}" style="width:100%">
                            <h5>{img_ref} cov = {cov_sum:.3f}, frac = {area:.3f}, dist = {info_dict['distance']:.3f}</h5>
                        </div>
                        """
            if i % 5 == 4:
                html_body += "</div>"

        html_body += """</body>
                    </html>"""
        f = open(file_name, "w")
        f.write(html_head + html_style + html_script + html_body)
        f.close()

    def get_relation_3d(
        self,
        sim: Simulator,
        agent: Agent,
        pose_sampler: PoseSampler,
        a: SemanticObject,
        b: SemanticObject,
        closest_points: Tuple = None,
    ):
        """Finds spatial relationship [above,below,near] from 3D bounding boxes of objects and returns the image"""
        name_b = b.category.name()
        name_a = a.category.name()

        pt_1, pt_2 = closest_points
        disp = pt_1 - pt_2
        y_disp_greater = disp[1] >= np.linalg.norm(disp[0:3:2])

        def is_above(b: SemanticObject, a: SemanticObject, eps=0.05) -> bool:
            b_center = b.aabb.center
            a_center = a.aabb.center
            _, a_y, _ = a.aabb.sizes / 2
            _, b_y, _ = b.aabb.sizes / 2

            if b_center[1] - b_y + eps > a_center[1] + a_y:
                return True
            return False

        if is_above(b, a) and y_disp_greater:
            rel = f"{name_b} above {name_a}"
        elif is_above(a, b) and y_disp_greater:
            rel = f"{name_b} below {name_a}"
        else:
            rel = f"{name_b} near {name_a}"

        check, view = get_best_viewpoint_with_posesampler(sim, pose_sampler, [a, b])
        if check:
            cov, pose, _ = view
            agent.set_state(pose)
            obs = sim.get_sensor_observations()
            drawn_img, _, area = get_bounding_box(obs, [a, b])
            return True, rel, drawn_img, cov, area
        return False, None, None, 0, 0

    def get_relation_2d(
        self,
        sim: Simulator,
        agent: Agent,
        pose_sampler: PoseSampler,
        a: SemanticObject,
        b: SemanticObject,
        eps: float = 60,
    ):
        """Finds spatial relationship [above, below, near] from
        the 2D viewpoint with BB for objects and returns image"""
        check, view = get_best_viewpoint_with_posesampler(sim, pose_sampler, [a, b])
        name_b = b.category.name()
        name_a = a.category.name()

        if check:
            cov, pose, _ = view
            agent.set_state(pose)
            obs = sim.get_sensor_observations()
            total_obs_area = obs["semantic"].shape[0] * obs["semantic"].shape[1]
            drawn_img, bb, area = get_bounding_box(obs, [a, b])

            def get_intersection_area(boxA, boxB, total_obs_area):
                # Dimensions of both bounding boxes
                xA, yA = boxA[2] - boxA[0], boxA[3] - boxA[1]
                xB, yB = boxB[2] - boxB[0], boxB[3] - boxB[1]

                inter_xmin = max(boxA[0], boxB[0])
                inter_ymin = max(boxA[1], boxB[1])
                inter_xmax = min(boxA[2], boxB[2])
                inter_ymax = min(boxA[3], boxB[3])

                # Compute the area of intersection rectangle
                interArea = (
                    max((inter_xmax - inter_xmin, 0))
                    * max((inter_ymax - inter_ymin), 0)
                ) / total_obs_area

                if interArea == 0:
                    return 0, None, None

                # Check if edge of intersection is the left or right edges
                # Height of intersection rectangle will be greater than width
                side_intersection = max((inter_ymax - inter_ymin), 0) > 2 * max(
                    (inter_xmax - inter_xmin, 0)
                )

                # Check if width of intersection rectangle is nearly equal to size of smaller rectangle
                lower_edge_covered = max((inter_xmax - inter_xmin), 0) > 0.85 * (
                    min(xA, xB)
                )

                is_near = side_intersection or (not lower_edge_covered)
                is_b_above_a = False

                if not is_near:
                    if boxB[1] < boxA[1]:
                        is_b_above_a = True
                return interArea, is_near, is_b_above_a

            if np.sum(area) > 0:
                boxA = bb[0]
                boxB = bb[1]
                intersection, is_near, is_b_above_a = get_intersection_area(
                    boxA, boxB, total_obs_area
                )
                if intersection > 0:
                    # A lot of overlap between bounding boxes ('on' relationship)
                    if (intersection) >= 0.85 * (np.min(area)):
                        if area[0] > area[1]:
                            return (
                                True,
                                f"{name_b} on {name_a}",
                                bb,
                                drawn_img,
                                cov,
                                area,
                            )
                        else:
                            return False, None, None, None, None, None
                    else:
                        if is_near:
                            return (
                                True,
                                f"{name_b} near {name_a}",
                                bb,
                                drawn_img,
                                cov,
                                area,
                            )
                        elif is_b_above_a:
                            return (
                                True,
                                f"{name_b} above {name_a}",
                                bb,
                                drawn_img,
                                cov,
                                area,
                            )
                        else:
                            return (
                                True,
                                f"{name_b} below {name_a}",
                                bb,
                                drawn_img,
                                cov,
                                area,
                            )

                else:
                    xmin1, ymin1, xmax1, ymax1 = bb[0]
                    xmin2, ymin2, xmax2, ymax2 = bb[1]

                    xA, yA = boxA[2] - boxA[0], boxA[3] - boxA[1]
                    xB, yB = boxB[2] - boxB[0], boxB[3] - boxB[1]

                    # above/below relationships
                    inter_xmin = max(xmin1, xmin2)
                    inter_xmax = min(xmax2, xmax2)
                    edge_covered = max((inter_xmax - inter_xmin), 0) > 0.85 * (
                        min(xA, xB)
                    )

                    # above/below relationships in these cases
                    if ymin1 + eps > ymax2 and edge_covered:
                        rel = f"{name_b} above {name_a}"
                    elif ymin2 + eps > ymax1 and edge_covered:
                        rel = f"{name_b} below {name_a}"
                    else:
                        rel = f"{name_b} near {name_a}"
                    return True, rel, bb, drawn_img, cov, area

        return False, None, None, None, None, None

    def is_close_to(self, obj_a: SemanticObject, obj_b: SemanticObject) -> Tuple:
        def get_surface_points(obj_a: SemanticObject) -> np.ndarray:
            xmin, ymin, zmin = obj_a.aabb.center - obj_a.aabb.sizes / 2
            xmax, ymax, zmax = obj_a.aabb.center + obj_a.aabb.sizes / 2

            x_coords = np.linspace(xmin, xmax, num=100)
            y_coords = np.linspace(ymin, ymax, num=100)
            z_coords = np.linspace(zmin, zmax, num=100)

            points_minx = np.array([(xmin, y, z) for y, z in zip(y_coords, z_coords)])
            points_maxx = np.array([(xmax, y, z) for y, z in zip(y_coords, z_coords)])
            points_miny = np.array([(x, ymin, z) for x, z in zip(x_coords, z_coords)])
            points_maxy = np.array([(x, ymax, z) for x, z in zip(x_coords, z_coords)])
            points_minz = np.array([(x, y, zmin) for x, y in zip(x_coords, y_coords)])
            points_maxz = np.array([(x, y, zmax) for x, y in zip(x_coords, y_coords)])

            points = np.concatenate(
                (
                    points_minx,
                    points_maxx,
                    points_miny,
                    points_maxy,
                    points_minz,
                    points_maxz,
                )
            )
            return points

        pts_a = get_surface_points(obj_a)
        pts_b = get_surface_points(obj_b)
        dist = distance.cdist(pts_a, pts_b, "euclidean")
        pta, ptb = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        if dist[pta, ptb] < self.max_dist_between_objects:
            return True, dist[pta, ptb], pts_a[pta], pts_b[ptb]

        return (False, "", None, None)

    def get_spatial_relationships(
        self,
        scene: str,
    ) -> None:
        sim = self._config_sim(scene)
        agent = sim.get_agent(0)
        pose_sampler = PoseSampler(sim=sim, **self.pose_sampler_args)
        objects = [
            o
            for o in sim.semantic_scene.objects
            if self.cat_map[o.category.name()] is not None
        ]
        os.makedirs(f"data/images/relationships_{self.dim}d/{scene}", exist_ok=True)
        all_relationships = []
        for a in objects:
            for b in objects:
                if np.linalg.norm(a.aabb.center - b.aabb.center) > 2.0:
                    continue

                name_a = a.category.name()
                name_b = b.category.name()
                close, min_distance, pta, ptb = self.is_close_to(a, b)
                closest_points = (pta, ptb)

                if name_a != name_b and close:
                    if self.dim == 2:
                        check, rel, bbs, img, cov, frac = self.get_relation_2d(
                            sim, agent, pose_sampler, a, b
                        )
                    if self.dim == 3:
                        check, rel, img, cov, frac = self.get_relation_3d(
                            sim, agent, pose_sampler, a, b, closest_points
                        )
                    if check and all(
                        cov_obj >= self.frame_cov_thresh for cov_obj in cov
                    ):
                        name = rel.replace("/", "_").replace(" ", "_")
                        img_ref = name + f"_{b.semantic_id}_{a.semantic_id}"
                        print(f"Found relationship: {name}")
                        current_rel = ""
                        for reln in ["above", "below", "near", "on"]:
                            if reln in name:
                                current_rel = reln
                        all_relationships.append(
                            {
                                "scene": scene,
                                "relation": current_rel,
                                "ref_object": name_a,
                                "target_object": name_b,
                                "ref_obj_semantic_id": a.semantic_id,
                                "target_obj_semantic_id": b.semantic_id,
                                "distance": min_distance,
                                "cov": cov,
                                "area": frac,
                                "name": name,
                                "img_ref": img_ref,
                                "bbA": bbs[0],
                                "bbB": bbs[1],
                            }
                        )
                        self.save_img(img, scene, img_ref)
        sim.close()
        return all_relationships


def find_relationships_for_scene(args):
    scene, outpath, device_id, dim, vis = args
    if isinstance(scene, tuple) and outpath is None:
        scene, outpath = scene
    scene_key = os.path.basename(scene).split(".")[0]

    prompt_generator = PromptGenerator(
        semantic_spec_filepath=(
            "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        ),
        img_size=(2048, 2048),
        hfov=90,
        agent_height=1.41,
        agent_radius=0.17,
        sensor_height=1.31,
        pose_sampler_args={
            "r_min": 1.5,
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
        frame_cov_thresh=0.05,
        max_viewpoint_radius=1.0,
        start_poses_tilt_angle=30.0,
        device_id=device_id,
        dim=dim,
        max_dist_between_objects=0.5,
        outpath=outpath,
    )
    print(f"Starting scene: {scene_key}")
    relationships = prompt_generator.get_spatial_relationships(scene=scene_key)
    prompt_generator.save_meta_file(relationships=relationships, scene_key=scene_key)

    if vis:
        html_outpath = os.path.join(outpath.format("webpage"), f"{scene_key}.html")
        prompt_generator.create_html(html_outpath, relationships, dim=dim)


def find_relationships_for_split(
    split: str,
    outpath: str,
    vis: bool,
    multiprocessing_enabled: bool,
    tasks_per_gpu: int,
    dim: int,
    num_scenes: int = None,
):
    # HM3D_SCENES = get_hm3d_semantic_scenes("data/scene_datasets/hm3d")
    # scenes = list(HM3D_SCENES[split])
    scenes = ["XiJhRLvpKpX", "1S7LAXRdDqK"]

    print("Total number of scenes: ", len(scenes[:num_scenes]))
    if num_scenes is None:
        num_scenes = len(scenes)

    deviceIds = GPUtil.getAvailable(order="memory", limit=1, maxLoad=1.0, maxMemory=1.0)

    if multiprocessing_enabled:
        gpus = len(GPUtil.getAvailable(limit=256))
        cpu_threads = gpus * 16
        print("In multiprocessing setup - cpu {}, GPU: {}".format(cpu_threads, gpus))
        items = []
        for i, s in enumerate(scenes[:num_scenes]):
            deviceId = deviceIds[0]
            if i < gpus * tasks_per_gpu or len(deviceIds) == 0:
                deviceId = i % gpus
            items.append((s, outpath, deviceId, dim, vis))

        mp_ctx = multiprocessing.get_context("forkserver")
        with (
            mp_ctx.Pool(cpu_threads) as pool,
            tqdm(total=len(scenes[:num_scenes]), position=0) as pbar,
        ):
            for _ in pool.imap_unordered(find_relationships_for_scene, items):
                pbar.update()
    else:
        for scene in tqdm(scenes[:num_scenes], total=len(scenes), dynamic_ncols=True):
            find_relationships_for_scene((scene, outpath, deviceIds[0], dim, vis))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        help="split of data to be used",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dim",
        help="Dimension of BB to generate spatial relationships",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_scenes",
        help="Number of scenes to generate prompts for",
        type=int,
    )
    parser.add_argument(
        "--outpath",
        help="Output path for relationships dictionary and HTML webpage",
        type=str,
        default="data/",
    )
    parser.add_argument(
        "--vis",
        help="Create HTML webpage or not",
        default=False,
        action="store_true",
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

    split = args.split
    dim = args.dim
    num_scenes = args.num_scenes
    multiprocessing_enabled = args.multiprocessing_enabled
    tasks_per_gpu = args.tasks_per_gpu
    vis = args.vis

    # Output Path Information
    outpath = args.outpath
    outpath = os.path.join(outpath + "{}/", f"{split}/{dim}d/")

    if dim not in [2, 3]:
        print("Invalid Dimension. Please select dim = 2 or 3!")
        sys.exit(1)

    find_relationships_for_split(
        split=split,
        outpath=outpath,
        vis=vis,
        multiprocessing_enabled=multiprocessing_enabled,
        tasks_per_gpu=tasks_per_gpu,
        dim=dim,
        num_scenes=num_scenes,
    )
