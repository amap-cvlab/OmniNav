import argparse
import multiprocessing
import os
import pickle
from typing import Dict, Union

import GPUtil
import habitat
import habitat_sim
import numpy as np
from habitat.config.default import get_agent_config, get_config
from habitat.config.default_structured_configs import (
    HabitatSimSemanticSensorConfig,
)
from habitat.config.read_write import read_write
from habitat_sim._ext.habitat_sim_bindings import BBox
from habitat_sim.simulator import Simulator
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from ovon.dataset.pose_sampler import PoseSampler
from ovon.dataset.semantic_utils import (
    ObjectCategoryMapping,
    get_hm3d_semantic_scenes,
)
from ovon.dataset.visualization import (
    get_best_viewpoint_with_posesampler,
    get_bounding_box,
    get_color,
    get_depth,
)

SCENES_ROOT = "data/scene_datasets/hm3d"
NUM_GPUS = len(GPUtil.getAvailable(limit=256))
TASKS_PER_GPU = 12


def create_html(
    file_name: str,
    objects_mapping: Dict,
    visualised: bool = True,
    threshold: float = 0.05,
):
    html_head = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Objects Visualisation</title>
    </head>"""
    html_style = """
    <style>
        /* Three image containers (use 25% for four, and 50% for two, etc) */
        .column {
        float: left;
        width: 20.00%;
        padding: 5px;
        }

        /* Clear floats after image containers */
        {
        box-sizing: border-box;
        }

        .row {
        display: flex;
        }
    </style>
    """
    html_script = """
    <script>
    var li_categories = []

    const download0 = () => (
        encodeURIComponent(
            JSON.stringify(
            localStorage.getItem('categories')
            ),
            null,
            2
            )
    )

    function addObjectToCategories(cb) {
    if (cb.checked) {
        li_categories.push(cb.id);
    }
    else {
        var index = li_categories.indexOf(cb.id);
        if (index > -1) {
            li_categories.splice(index, 1);
        }
    }

    console.log(li_categories)
    localStorage.setItem("categories",li_categories)
    download0()

    }
    </script>
    """
    cnt = 0
    html_body = ""
    for obj in objects_mapping.keys():
        # Visualized Objects
        if visualised and objects_mapping[obj][0][1] >= threshold:
            cnt += 1
            html_body += f"""<h3>{obj}</h3><input name="checkbox" onclick="addObjectToCategories(this);" type="checkbox" id="{obj}" />
                            <div class="row">
                            """
            for cov, frac, scene in objects_mapping[obj][:5]:
                html_body += f"""
                            <div class="column">
                                <img src="../images/objects/{scene}/{obj}.png" alt="{obj}" style="width:100%">
                                <h5>cov = {cov:.3f}, frac = {frac:.3f}</h5>
                            </div>
                            """
            html_body += "</div>"
        # Filtered Objects
        elif not visualised and objects_mapping[obj][0][1] < threshold:
            cnt += 1
            html_body += f"""<h3>{obj}</h3>
                            <div class="row">
                            """
            for cov, frac, scene in objects_mapping[obj][:5]:
                html_body += f"""
                            <div class="column">
                                <img src="../images/objects/{scene}/{obj}.png" alt="{obj}" style="width:100%">
                                <h5>cov = {cov:.3f}, frac = {frac:.3f}</h5>
                            </div>
                            """
            html_body += "</div>"
    html_body = f"""
                <body>
                <h2> Visualising {cnt} objects </h2>
                """ + html_body
    html_body += """</body>
                    </html>"""
    f = open(file_name, "w")
    f.write(html_head + html_style + html_script + html_body)
    f.close()


def save_img(img, path):
    (ToPILImage()(img)).convert("RGB").save(path)


def get_objnav_config(i: int, scene: str):
    CFG = "habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
    SCENE_CFG = f"{SCENES_ROOT}/hm3d_annotated_basis.scene_dataset_config.json"
    objnav_config = get_config(CFG)

    with read_write(objnav_config):
        agent_config = get_agent_config(objnav_config.habitat.simulator)

        # Stretch agent
        agent_config.height = 1.41
        agent_config.radius = 0.17

        sensor_pos = [0, 1.31, 0]

        agent_config.sim_sensors.update(
            {"semantic_sensor": HabitatSimSemanticSensorConfig()}
        )
        FOV = 90

        for sensor, sensor_config in agent_config.sim_sensors.items():
            agent_config.sim_sensors[sensor].hfov = FOV
            agent_config.sim_sensors[sensor].width //= 2
            agent_config.sim_sensors[sensor].height //= 2
            agent_config.sim_sensors[sensor].position = sensor_pos

        objnav_config.habitat.task.measurements = {}

        deviceIds = GPUtil.getAvailable(
            order="memory", limit=1, maxLoad=1.0, maxMemory=1.0
        )
        if i < NUM_GPUS * TASKS_PER_GPU or len(deviceIds) == 0:
            deviceId = i % NUM_GPUS
        else:
            deviceId = deviceIds[0]
        objnav_config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
            deviceId  # i % NUM_GPUS
        )
        objnav_config.habitat.dataset.scenes_dir = "./data/scene_datasets/"
        objnav_config.habitat.dataset.split = "train"
        objnav_config.habitat.simulator.scene = scene
        objnav_config.habitat.simulator.scene_dataset = SCENE_CFG
    return objnav_config


def get_simulator(objnav_config) -> Simulator:
    sim = habitat.sims.make_sim("Sim-v0", config=objnav_config.habitat.simulator)
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = (
        objnav_config.habitat.simulator.agents.main_agent.radius
    )
    navmesh_settings.agent_height = (
        objnav_config.habitat.simulator.agents.main_agent.height
    )
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    return sim


def is_on_ceiling(sim: Simulator, aabb: BBox):
    point = np.asarray(aabb.center)
    snapped = sim.pathfinder.snap_point(point)

    # The snapped point is on the floor above
    # It is more than 20 cms above the object's upper edge
    if snapped[1] > point[1] + aabb.sizes[0] / 2 + 0.20:
        return True

    # Snapped point is on the ground
    if snapped[1] < point[1] - 1.5:
        return True
    return False


def get_all_objects_in_view(obs, target_obj, threshold=0.01):
    area = np.prod(obs["semantic"].shape)
    obj_ids, num_pixels = np.unique(obs["semantic"], return_counts=True)
    objects = [
        obj_ids[i]
        for i in range(len(num_pixels))
        if num_pixels[i] / (area) > threshold and obj_ids[i] != target_obj
    ]
    return objects


def get_objects_for_scene(args) -> None:
    scene, outpath, device_id = args
    scene_key = os.path.basename(scene).split(".")[0]

    split = outpath

    """
    if os.path.isfile(os.path.join(outpath, f"meta/{scene_key}.pkl")):
        return
    """

    cfg = get_objnav_config(device_id, scene_key)
    sim = get_simulator(cfg)

    objects_info = sim.semantic_scene.objects
    objects_dict = {obj.semantic_id: obj for obj in objects_info}

    pose_sampler = PoseSampler(
        sim=sim,
        r_min=1.0,
        r_max=2.0,
        r_step=0.5,
        rot_deg_delta=10.0,
        h_min=0.8,
        h_max=1.4,
        sample_lookat_deg_delta=5.0,
    )

    split = outpath.split("/")[-2]
    print(split)
    objects_visualized = []
    cnt = 0
    agent = sim.get_agent(0)

    cat_map = ObjectCategoryMapping(
        mapping_file="ovon/dataset/source_data/Mp3d_category_mapping.tsv",
        allowed_categories=None,
        coverage_meta_file="data/coverage_meta/{}.pkl".format(split),
        frame_coverage_threshold=0.05,
    )

    os.makedirs(os.path.join(outpath, f"images/{scene_key}"), exist_ok=True)

    object_view_data = []
    objects_info = list(
        filter(lambda obj: cat_map[obj.category.name()] is not None, objects_info)
    )

    for object in objects_info:
        name = object.category.name().replace("/", "_")
        if is_on_ceiling(sim, object.aabb):
            continue

        check, view = get_best_viewpoint_with_posesampler(sim, pose_sampler, [object])
        if check:
            cov, pose, _ = view
            if cov < 0.05:
                continue
            agent.set_state(pose)
            obs = sim.get_sensor_observations()
            object_ids_in_view = get_all_objects_in_view(obs, object.semantic_id)
            objects_in_view = list(
                filter(
                    lambda obj: obj is not None
                    and (
                        cat_map[obj.category.name()] is not None
                        or "wall" in obj.category.name().lower()
                    ),
                    [*map(objects_dict.get, object_ids_in_view)],
                )
            )
            colors = get_color(obs, [object] + objects_in_view)
            depths = get_depth(obs, [object] + objects_in_view)
            drawn_img, bbs, area_covered = get_bounding_box(
                obs, [object] + objects_in_view, depths=depths
            )
            if np.sum(area_covered) > 0:
                # Save information of this object and all objects on top
                path = os.path.join(
                    outpath,
                    f"images/{scene_key}/{name}_{object.semantic_id}.png",
                )
                save_img(drawn_img, path)
                view_info = {
                    "target_obj_id": object.semantic_id,
                    "target_obj_name": object.category.name(),
                    "target_obj_2d_bb": bbs[0],
                    "target_obj_3d_bb": {
                        "center": object.aabb.center,
                        "sizes_x_y_z": object.aabb.sizes,
                    },
                    "target_obj_depth": depths[0],
                    "target_obj_color": colors[0],
                    "ref_objects": {
                        f"{obj.category.name()}_{obj.semantic_id}": {
                            "2d_bb": bbs[i + 1],
                            "3d_bb": {
                                "center": obj.aabb.center,
                                "sizes_x_y_z": obj.aabb.sizes,
                            },
                        }
                        for i, obj in enumerate(objects_in_view)
                    },
                    "scene": scene_key,
                    "img_ref": path,
                }
                object_view_data.append(view_info)
                objects_visualized.append(object.category.name().strip())
                cnt += 1

    meta_save_path = os.path.join(outpath, f"meta/{scene_key}.pkl")
    with open(meta_save_path, "wb") as handle:
        pickle.dump(object_view_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_objects_for_split(
    split: str,
    outpath: str,
    num_scenes: Union[int, None],
    tasks_per_gpu: int = 1,
    multiprocessing_enabled: bool = False,
):
    """Makes episodes for all scenes in a split"""

    scenes = sorted(
        list(get_hm3d_semantic_scenes("data/scene_datasets/hm3d", [split])[split])
    )
    num_scenes = len(scenes) if num_scenes is None else num_scenes
    scenes = scenes[:num_scenes]

    scenes = ["vLpv2VX547B"]
    print(scenes)
    print(
        "Starting visualisation for split {} with {} scenes".format(split, len(scenes))
    )

    os.makedirs(os.path.join(outpath.format(split), "meta"), exist_ok=True)

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
            for _ in pool.imap_unordered(get_objects_for_scene, items):
                pbar.update()
    else:
        for scene in tqdm(scenes, total=len(scenes), dynamic_ncols=True):
            get_objects_for_scene((scene, outpath.format(split), 0))


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
        "-o",
        "--outpath",
        help="output path",
        type=str,
        default="data/object_views/{}/",
    )
    parser.add_argument(
        "-n",
        "--num_scenes",
        help="number of scenes",
        type=int,
    )
    parser.add_argument("--tasks_per_gpu", help="number of scenes", type=int, default=1)
    parser.add_argument(
        "-m",
        "--multiprocessing_enabled",
        dest="multiprocessing_enabled",
        action="store_true",
    )
    args = parser.parse_args()
    split = args.split
    num_scenes = args.num_scenes
    outpath = args.outpath
    tasks_per_gpu = args.tasks_per_gpu
    multiprocessing_enabled = args.multiprocessing_enabled
    get_objects_for_split(
        split, outpath, num_scenes, tasks_per_gpu, multiprocessing_enabled
    )
