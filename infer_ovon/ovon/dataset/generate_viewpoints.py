from typing import Dict, List, Tuple

import habitat_sim
import numpy as np
from habitat_sim import bindings as hsim
from habitat_sim._ext.habitat_sim_bindings import SemanticObject
from habitat_sim.agent.agent import AgentConfiguration, AgentState
from habitat_sim.simulator import Simulator
from numpy import ndarray

from ovon.dataset.pose_sampler import PoseSampler
from ovon.dataset.semantic_utils import ObjectCategoryMapping
from ovon.dataset.visualization import save_candidate_imgs


def config_sim(
    scene_filepath: str,
    semantic_spec_filepath: str,
    img_size: Tuple[float],
    sensor_height: float,
    hfov: float,
    agent_height: float,
    agent_radius: float,
    device_id: int,
) -> Simulator:
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = device_id
    sim_cfg.scene_dataset_config_file = semantic_spec_filepath
    sim_cfg.scene_id = scene_filepath

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
        sensor_spec.resolution = [img_size[0], img_size[1]]
        sensor_spec.position = [0.0, sensor_height, 0.0]
        sensor_spec.hfov = hfov
        sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(sensor_spec)

    # create agent specifications
    agent_cfg = AgentConfiguration(
        height=agent_height,
        radius=agent_radius,
        sensor_specifications=sensor_specs,
    )

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    # set the navmesh
    assert sim.pathfinder.is_loaded, "pathfinder is not loaded!"
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_height = agent_height
    navmesh_settings.agent_radius = agent_radius
    navmesh_success = sim.recompute_navmesh(
        sim.pathfinder, navmesh_settings, include_static_objects=False
    )
    assert navmesh_success, "Failed to build the navmesh!"

    scene = sim.semantic_scene
    print(
        f"simulator loaded. House has {len(scene.levels)} levels, "
        f"{len(scene.regions)} regions and {len(scene.objects)} objects"
    )

    return sim


def compute_frame_coverage(obs: List[Dict[str, ndarray]], oid: int):
    img_shape = obs[0]["semantic_sensor"].shape
    pxls = img_shape[0] * img_shape[1]
    return [(o["semantic_sensor"].reshape(-1) == oid).sum() / float(pxls) for o in obs]


def determine_keep_frames(
    frame_covs: List[float],
    min_obj_cov: float,
):
    """Absolute thresholding of object coverage. Linear dynamic thresholding
    of frame coverage.
    """

    keep = []
    for fc in frame_covs:
        if fc < min_obj_cov:
            keep.append(False)
            continue
        keep.append(True)
    return keep


def render_keep_visible(sim, candidate_states, obj: SemanticObject):
    """render all observations. keep those that can see the object."""
    observations = []
    states = []
    for agent_state in candidate_states:
        sim.agents[0].set_state(agent_state, infer_sensor_states=False)
        obs = sim.get_sensor_observations()

        if np.isin(obj.semantic_id, obs["semantic_sensor"]):
            observations.append(obs)
            states.append(agent_state)

    return observations, states


def purge_by_keep_frames(obs, candidate_states, keeps):
    new_o = []
    new_cs = []
    for o, cs, k in zip(obs, candidate_states, keeps):
        if k:
            new_o.append(o)
            new_cs.append(cs)
    return new_o, new_cs


def object_goals_for_object(
    sim: Simulator,
    pose_sampler: PoseSampler,
    obj: SemanticObject,
    hfov: float,
    min_object_coverage: float,
    save_all_candidates: bool = True,
    save_filtered_candidates: bool = True,
) -> Tuple[List[Dict[str, ndarray]], List[AgentState]]:
    print(f"\nObject ID: `{obj.id}`. Size: {obj.aabb.sizes}")

    candidate_states = pose_sampler.sample_agent_poses_radially(obj)
    print(f"{len(candidate_states)} poses found.")

    observations, candidate_states = render_keep_visible(sim, candidate_states, obj)
    print(f"Rendered: {len(observations)} can see the object.")

    if len(observations) == 0:
        return [], [], [], [], 0.0

    frame_coverages = compute_frame_coverage(observations, obj.semantic_id)

    if save_all_candidates:
        save_candidate_imgs(
            observations,
            frame_coverages,
            f"data/object_goals_debug/{obj.id}",
        )

    keep_frames = determine_keep_frames(
        frame_coverages,
        min_object_coverage,
    )
    observations, candidate_states = purge_by_keep_frames(
        observations, candidate_states, keep_frames
    )

    if save_filtered_candidates:
        save_candidate_imgs(
            observations,
            frame_coverages,
            f"data/object_goals_filtered/{obj.id}",
        )

    print(len(observations), "images satisfied the criteria.")
    return (
        observations,
        candidate_states,
        frame_coverages,
    )


if __name__ == "__main__":
    hfov = 79

    hm3d_to_cat = ObjectCategoryMapping(
        mapping_file=(
            "ovon/dataset/source_data/Mp3d_category_mapping_fixed_taxonomy.tsv"
        ),
        allowed_categories={
            "chair",
            "bed",
            "toilet",
            "sofa",
            "plant",
            "tv_monitor",
        },
    )

    sim = config_sim(
        scene_filepath=(
            "data/scene_datasets/hm3d/val/00802-wcojb4TFT35/wcojb4TFT35.basis.glb"
        ),
        semantic_spec_filepath=(
            "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        ),
        img_size=(512, 512),
        sensor_height=0.88,  # can be overridden by PoseSampler h_{min,max}
        hfov=hfov,
        agent_height=0.88,
        agent_radius=0.18,
    )

    pose_sampler = PoseSampler(
        sim=sim,
        r_min=0.5,
        r_max=2.0,
        r_step=0.5,
        rot_deg_delta=10.0,
        h_min=0.8,
        h_max=1.4,
        sample_lookat_deg_delta=5.0,
    )

    objects = [
        o
        for o in sim.semantic_scene.objects
        if hm3d_to_cat[o.category.name()] is not None
    ][:5]

    x1, y1 = (0.0, 0.02)
    x2, y2 = (25.0, 0.6)
    m = (y2 - y1) / (x2 - x1)
    b = y2 - m * x2
    frame_cov_thresh_line = (m, b)

    for obj in objects:
        object_goals_for_object(
            sim,
            pose_sampler,
            obj,
            hfov,
            min_object_coverage=0.1,
            save_all_candidates=False,
            save_filtered_candidates=True,
        )

    sim.close()
