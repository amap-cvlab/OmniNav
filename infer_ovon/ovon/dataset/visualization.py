import os
from typing import Dict, List

import cv2
import numpy as np
import torch
from habitat.tasks.utils import compute_pixel_coverage
from habitat_sim._ext.habitat_sim_bindings import SemanticObject
from habitat_sim.agent.agent import AgentState
from habitat_sim.simulator import Simulator
from habitat_sim.utils.common import d3_40_colors_rgb
from numpy import ndarray
from PIL import Image
from torchvision.ops import masks_to_boxes
from torchvision.transforms import PILToTensor
from torchvision.utils import draw_bounding_boxes

from ovon.dataset.pose_sampler import PoseSampler

IMAGE_DIR = "data/images/ovon_dataset_gen/debug"
MAX_DIST = [0, 0, 200]  # Blue
NON_NAVIGABLE = [150, 150, 150]  # Grey
POINT_COLOR = [150, 150, 150]  # Grey
VIEW_POINT_COLOR = [0, 200, 0]  # Green
CENTER_POINT_COLOR = [200, 0, 0]  # Red

color2RGB = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Red": (255, 0, 0),
    "Lime": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "Silver": (192, 192, 192),
    "Gray": (128, 128, 128),
    "Maroon": (128, 0, 0),
    "Olive": (128, 128, 0),
    "Green": (0, 128, 0),
    "Purple": (128, 0, 128),
    "Teal": (0, 128, 128),
    "Navy": (0, 0, 128),
}


def get_depth(obs, objects):
    obj_depths = []
    for obj in objects:
        id = obj.semantic_id
        depth = np.mean(obs["depth"][obs["semantic"] == id])
        obj_depths.append("{:.2f}".format(depth))

    return obj_depths


def get_color(obs, objects):
    """
    Returns color name or None if object does not have specific color
    """
    rgb_key = "color" if "color" in obs.keys() else "rgb"
    colors = np.array(list(color2RGB.values()))
    obj_colors = []
    for obj in objects:
        id = obj.semantic_id
        rgb = obs[rgb_key][obs["semantic"] == id][:, :3]
        color_ids = np.argmin(
            np.linalg.norm(rgb[:, np.newaxis, :] - colors, axis=2),
            axis=1,
        )
        maj_color = np.bincount(color_ids).argmax()
        if (color_ids[color_ids == maj_color]).shape[0] / color_ids.shape[0] > 0.5:
            obj_colors.append(list(color2RGB.keys())[maj_color])
        else:
            obj_colors.append(None)
    return obj_colors


def obs_to_frame(obs):
    rgb = cv2.cvtColor(obs["color_sensor"], cv2.COLOR_BGRA2RGB)
    dmap = (obs["depth_sensor"] / 10 * 255).astype(np.uint8)
    dmap_colored = cv2.applyColorMap(dmap, cv2.COLORMAP_VIRIDIS)

    semantic_obs = obs["semantic_sensor"]
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    return np.concatenate([rgb, dmap_colored, semantic_img], axis=1)


def save_candidate_imgs(
    obs: List[Dict[str, ndarray]],
    frame_covs: List[float],
    save_to: str,
) -> None:
    """Write coverage stats on candidate images and save all to disk"""
    os.makedirs(save_to, exist_ok=True)

    font = cv2.FONT_HERSHEY_SIMPLEX
    loc = (50, 50)
    lt = cv2.LINE_AA
    w = (255, 255, 255)
    b = (0, 0, 0)

    for i, (o, fc) in enumerate(zip(obs, frame_covs)):
        txt = f"Frame coverage: {round(fc, 2)}"
        img = obs_to_frame(o)
        img = cv2.putText(img, txt, loc, font, 1, b, 4, lt)
        img = cv2.putText(img, txt, loc, font, 1, w, 2, lt)
        cv2.imwrite(os.path.join(save_to, f"candidate_{i}.png"), img)


def get_bounding_box(
    obs: List[Dict[str, ndarray]], objectList: List[SemanticObject], depths=None
):
    """Return the image with bounding boxes drawn on objects inside objectList"""
    N, H, W = (
        len(objectList),
        obs["semantic"].shape[0],
        obs["semantic"].shape[1],
    )
    masks = np.zeros((N, H, W))
    for i, object in enumerate(objectList):
        masks[i] = (obs["semantic"] == np.array([[(object.semantic_id)]])).reshape(
            (1, H, W)
        )

    boxes = masks_to_boxes(torch.from_numpy(masks))
    area = []
    for box in boxes:
        area.append(
            ((box[2] - box[0]) * (box[3] - box[1])).cpu().detach().numpy() / (H * W)
        )
    rgb_key = "color" if "color" in obs.keys() else "rgb"
    img = Image.fromarray(obs[rgb_key][:, :, :3], "RGB")
    if depths is None:
        labels = [
            f"{obj.category.name()}_{obj.semantic_id}"
            for i, obj in enumerate(objectList)
        ]
    else:
        labels = [
            f"{obj.category.name()}_{obj.semantic_id}_d = {depths[i]}"
            for i, obj in enumerate(objectList)
        ]
    drawn_img = draw_bounding_boxes(
        PILToTensor()(img),
        boxes,
        colors="red",
        width=2,
        labels=labels,
        font_size=10,
    )
    boxes = boxes.cpu().detach().numpy()
    return drawn_img, boxes, area


def _get_iou_pose(sim: Simulator, pose: AgentState, objectList: List[SemanticObject]):
    """Get coverage of all the objects in the objectList"""
    agent = sim.get_agent(0)
    agent.set_state(pose)
    obs = sim.get_sensor_observations()
    cov = np.zeros((len(objectList), 1))
    for i, obj in enumerate(objectList):
        cov_obj = compute_pixel_coverage(obs["semantic"], obj.semantic_id)
        if cov_obj <= 0:
            return None, None, "Failure: All Objects are not Visualized"
        cov[i] = cov_obj
    return cov, pose, "Successs"


def get_best_viewpoint_with_posesampler(
    sim: Simulator,
    pose_sampler: PoseSampler,
    objectList: List[SemanticObject],
):
    search_center = np.mean(np.array([obj.aabb.center for obj in objectList]), axis=0)
    candidate_states = pose_sampler.sample_agent_poses_radially(search_center)
    candidate_poses_ious = list(
        _get_iou_pose(sim, pos, objectList) for pos, _ in candidate_states
    )
    candidate_poses_ious_filtered = [
        p for p in candidate_poses_ious if (p[0] is not None)
    ]
    candidate_poses_sorted = sorted(
        candidate_poses_ious_filtered, key=lambda x: np.sum(x[0]), reverse=True
    )
    if candidate_poses_sorted:
        return True, candidate_poses_sorted[0]
    else:
        return False, None
