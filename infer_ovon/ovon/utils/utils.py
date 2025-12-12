import glob
import gzip
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from habitat.utils.visualizations import maps
from PIL import Image
from tqdm import tqdm

from ovon.models.encoders.resnet_gn import ResNet


def write_json(data, path):
    with open(path, "w") as file:
        file.write(json.dumps(data))


def load_json(path):
    file = open(path, "r")
    data = json.loads(file.read())
    return data


def write_txt(data, path):
    with open(path, "w") as file:
        file.write("\n".join(data))


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save(file_name)


def load_dataset(path):
    with gzip.open(path, "rt") as file:
        data = json.loads(file.read(), encoding="utf-8")
    return data


def save_pickle(data, path):
    file = open(path, "wb")
    data = pickle.dump(data, file)


def load_pickle(path):
    file = open(path, "rb")
    data = pickle.load(file)
    return data


def write_dataset(data, path):
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


def is_on_same_floor(height, ref_floor_height, ceiling_height=0.5):
    return (
        (ref_floor_height - ceiling_height)
        <= height
        < (ref_floor_height + ceiling_height)
    )


def draw_point(sim, top_down_map, position, point_type, point_padding=2):
    t_x, t_y = maps.to_grid(
        position[2],
        position[0],
        (top_down_map.shape[0], top_down_map.shape[1]),
        sim=sim,
    )
    top_down_map[
        t_x - point_padding : t_x + point_padding + 1,
        t_y - point_padding : t_y + point_padding + 1,
    ] = point_type
    return top_down_map


def draw_bounding_box(
    sim, top_down_map, goal_object_id, ref_floor_height, line_thickness=4
):
    sem_scene = sim.semantic_annotations()
    object_id = goal_object_id

    sem_obj = None
    for object in sem_scene.objects:
        if object.id == object_id:
            sem_obj = object
            break

    center = sem_obj.aabb.center
    x_len, _, z_len = sem_obj.aabb.sizes / 2.0
    # Nodes to draw rectangle
    corners = [
        center + np.array([x, 0, z])
        for x, z in [
            (-x_len, -z_len),
            (-x_len, z_len),
            (x_len, z_len),
            (x_len, -z_len),
            (-x_len, -z_len),
        ]
        if is_on_same_floor(center[1], ref_floor_height=ref_floor_height)
    ]

    map_corners = [
        maps.to_grid(
            p[2],
            p[0],
            (
                top_down_map.shape[0],
                top_down_map.shape[1],
            ),
            sim=sim,
        )
        for p in corners
    ]

    maps.draw_path(
        top_down_map,
        map_corners,
        maps.MAP_TARGET_BOUNDING_BOX,
        line_thickness,
    )
    return top_down_map


def load_encoder(encoder, path):
    assert os.path.exists(path)
    if isinstance(encoder.backbone, ResNet):
        state_dict = torch.load(path, map_location="cpu")["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        return encoder.load_state_dict(state_dict=state_dict, strict=False)
    else:
        raise ValueError("unknown encoder backbone")


def count_episodes(path):
    files = glob.glob(os.path.join(path, "*.json.gz"))
    count = 0
    categories = defaultdict(int)
    for f in tqdm(files):
        dataset = load_dataset(f)
        for episode in dataset["episodes"]:
            categories[episode["object_category"]] += 1
        count += len(dataset["episodes"])
    print("Total episodes: {}".format(count))
    print("Categories: {}".format(categories))
    print("Total categories: {}".format(len(categories)))
    return count, categories
