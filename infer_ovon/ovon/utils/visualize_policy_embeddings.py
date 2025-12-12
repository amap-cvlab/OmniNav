import argparse

import numpy as np
import torch
from gym import spaces

from ovon.models.clip_policy import PointNavResNetCLIPPolicy
from ovon.utils.plot_tsne import plot_tsne
from ovon.utils.utils import save_pickle


def visualize_policy_embeddings(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    print(checkpoint.keys())

    OBJECT_MAPPING = {
        "chair": 0,
        "bed": 1,
        "plant": 2,
        "toilet": 3,
        "tv_monitor": 4,
        "sofa": 5,
    }

    h, w = (
        480,
        640,
    )

    observation_space = {
        "compass": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
        "gps": spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        ),
        "rgb": spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 3),
            dtype=np.uint8,
        ),
        "objectgoal": spaces.Box(
            low=0, high=len(OBJECT_MAPPING) - 1, shape=(1,), dtype=np.int64
        ),
    }

    observation_space = spaces.Dict(observation_space)

    action_space = spaces.Discrete(6)

    policy = PointNavResNetCLIPPolicy.from_config(
        checkpoint["config"], observation_space, action_space
    )
    policy.load_state_dict(
        {k.replace("actor_critic.", ""): v for k, v in checkpoint["state_dict"].items()}
    )

    policy.eval()
    print("Policy initialized....")

    embeddings = {}
    for cat, cat_id in OBJECT_MAPPING.items():
        out = policy.net.obj_categories_embedding(torch.tensor(cat_id))
        embeddings[cat] = out.detach().numpy()

    save_pickle(embeddings, "data/clip_embeddings/hm3d_onehot.pkl")
    plot_tsne(embeddings, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to the habitat baselines file",
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="path to the TSNE plot"
    )

    args = parser.parse_args()
    visualize_policy_embeddings(args.checkpoint, args.output_path)
