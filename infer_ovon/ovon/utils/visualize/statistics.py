import argparse
import glob
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ovon.utils.utils import load_dataset


def plot_statistics(path, output_path, split="train"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    geodesic_distances = []
    euclidean_distances = []
    files = glob.glob(os.path.join(path, "*json.gz"))

    categories = {}
    for f in tqdm(files):
        dataset = load_dataset(f)
        for ep in dataset["episodes"]:
            geodesic_distances.append(ep["info"]["geodesic_distance"])
            euclidean_distances.append(ep["info"]["euclidean_distance"])
            categories[ep["object_category"]] = (
                categories.get(ep["object_category"], 0) + 1
            )

    # Plot distances for visualization
    plt.figure(figsize=(8, 8))
    hist_data = list(filter(math.isfinite, geodesic_distances))
    hist_data = pd.DataFrame.from_dict({"Geodesic distance": hist_data})
    ax = sns.histplot(data=hist_data, x="Geodesic distance")

    ax.set_xticks(range(0, 32, 2))

    plt.title("Geodesic distance to closest goal")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "ovon_{}_geodesic_distances.png".format(split))
    )

    plt.figure(figsize=(8, 8))
    hist_data = list(
        filter(
            math.isfinite,
            [g / e for g, e in zip(geodesic_distances, euclidean_distances)],
        )
    )
    hist_data = pd.DataFrame.from_dict({"Euc Geo ratio": hist_data})
    ax = sns.histplot(data=hist_data, x="Euc Geo ratio")

    ax.set_ylim(0, 5000)
    ax.set_xlim(0, 5)
    ax.set_xticks(range(0, 5, 1))

    plt.title("Euc Geo ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "ovon_{}_euc_geo_ratio.png".format(split)))

    categories = {
        "objects": list(categories.keys()),
        "frequency": list(categories.values()),
    }

    df = pd.DataFrame.from_dict(categories)
    df.sort_values(by="frequency", inplace=True, ascending=False)
    print(df.columns)

    fig, axs = plt.subplots(1, 1, figsize=(8, 50))

    sns.barplot(data=df, x="frequency", y="objects", ax=axs)

    fig.savefig(
        os.path.join(output_path, "ovon_{}_categories.png".format(split)),
        dpi=100,
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="data/datasets/ovon/hm3d/v4_stretch/val_seen/content/",
    )
    parser.add_argument("--output-path", type=str, default="val_unseen.png")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    plot_statistics(args.path, args.output_path, args.split)
