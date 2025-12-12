import argparse
import os
from collections import defaultdict

import clip
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from ovon.utils.utils import load_json

PROMPT = "{category}"


def clip_embeddings(clip_m, prompts):
    tokens = []
    for prompt in prompts:
        tokens.append(clip.tokenize(prompt, context_length=77).numpy())

    batch = torch.tensor(np.array(tokens)).cuda()
    with torch.no_grad():
        text_embedding = clip_m.encode_text(batch.flatten(0, 1)).float()
    return text_embedding


def max_similarity(clip_m, category, val_seen_categories):
    categories = val_seen_categories.copy()
    if category in categories:
        categories.remove(category)

    prompt = PROMPT.format(category=category)
    text_embedding = clip_embeddings(clip_m, [prompt] + categories)
    return (
        torch.cosine_similarity(text_embedding[0].unsqueeze(0), text_embedding[1:])
        .max()
        .item()
    )


def plot_scatterplot(x, y, output_path):
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=x, y=y, ax=ax)

    fig.savefig(output_path)


def plot_barplot(x, y, output_path):
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(x=x, y=y, ax=ax)

    fig.savefig(output_path)


def region_analysis(input_path, output_path):
    categories_by_region = load_json("data/hm3d_meta/hm3d_categories_by_region.json")

    region_per_category = {}
    for region, categories in categories_by_region.items():
        for category in categories:
            region_per_category[category] = region

    files = [
        "/coc/testnvme/nyokoyama3/public/ft_dagger_ckpt_30_vue.json",
        "/coc/testnvme/nyokoyama3/public/ft_dagger_ckpt_30_vuh.json",
    ]

    val_seen_categories = load_json("data/hm3d_meta/ovon_categories.json")["val_seen"]
    val_seen_categories = [
        PROMPT.format(category=category) for category in val_seen_categories
    ]

    clip_m, preprocess = clip.load("RN50", device="cuda")

    for file in files:
        print("Split: {}".format(file.split("/")[-1]))
        metrics = load_json(file)

        episodes_per_region = defaultdict(list)
        success_per_region = defaultdict(int)
        all_categories = []
        for k, meta in metrics.items():
            region = region_per_category.get(meta["target"])
            region = region if region is not None else "NaN"
            success_per_region[region] += meta["success"]
            episodes_per_region[region].append(meta)
            all_categories.append(meta["target"])

        all_categories = list(set(all_categories))

        category_to_max_similarity = {}
        for category in all_categories:
            region = region_per_category.get(category)
            if region is None:
                continue
            region_categories = categories_by_region[region]

            vs_region_categories = list(
                set(region_categories).intersection(set(val_seen_categories))
            )
            category_to_max_similarity[category] = round(
                max_similarity(clip_m, category, vs_region_categories), 2
            )

        for region, episodes in episodes_per_region.items():
            success_per_similarity = defaultdict(int)
            count_per_similarity = defaultdict(int)

            for episode in episodes:
                cos_sim = category_to_max_similarity.get(episode["target"])
                if cos_sim is None:
                    continue
                success_per_similarity[cos_sim] += episode["success"]
                count_per_similarity[cos_sim] += 1

            success_per_similarity = {
                k: v / count_per_similarity[k]
                for k, v in success_per_similarity.items()
            }

            out_path = os.path.join(
                output_path, "{}_success_vs_sim.png".format(region.replace("/", "_"))
            )

            plot_scatterplot(
                list(count_per_similarity.keys()),
                list(success_per_similarity.values()),
                out_path,
            )

        # print(category_to_max_similarity)

        # print({k: v / 3000 for k, v in success_per_region.items()})
        # print({k: v / 3000 for k, v in episodes_per_region.items()})
        # # print(list(set(categories)))
        # for region, categories in categories_by_region.items():
        #     intersection = set(all_categories).intersection(set(categories))
        #     if region == "bathroom":
        #         print("Region: {}, Intersection: {}".format(region, intersection))

        # print("Categories in NaN: {}".format(set(all_categories).intersection(set(categories_by_region["NaN"]))))
        # print("Regions: {}".format(set(categories_by_region.keys())))


def semantic_failures():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, help="Path to the episode metrics file", required=True
    )
    parser.add_argument(
        "--output-path", type=str, help="Path to the output file", required=True
    )
    args = parser.parse_args()
    region_analysis(args.input_path, args.output_path)
