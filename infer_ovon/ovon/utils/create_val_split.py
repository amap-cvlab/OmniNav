import argparse
import glob
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from ovon.dataset.semantic_utils import WordnetMapping
from ovon.utils.utils import load_dataset, load_json, write_dataset, write_json


def group_by_wordnet_ancestors(
    categories, wordnet_mapping_path="data/wordnet/wordnet_mapping.json"
):
    wordnet_mapping = WordnetMapping(wordnet_mapping_path)

    child_to_ancestor_map = {}

    for ancestor, children in wordnet_mapping.items():
        for child in children:
            child_to_ancestor_map[child] = ancestor

    categories_grouped_by_ancestor = {}
    for category in categories:
        ancestor = child_to_ancestor_map.get(category)
        if ancestor is None or ancestor == "object":
            categories_grouped_by_ancestor[category] = [category]
        if ancestor not in categories_grouped_by_ancestor:
            categories_grouped_by_ancestor[ancestor] = []
        categories_grouped_by_ancestor[ancestor].append(category)

    return categories_grouped_by_ancestor, child_to_ancestor_map


def validate_val_unseen_categories(
    train_categories, val_categories, child_to_ancestor_map
):
    train_ancestors = [
        child_to_ancestor_map[c]
        for c in train_categories
        if child_to_ancestor_map.get(c) is not None
    ]

    valid_val_categories = []
    for cat in val_categories:
        if (
            child_to_ancestor_map.get(cat) is None
            or child_to_ancestor_map.get(cat) not in train_ancestors
        ):
            valid_val_categories.append(cat)
    return valid_val_categories


def sample_val_categories(bin_to_categories, val_categories):
    for bin in bin_to_categories:
        if len(bin_to_categories[bin]) == 0:
            continue
        filtered_cats = list(set(bin_to_categories[bin]) - set(val_categories))
        val_categories.append(np.random.choice(filtered_cats))
    return val_categories


def load_val_categories(path="data/hm3d_meta/ovon_val_categories.csv"):
    df = pd.read_csv(path)
    val_unseen_easy = list(set(df["val_unseen_easy"].values))
    print(type(df["val_unseen_hard"]))
    val_unseen_hard = list(
        set(list(df["val_unseen_hard"].values) + list(df["train_blacklist"].values))
    )
    print(len(val_unseen_easy), len(val_unseen_hard))
    return val_unseen_easy, val_unseen_hard


def split_val_unseen_manual(path, output_path):
    categories = []
    category_count = {}

    val_unseen_easy, val_unseen_hard = load_val_categories()

    for split in ["train"]:
        files = glob.glob(os.path.join(path, "{}/content/".format(split), "*.json.gz"))

        for i, file in enumerate(tqdm(files)):
            dataset = load_dataset(file)

            goal_categories = [
                key.split("_")[1] for key in dataset["goals_by_category"].keys()
            ]
            categories.extend(goal_categories)
            for cat in goal_categories:
                if cat not in category_count:
                    category_count[cat] = 0
                category_count[cat] += 1

    train_categories = list(
        set(categories) - set(val_unseen_hard) - set(val_unseen_easy)
    )
    train_categories = sorted(
        train_categories, key=lambda x: category_count[x], reverse=True
    )

    data = {
        "train": train_categories,
        "val_seen": train_categories,
        "val_unseen_easy": list(set(val_unseen_easy)),
        "val_unseen_hard": list(set(val_unseen_hard)),
    }

    for k, v in data.items():
        print("Split: {}, Categories: {}".format(k, len(v)))

    write_json(
        data,
        os.path.join(output_path, "ovon_categories.json"),
    )


def split_val_unseen(path, output_path, n_bins=50):
    categories = []
    category_count = {}

    val_categories = []
    val_category_count = {}

    for split in ["train", "val"]:
        files = glob.glob(os.path.join(path, "{}/content/".format(split), "*.json.gz"))

        for file in tqdm(files):
            dataset = load_dataset(file)

            goal_categories = [
                key.split("_")[1] for key in dataset["goals_by_category"].keys()
            ]
            if split == "val":
                val_categories.extend(goal_categories)
                for cat in goal_categories:
                    if cat not in val_category_count:
                        val_category_count[cat] = 0
                    val_category_count[cat] += 1
            else:
                categories.extend(goal_categories)
                for cat in goal_categories:
                    if cat not in category_count:
                        category_count[cat] = 0
                    category_count[cat] += 1

    val_categories = list(set(val_categories) - set(categories))
    print("Only val categories: {}".format(val_categories))

    _, child_to_ancestor_map = group_by_wordnet_ancestors(list(category_count.keys()))
    ordered_categories = [
        k
        for k, v in sorted(
            category_count.items(), key=lambda item: item[1], reverse=True
        )
    ]
    category_to_category_id = {cat: i for i, cat in enumerate(ordered_categories)}

    category_ids = [category_to_category_id[cat] for cat in categories]
    _, bins = np.histogram(category_ids, bins=n_bins)

    category_to_bins = np.digitize(category_ids, bins)
    print(
        "Total categories: {}, val categories: {}".format(
            len(categories), len(val_categories)
        )
    )
    bin_to_categories = {i: [] for i in range(int(bins.max() + 1))}

    for category, bin in zip(categories, category_to_bins):
        bin_to_categories[bin].append(category)

    for bin in bin_to_categories:
        if len(bin_to_categories[bin]) == 0:
            continue
        val_categories.append(np.random.choice(bin_to_categories[bin]))

    train_categories = list(set(categories) - set(val_categories))
    val_categories = validate_val_unseen_categories(
        train_categories, val_categories, child_to_ancestor_map
    )

    print(train_categories)
    print(val_categories)
    print("Total train categories: {}".format(len(train_categories)))
    print("Total val categories: {}".format(len(val_categories)))

    df = pd.DataFrame(
        {
            "category": list(category_count.keys()),
            "instances": list(category_count.values()),
        }
    )
    df.sort_values(by="instances", inplace=True, ascending=False)
    print(df.columns)

    df.to_csv(os.path.join(output_path, "all_category_count.csv"), index=False)

    df = pd.DataFrame(
        {
            "category": list(val_category_count.keys()),
            "instances": list(val_category_count.values()),
        }
    )
    df.sort_values(by="instances", inplace=True, ascending=False)
    df.to_csv(os.path.join(output_path, "val_category_count.csv"), index=False)

    write_json(
        train_categories,
        os.path.join(output_path, "ovon_train_categories.json"),
    )
    write_json(val_categories, os.path.join(output_path, "ovon_val_categories.json"))
    write_json(
        {k: v for k, v in category_count.items() if k in train_categories},
        os.path.join(output_path, "ovon_train_category_count.json"),
    )
    write_json(
        {k: v for k, v in category_count.items() if k in val_categories},
        os.path.join(output_path, "ovon_val_category_count.json"),
    )


def filter_episodes(dataset, categories):
    filtered_episodes = []
    for episode in dataset["episodes"]:
        if episode["object_category"] not in categories:
            continue
        filtered_episodes.append(episode)

    return filtered_episodes, dataset["goals_by_category"]


def group_episodes_by_category(episodes):
    episodes_by_category = defaultdict(list)
    for episode in episodes:
        episodes_by_category[episode["object_category"]].append(episode)
    return episodes_by_category


def filter_and_save_dataset(
    path, output_path, categories, max_episodes=2000, split="train"
):
    files = glob.glob(os.path.join(path, "*.json.gz"))

    num_added = 0
    num_gz_files = len(files)
    sampled_categories = []

    for idx, file in tqdm(enumerate(files)):
        dataset = load_dataset(file)
        filtered_episodes, filtered_goals = filter_episodes(dataset, categories)
        # print("Pre and post filtering episodes: {}/{}".format(len(filtered_episodes), len(dataset["episodes"])))
        assert len(filtered_episodes) == len(
            dataset["episodes"]
        ), "Filtering should not change the number of episodes"

        if split == "val":
            num_left = max_episodes - num_added
            num_gz_remaining = num_gz_files - idx
            num_needed = min(num_left / num_gz_remaining, len(dataset["episodes"]))

            filtered_episodes = random.sample(filtered_episodes, int(num_needed))
            num_added += len(filtered_episodes)

        dataset["goals_by_category"] = filtered_goals
        dataset["episodes"] = filtered_episodes

        for ep in filtered_episodes:
            sampled_categories.append(ep["object_category"])

        write_dataset(
            dataset,
            os.path.join(output_path, os.path.basename(file)),
        )
    sampled_categories = set(sampled_categories)
    print(
        "Total episodes: {}, categories: {}".format(num_added, len(sampled_categories))
    )


def split_dataset(path, output_path, max_episodes=3000):
    categories = load_json("data/hm3d_meta/ovon_categories.json")

    # os.makedirs(os.path.join(output_path, "train/content"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "val_seen_filtered/content"), exist_ok=True)
    os.makedirs(
        os.path.join(output_path, "val_unseen_easy_filtered/content"), exist_ok=True
    )
    os.makedirs(
        os.path.join(output_path, "val_unseen_hard_filtered/content"), exist_ok=True
    )

    # filter_and_save_dataset(
    #     os.path.join(path, "train/content"),
    #     os.path.join(output_path, "train/content"),
    #     categories["train"],
    # )
    filter_and_save_dataset(
        os.path.join(path, "val_seen/content"),
        os.path.join(output_path, "val_seen_filtered/content"),
        categories["val_seen"],
        max_episodes,
        split="val",
    )
    filter_and_save_dataset(
        os.path.join(path, "val_unseen_easy/content"),
        os.path.join(output_path, "val_unseen_easy_filtered/content"),
        categories["val_unseen_easy"],
        max_episodes,
        split="val",
    )
    filter_and_save_dataset(
        os.path.join(path, "val_unseen_hard/content"),
        os.path.join(output_path, "val_unseen_hard_filtered/content"),
        categories["val_unseen_hard"],
        max_episodes,
        split="val",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--split-dataset", action="store_true", dest="split_dataset")
    args = parser.parse_args()

    if args.split_dataset:
        split_dataset(args.path, args.output_path)
    else:
        split_val_unseen_manual(args.path, args.output_path)
