import argparse
from collections import defaultdict

import clip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from ovon.utils.utils import load_json, write_json

PROMPT = "{category}"


def postprocess_meta(input_path, output_path):
    df = pd.read_csv(input_path)

    categories_by_region = defaultdict(list)
    for idx, row in df.iterrows():
        categories_by_region[row["Region Proposal"]].append(row["Category"])

    for k, v in categories_by_region.items():
        print(k, len(v))

    write_json(categories_by_region, output_path)


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


def semantic_failures(input_path):
    clip_m, preprocess = clip.load("RN50", device="cuda")

    records = load_json(input_path)

    failures = 0
    defaultdict(int)
    for k, v in records.items():
        failures += 1 - v["success"]
        if not v["success"]:
            nearest_objects = v["failure_modes.objects_within_2m"].split(",")
            category = v["target"]
            if category in nearest_objects:
                nearest_objects.remove(category)


def failure_metrics(input_path, output_path):
    records = load_json(input_path)

    failures = 0
    failure_modes = defaultdict(int)
    for k, v in records.items():
        failures += 1 - v["success"]
        if not v["success"]:
            for kk in v.keys():
                if kk in [
                    "failure_modes.recognition",
                    "failure_modes.exploration",
                    "failure_modes.last_mile_nav",
                    "failure_modes.stop_failure",
                ]:
                    failure_modes[kk] += v[kk]

    failure_modes = {k: v / failures for k, v in failure_modes.items()}
    labels = list(failure_modes.keys())
    metrics = list(failure_modes.values())

    colors = sns.color_palette("pastel")[0:5]

    # create pie chart
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.pie(metrics, labels=labels, colors=colors, autopct="%.0f%%")
    fig.savefig(output_path, bbox_inches="tight")

    print(failure_modes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, help="Path to the meta csv file", required=True
    )
    parser.add_argument(
        "--output-path", type=str, help="Path to the output file", required=True
    )
    args = parser.parse_args()
    # postprocess_meta(args.input_path, args.output_path)
    failure_metrics(args.input_path, args.output_path)
