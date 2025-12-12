import argparse
import glob
import gzip
import json
import os
import pickle

import clip
import numpy as np
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from tqdm import tqdm

# PROMPT = "{category}"
PROMPT = "Seems like there is a {category} ahead."


def save_to_disk(text_embedding, goal_categories, output_path):
    output = {}
    for goal_category, embedding in zip(goal_categories, text_embedding):
        output[goal_category] = embedding.detach().cpu().numpy()
    save_pickle(output, output_path)


def cache_embeddings(goal_categories, output_path, clip_model="RN50"):
    model, _ = clip.load(clip_model)
    batch = tokenize_and_batch(clip, goal_categories)

    with torch.no_grad():
        print(batch.shape)
        text_embedding = model.encode_text(batch.flatten(0, 1)).float()
    save_to_disk(text_embedding, goal_categories, output_path)


def tokenize_and_batch(clip, goal_categories):
    tokens = []
    for category in goal_categories:
        prompt = PROMPT.format(category=category)
        tokens.append(clip.tokenize(prompt, context_length=77).numpy())
    return torch.tensor(np.array(tokens)).cuda()


def tokenize_and_batch_siglip(
    goal_categories, model_name="hf-hub:timm/ViT-B-16-SigLIP-256"
):
    tokenizer = get_tokenizer(model_name)
    tokens = []
    for category in tqdm(goal_categories):
        prompt = category
        tokens.append(tokenizer([prompt], context_length=64).numpy())
    return torch.tensor(np.array(tokens))


def cache_embeddings_siglip(
    goal_categories, output_path, model_name="hf-hub:timm/ViT-B-16-SigLIP-256"
):
    model, _ = create_model_from_pretrained(model_name)
    batch = tokenize_and_batch_siglip(goal_categories, model_name)

    with torch.no_grad():
        print(batch.shape)
        text_embedding = model.encode_text(batch.flatten(0, 1)).float()
    save_to_disk(text_embedding, goal_categories, output_path)


def load_categories_from_dataset(path):
    files = glob.glob(os.path.join(path, "*json.gz"))

    categories = []
    for f in tqdm(files):
        dataset = load_dataset(f)
        for goal_key in dataset["goals_by_category"].keys():
            categories.append(goal_key.split("_")[1])
    return list(set(categories))


def main(dataset_path, output_path, use_siglip):
    goal_categories = load_categories_from_dataset(dataset_path)
    val_seen_categories = load_categories_from_dataset(
        dataset_path.replace("train", "val_seen")
    )
    val_unseen_easy_categories = load_categories_from_dataset(
        dataset_path.replace("train", "val_unseen_easy")
    )
    val_unseen_hard_categories = load_categories_from_dataset(
        dataset_path.replace("train", "val_unseen_hard")
    )

    # Print the first 5 categories of each split
    print("Total categories: {}".format(len(goal_categories)))
    print("First 5 categories:")
    print("goal_categories: {}".format(goal_categories[:5]))
    print("val_seen_categories: {}".format(val_seen_categories[:5]))
    print("val_unseen_easy_categories: {}".format(val_unseen_easy_categories[:5]))
    print("val_unseen_hard_categories: {}".format(val_unseen_hard_categories[:5]))

    goal_categories.extend(val_seen_categories)
    goal_categories.extend(val_unseen_easy_categories)
    goal_categories.extend(val_unseen_hard_categories)

    print("Total goal categories: {}".format(len(goal_categories)))
    print(
        "Train categories: {}, Val seen categories: {}, Val unseen easy categories: {},"
        " Val unseen hard categories: {}".format(
            len(goal_categories),
            len(val_seen_categories),
            len(val_unseen_easy_categories),
            len(val_unseen_hard_categories),
        )
    )
    if use_siglip:
        cache_embeddings_siglip(goal_categories, output_path)
    else:
        cache_embeddings(goal_categories, output_path)


def load_dataset(path):
    with gzip.open(path, "rt") as file:
        data = json.loads(file.read())
    return data


def save_pickle(data, path):
    with open(path, "wb") as file:
        pickle.dump(data, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="file path of OVON dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="output path of text embeddings",
    )
    parser.add_argument(
        "--use-siglip",
        action="store_true",
        help="use siglip model",
    )
    args = parser.parse_args()
    main(args.dataset_path, args.output_path, args.use_siglip)
