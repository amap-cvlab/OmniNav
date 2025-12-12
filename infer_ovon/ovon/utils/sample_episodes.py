import argparse
import glob
import os
import os.path as osp
import random
from collections import defaultdict

from tqdm import tqdm

from ovon.utils.utils import count_episodes, load_dataset, load_json, write_dataset


def sample_val(input_path, output_path, max_episodes):
    files = glob.glob(osp.join(input_path, "*.json.gz"))
    len(files)

    count, categories = count_episodes(input_path)
    print("Total episodes: {}".format(count))

    max_epsiodes_per_category = max_episodes // len(categories)
    print("Max episodes per category: {}".format(max_epsiodes_per_category))

    episodes_per_category = defaultdict(list)
    for f in tqdm(files):
        dataset = load_dataset(f)
        for episode in dataset["episodes"]:
            episodes_per_category[episode["object_category"]].append(episode)

    sampled_buffer = []
    episodes_per_scene = defaultdict(list)
    total_episodes = 0
    for category, episodes in episodes_per_category.items():
        random.shuffle(episodes)
        episodes_per_category[category] = episodes[:max_epsiodes_per_category]
        sampled_buffer.extend(episodes[max_epsiodes_per_category + 1 :])
        total_episodes += len(episodes_per_category[category])
        for episode in episodes_per_category[category]:
            episodes_per_scene[episode["scene_id"]].append(episode)

    if total_episodes != max_episodes:
        missing_episodes = max_episodes - total_episodes
        sampled_episodes = random.sample(sampled_buffer, missing_episodes)
        for episode in sampled_episodes:
            episodes_per_scene[episode["scene_id"]].append(episode)

    num_added = 0
    for idx, file in enumerate(tqdm(files)):
        dataset = load_dataset(file)
        scene_id = dataset["episodes"][0]["scene_id"]

        dataset["episodes"] = episodes_per_scene[scene_id]
        num_added += len(dataset["episodes"])

        output_file = osp.join(output_path, osp.basename(file))
        print(f"Copied {len(dataset['episodes'])} episodes to {output_file}!")
        write_dataset(dataset, output_file)

    print(f"Added {num_added} episodes in total!")


def sample_custom(input_path, output_path, episode_meta_file):
    files = glob.glob(osp.join(input_path, "*.json.gz"))

    episode_meta = load_json(episode_meta_file)

    os.makedirs(output_path, exist_ok=True)

    num_added = 0
    eps_per_scene = []
    for idx, file in enumerate(tqdm(files)):
        dataset = load_dataset(file)
        scene_id = file.split("/")[-1].split(".")[0]
        print(file, scene_id)

        episodes_by_category = defaultdict(list)
        for episode in dataset["episodes"]:
            episodes_by_category[episode["object_category"]].append(episode)

        dataset["episodes"] = []
        for category in episode_meta[scene_id]:
            min_episodes = min(
                episode_meta[scene_id][category], len(episodes_by_category[category])
            )
            sampled_episodes = random.sample(
                episodes_by_category[category], min_episodes
            )
            if min_episodes < episode_meta[scene_id][category]:
                print(f"Warning: not enough episodes for {scene_id} {category}!")
            dataset["episodes"].extend(sampled_episodes)

        output_file = osp.join(output_path, osp.basename(file))
        print(f"Copied {len(dataset['episodes'])} episodes to {output_file}!")
        write_dataset(dataset, output_file)
        num_added += len(dataset["episodes"])

    print(f"Added {num_added} episodes in total!")
    print(f"Episodes per scene: {eps_per_scene}")


def main(input_path, output_path, max_episodes):
    files = glob.glob(osp.join(input_path, "*.json.gz"))
    num_gz_files = len(files)

    os.makedirs(output_path, exist_ok=True)

    num_added = 0
    eps_per_scene = []
    for idx, file in enumerate(tqdm(files)):
        dataset = load_dataset(file)
        random.shuffle(dataset["episodes"])

        num_left = max_episodes - num_added
        num_gz_remaining = num_gz_files - idx
        num_needed = min(num_left / num_gz_remaining, len(dataset["episodes"]))
        eps_per_scene.append(num_needed)

        sampled_episodes = random.sample(dataset["episodes"], int(num_needed))
        num_added += len(sampled_episodes)

        dataset["episodes"] = sampled_episodes

        output_file = osp.join(output_path, osp.basename(file))
        print(f"Copied {len(sampled_episodes)} episodes to {output_file}!")
        write_dataset(dataset, output_file)

    print(f"Added {num_added} episodes in total!")
    print(f"Episodes per scene: {eps_per_scene}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        help="Path to episode dir containing content/",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to episode dir containing content/",
    )
    parser.add_argument(
        "--episode-meta-file",
        type=str,
        help="Path to num episode per category per scene meta file",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--val", dest="is_val", action="store_true")
    args = parser.parse_args()

    if args.episode_meta_file is not None:
        sample_custom(args.input_path, args.output_path, args.episode_meta_file)
    elif args.is_val:
        sample_val(args.input_path, args.output_path, args.max_episodes)
    else:
        main(args.input_path, args.output_path, args.max_episodes)
