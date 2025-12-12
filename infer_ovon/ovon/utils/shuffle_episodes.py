import argparse
import glob
import os
from collections import defaultdict

from ovon.utils.utils import load_dataset, load_json, write_dataset


def shuffle_episodes(
    dataset_path: str,
    output_path: str,
    meta_path: str = "data/hm3d_meta/val_splits.json",
):
    category_per_splits = load_json(meta_path)
    splits = list(category_per_splits.keys())
    print(splits)

    scenes = glob.glob(os.path.join(dataset_path, splits[0], "content/*json.gz"))

    # make directories
    for split in splits:
        os.makedirs(os.path.join(output_path, split, "content"), exist_ok=True)

    for scene in scenes:
        scene_id = scene.split("/")[-1]
        scene_name = scene_id.split(".")[0]

        goals_by_category = {}
        episodes_by_category = defaultdict(list)
        for split in splits:
            path = os.path.join(dataset_path, split, "content", scene_id)
            dataset = load_dataset(path)

            for goal_key, goal in dataset["goals_by_category"].items():
                goals_by_category[goal_key] = goal

            for episode in dataset["episodes"]:
                episodes_by_category[episode["object_category"]].append(episode)

        for split in splits:
            path = os.path.join(dataset_path, split, "content", scene_id)
            dataset = load_dataset(path)

            goals_before = len(dataset["goals_by_category"].keys())
            episodes_before = len(dataset["episodes"])

            dataset["goals_by_category"] = {}
            dataset["episodes"] = []

            for key in category_per_splits[split]:
                g_key = "{}.basis.glb_{}".format(scene_name, key)
                if goals_by_category.get(g_key) is None:
                    continue
                dataset["goals_by_category"][g_key] = goals_by_category[g_key]
                dataset["episodes"].extend(episodes_by_category[key])
            print(
                "Split: {}, # of goals: {}/{}, # of episodes: {}/{}".format(
                    split,
                    len(dataset["goals_by_category"].keys()),
                    goals_before,
                    len(dataset["episodes"]),
                    episodes_before,
                )
            )

            op = os.path.join(output_path, split, "content", scene_id)
            print("Output: {}".format(op))
            write_dataset(dataset, op)

    print("\n")
    for split in splits:
        files = glob.glob(os.path.join(output_path, split, "content/*json.gz"))

        goals = []
        episodes = 0
        for f in files:
            dataset = load_dataset(f)
            episodes += len(dataset["episodes"])

            goal_keys = [k.split("_")[-1] for k in dataset["goals_by_category"].keys()]
            goals.extend(goal_keys)

        diff = set(category_per_splits[split]).difference(set(goals))
        print(
            "Validating Split: {}, # of goals: {}, # of episodes: {}, Difference: {}"
            .format(split, len(set(goals)), episodes, len(diff))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default="data/hm3d_meta/val_splits.json"
    )
    parser.add_argument(
        "--output_path", type=str, default="data/hm3d_meta/val_splits.json"
    )

    args = parser.parse_args()

    shuffle_episodes(args.dataset_path, args.output_path)
