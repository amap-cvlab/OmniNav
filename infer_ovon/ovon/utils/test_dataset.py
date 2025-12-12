import argparse
import glob
import os

from ovon.utils.utils import count_episodes, load_dataset


def test_dataset(path):
    print(path)
    files = glob.glob(os.path.join(path, "*.json.gz"))
    dataset = load_dataset(files[0])

    print("Total # of episodes: {}".format(count_episodes(dataset)))

    for ep in dataset["episodes"]:
        if len(ep["children_object_categories"]) > 0:
            print("Found episode with children object categories")
            break

    print("Object goal: {}".format(ep["object_category"]))
    for children in ep["children_object_categories"]:
        print(children)
        scene_id = ep["scene_id"].split("/")[-1]
        goal_key = f"{scene_id}_{children}"

        # Ignore if there are no valid viewpoints for goal
        if goal_key not in dataset["goals_by_category"]:
            print("No valid viewpoints for child: {}".format(children))
            continue
        print(
            "Viewpoints: {} for child: {}".format(
                len(dataset["goals_by_category"][goal_key]), children
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/ovon_dataset.json")
    args = parser.parse_args()

    test_dataset(args.path)
