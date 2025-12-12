import csv
import os
import pickle
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

from ovon.utils.utils import load_json


class Relationship:
    scene: str
    relation: str
    ref_object: str
    target_object: str
    ref_obj_semantic_id: int
    target_obj_semantic_id: int
    distance: float
    cov: float
    area: float
    img_ref: str
    name: str
    bb: np.ndarray


class ObjectCategoryMapping(MutableMapping):
    _mapping: Dict[str, str]

    def __init__(
        self,
        mapping_file: str,
        allowed_categories: Optional[Set[str]] = None,
        coverage_meta_file: Optional[str] = None,
        frame_coverage_threshold: Optional[float] = None,
        blacklist_file: Optional[str] = "data/hm3d_meta/blacklist.txt",
    ) -> None:
        self._mapping = self.limit_mapping(
            self.load_categories(
                mapping_file, coverage_meta_file, frame_coverage_threshold
            ),
            allowed_categories,
            blacklist_file,
        )

    @staticmethod
    def load_categories(
        mapping_file: str,
        coverage_meta_file: str,
        frame_coverage_threshold: float,
        filter_attributes: Optional[Set[str]] = [
            "ceiling",
            "door",
            "floor",
            "object ",
            "wall",
            "unknown",
            "device",
            "decoration",
        ],
    ) -> Dict[str, str]:
        # Filter based on coverage
        file = open(coverage_meta_file, "rb")
        coverage_metadata = pickle.load(file)

        coverage_metadata_dict = defaultdict(list)
        for category, coverage_meta in coverage_metadata.items():
            for frame_coverage, _, scene in coverage_meta:
                if frame_coverage >= frame_coverage_threshold:
                    coverage_metadata_dict[category].append(frame_coverage)

        mapping = {}
        threshold_filtering = 0
        attr_filtering = 0
        with open(mapping_file, "r") as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter="\t")
            is_first_row = True
            for row in tsv_reader:
                if is_first_row:
                    is_first_row = False
                    continue
                raw_name = row[1]
                raw_cat_name = row[2]

                ignore_category = False
                for attribute in filter_attributes:
                    if attribute in raw_cat_name.lower():
                        ignore_category = True
                        attr_filtering += 1
                        break

                if len(coverage_metadata_dict[raw_name]) < 1:
                    threshold_filtering += 1
                    ignore_category = True

                if "otherroom" in raw_cat_name.lower():
                    raw_cat_name = raw_cat_name.split("/")[0].strip()

                if ignore_category:
                    continue
                mapping[raw_name] = raw_cat_name

        print(
            "Post filtering stats - Threshold filtering: {}, Ignore category: {},"
            " Final: {}".format(
                threshold_filtering, attr_filtering, len(mapping.keys())
            )
        )

        return mapping

    @staticmethod
    def limit_mapping(
        mapping: Dict[str, str],
        allowed_categories: Optional[Set[str]] = None,
        blacklist_file: Optional[str] = None,
    ) -> Dict[str, str]:
        if blacklist_file is not None:
            blacklist = [line.strip() for line in open(blacklist_file, "r").readlines()]
            mapping = {k: v for k, v in mapping.items() if k not in blacklist}

        if allowed_categories is None:
            return mapping

        return {k: v for k, v in mapping.items() if v in allowed_categories}

    def get_categories(self):
        return set(self._mapping.values())

    def __getitem__(self, key: str):
        k = self._keytransform(key)
        if k in self._mapping:
            return self._mapping[k]
        return None

    def __setitem__(self, key: str, value: str):
        self._mapping[self._keytransform(key)] = value

    def __delitem__(self, key: str):
        del self._mapping[self._keytransform(key)]

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def _keytransform(self, key: str):
        return key.lower()


class WordnetMapping(MutableMapping):
    _mapping: Dict[str, str]

    def __init__(
        self,
        mapping_file: str,
        allowed_categories: Optional[Set[str]] = None,
    ) -> None:
        self._mapping = self.limit_mapping(
            self.load_categories(mapping_file),
            allowed_categories,
        )

    @staticmethod
    def load_categories(mapping_file: str) -> Dict[str, str]:
        if mapping_file is None:
            return {}
        wordnet_mapping = load_json(mapping_file)
        return wordnet_mapping

    @staticmethod
    def limit_mapping(
        mapping: Dict[str, str], allowed_categories: Optional[Set[str]] = None
    ) -> Dict[str, str]:
        if allowed_categories is None:
            return mapping
        return {k: v for k, v in mapping.items() if v in allowed_categories}

    def get_categories(self):
        return set(self._mapping.values())

    def __getitem__(self, key: str):
        k = self._keytransform(key)
        if k in self._mapping:
            return self._mapping[k]
        return None

    def __setitem__(self, key: str, value: str):
        self._mapping[self._keytransform(key)] = value

    def __delitem__(self, key: str):
        del self._mapping[self._keytransform(key)]

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def _keytransform(self, key: str):
        return key.lower()


class SceneRelationshipsMapping(MutableMapping):
    _mapping: Dict[str, Dict[str, List[Relationship]]]

    def __init__(
        self,
        mapping_file: str,
    ) -> None:
        self._mapping = self.load_relationships(mapping_file)

    @staticmethod
    def load_relationships(
        mapping_file: str,
    ) -> Dict[str, Dict[str, Relationship]]:
        file = open(mapping_file, "rb")
        mapping = pickle.load(file)

        new_mapping = {}
        for scene in mapping.keys():
            new_mapping[scene] = {
                reln["name"]: [i for i in mapping[scene] if i["name"] == reln["name"]]
                for reln in mapping[scene]
            }

        return new_mapping

    def __getitem__(self, key: str):
        if key in self._mapping:
            return self._mapping[key]
        return None

    def __setitem__(self, key: str, value: str):
        self._mapping[key] = value

    def __delitem__(self, key: str):
        del self._mapping[key]

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)


def get_hm3d_semantic_scenes(
    hm3d_dataset_dir: str, splits: Optional[Iterable[str]] = None
) -> Dict[str, Set[str]]:
    if splits is None:
        splits = ["train", "minival", "val"]

    def include_scene(s):
        if not os.path.isdir(s):
            return False
        return len([f for f in os.listdir(s) if "semantic" in f]) > 0

    def get_basis_file(s):
        return [x for x in os.listdir(s) if x.endswith("basis.glb")][0]

    semantic_scenes = {}  # split -> scene file path
    for split in splits:
        split_dir = os.path.join(hm3d_dataset_dir, split)
        all_scenes = [os.path.join(split_dir, s) for s in os.listdir(split_dir)]
        all_scenes = [s for s in all_scenes if include_scene(s)]
        scene_paths = {os.path.join(s, get_basis_file(s)) for s in all_scenes}
        semantic_scenes[split] = scene_paths

    return semantic_scenes


if __name__ == "__main__":
    cat_map = ObjectCategoryMapping(
        mapping_file="data/Mp3d_category_mapping_updated.tsv",
        allowed_categories={
            "chair",
            "bed",
            "toilet",
            "sofa",
            "plant",
            "tv_monitor",
        },
    )
    print(cat_map.get_categories())
    print("category of `armchair`:", cat_map["armchair"])

    s = get_hm3d_semantic_scenes(
        "/nethome/akutumbaka3/files/ovonproject/data/scene_data/hm3d"
    )
    print(s["minival"])
