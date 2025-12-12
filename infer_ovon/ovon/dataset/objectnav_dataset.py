#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import attr
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.utils import DatasetFloatJSONEncoder, not_none_validator
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True, kw_only=True)
class AgentStateSpec:
    r"""Agent data specifications that capture states of agent and sensor in replay state."""

    position: Optional[List[float]] = attr.ib(default=None)
    rotation: Optional[List[float]] = attr.ib(default=None)
    sensor_data: Optional[dict] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ReplayActionSpec:
    r"""Replay specifications that capture metadata associated with action."""

    action: str = attr.ib(default=None, validator=not_none_validator)
    agent_state: Optional[AgentStateSpec] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode
    :param object_category: Category of the obect
    """

    object_category: Optional[str] = None
    reference_replay: Optional[List[ReplayActionSpec]] = None
    scene_state = None
    is_thda: Optional[bool] = False
    scene_dataset: Optional[str] = "hm3d"
    scene_dataset_config: Optional[str] = ""
    additional_obj_config_paths: Optional[List] = []
    attempts: Optional[int] = 1

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@registry.register_dataset(name="ObjectNav-v2")
class ObjectNavDatasetV2(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Navigation dataset."""

    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[ObjectGoalNavEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]
    gibson_to_mp3d_category_map: Dict[str, str] = {
        "couch": "sofa",
        "toilet": "toilet",
        "bed": "bed",
        "tv": "tv_monitor",
        "potted plant": "plant",
        "chair": "chair",
    }
    max_episode_steps: int = 500

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = {}
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = ObjectGoalNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals_by_category[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.goals_by_category = {}
        self.max_episode_steps = 500
        super().__init__(config)
        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_scene_annotation_category_id"
            ]

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        assert len(self.category_to_task_category_id) == len(
            self.category_to_scene_annotation_category_id
        )

        assert set(self.category_to_task_category_id.keys()) == set(
            self.category_to_scene_annotation_category_id.keys()
        ), "category_to_task and category_to_mp3d must have the same keys"

        if len(deserialized["episodes"]) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        for i, episode in enumerate(deserialized["episodes"]):
            if "_shortest_path_cache" in episode:
                del episode["_shortest_path_cache"]

            if "scene_state" in episode:
                del episode["scene_state"]

            episode = ObjectGoalNavEpisode(**episode)
            episode.episode_id = str(i)
            episode.start_position = list(map(float, episode.start_position))
            episode.start_rotation = list(map(float, episode.start_rotation))

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = self.goals_by_category[episode.goals_key]

            if episode.reference_replay is not None:
                for i, replay_step in enumerate(episode.reference_replay):
                    replay_step["agent_state"] = None
                    episode.reference_replay[i] = ReplayActionSpec(**replay_step)

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)

            if (
                episode.reference_replay is not None
                and len(episode.reference_replay) > self.max_episode_steps
            ):
                continue

            self.episodes.append(episode)  # type: ignore [attr-defined]
