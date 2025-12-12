import hashlib
import random
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from gym import spaces
from habitat.core.registry import registry
from habitat.core.simulator import RGBSensor, Sensor, SensorTypes, Simulator
from habitat.core.utils import try_cv2_import
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import NavigationEpisode

from ovon.utils.utils import load_pickle

if TYPE_CHECKING:
    from omegaconf import DictConfig

cv2 = try_cv2_import()


@registry.register_sensor
class ClipObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id, and we will generate the prompt corresponding to it
    so that it's usable by CLIP's text encoder.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """

    cls_uuid: str = "clip_objectgoal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.cache = load_pickle(config.cache)
        k = list(self.cache.keys())[0]
        self._embed_dim = self.cache[k].shape[0]
        for v in self.cache.values():
            assert self._embed_dim == v.shape[0] and v.ndim == 1
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._embed_dim,), dtype=np.float32
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        **kwargs: Any,
    ) -> Optional[int]:
        category = (
            episode.object_category if hasattr(episode, "object_category") else ""
        )
        if category not in self.cache:
            print("Missing category: {}".format(category))
        return self.cache[category]


@registry.register_sensor
class ClipImageGoalSensor(Sensor):
    cls_uuid: str = "clip_imagegoal"

    def __init__(
        self,
        sim: HabitatSim,
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid for uuid, sensor in sensors.items() if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                "ImageGoalNav requires one RGB sensor,"
                f" {len(rgb_sensor_uuids)} detected"
            )
        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        super().__init__(config=config)
        self._curr_ep_id = None
        self.image_goal = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[self._rgb_sensor_uuid]

    def _reset(self, episode):
        self._curr_ep_id = episode.episode_id
        sampled_goal = random.choice(episode.goals)
        sampled_viewpoint = random.choice(sampled_goal.view_points)
        observations = self._sim.get_observations_at(
            position=sampled_viewpoint.agent_state.position,
            rotation=sampled_viewpoint.agent_state.rotation,
            keep_agent_at_new_pose=False,
        )
        assert observations is not None
        self.image_goal = observations["rgb"]
        # Mutate the episode
        episode.goals = [sampled_goal]

    def get_observation(
        self,
        observations,
        episode: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.image_goal is None or self._curr_ep_id != episode.episode_id:
            self._reset(episode)
        assert self.image_goal is not None
        return self.image_goal


@registry.register_sensor
class ClipGoalSelectorSensor(Sensor):
    cls_uuid: str = "clip_goal_selector"

    def __init__(
        self,
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(config=config)
        self._image_sampling_prob = config.image_sampling_probability
        self._curr_ep_id = None
        self._use_image_goal = True

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.bool,
        )

    def _reset(self, episode):
        self._curr_ep_id = episode.episode_id
        self._use_image_goal = random.random() < self._image_sampling_prob

    def get_observation(
        self,
        observations,
        episode: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if self._curr_ep_id != episode.episode_id:
            self._reset(episode)
        return np.array([self._use_image_goal], dtype=np.bool)


@registry.register_sensor
class ImageGoalRotationSensor(Sensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.
    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """

    cls_uuid: str = "image_goal_rotation"

    def __init__(self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid for uuid, sensor in sensors.items() if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                "ImageGoalNav requires one RGB sensor,"
                f" {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[self._rgb_sensor_uuid]

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        # Add rotation to episode
        if self.config.sample_angle:
            angle = np.random.uniform(0, 2 * np.pi)
        else:
            # to be sure that the rotation is the same for the same episode_id
            # since the task is currently using pointnav Dataset.
            seed = abs(hash(episode.episode_id)) % (2**32)
            rng = np.random.RandomState(seed)
            angle = rng.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        episode.goals[0].rotation = source_rotation

        goal_observation = self._sim.get_observations_at(
            position=goal_position.tolist(), rotation=source_rotation
        )
        return goal_observation[self._rgb_sensor_uuid]

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(episode)
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor
class CurrentEpisodeUUIDSensor(Sensor):
    r"""Sensor for current episode uuid observations.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """

    cls_uuid: str = "current_episode_uuid"

    def __init__(self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any):
        self._sim = sim
        self._current_episode_id: Optional[str] = None

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.int64).min,
            high=np.iinfo(np.int64).max,
            shape=(1,),
            dtype=np.int64,
        )

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        episode_uuid = (
            int(hashlib.sha1(episode_uniq_id.encode("utf-8")).hexdigest(), 16) % 10**8
        )
        return episode_uuid


@registry.register_sensor
class StepIDSensor(Sensor):
    cls_uuid: str = "step_id"
    curr_ep_id: str = ""
    _elapsed_steps: int = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.int64).min,
            high=np.iinfo(np.int64).max,
            shape=(1,),
            dtype=np.int64,
        )

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if self.curr_ep_id != episode_uniq_id:
            self.curr_ep_id = episode_uniq_id
            self._elapsed_steps = 0
        else:
            self._elapsed_steps += 1
        return self._elapsed_steps
