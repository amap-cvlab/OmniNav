from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from habitat.core.embodied_task import Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask, Success

from ovon.dataset.semantic_utils import ObjectCategoryMapping
from ovon.utils.utils import load_json, load_pickle
import habitat_sim
if TYPE_CHECKING:
    from omegaconf import DictConfig
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.utils.visualizations import fog_of_war, maps
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.tasks.utils import cartesian_to_polar

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
    from habitat_sim import RigidState
    from habitat_sim.physics import VelocityControl
except ImportError:
    pass

cv2 = try_cv2_import()

MAP_THICKNESS_SCALAR: int = 128
@registry.register_measure
class TopDownMapObj(Measure):
    r"""Top Down Map measure"""

    def __init__(
        self,
        sim: "HabitatSim",
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.map_padding
        self._step_count: Optional[int] = None
        self._map_resolution = config.map_resolution
        self._ind_x_min: Optional[int] = None
        self._ind_x_max: Optional[int] = None
        self._ind_y_min: Optional[int] = None
        self._ind_y_max: Optional[int] = None
        self._previous_xy_location: Optional[Tuple[int, int]] = None
        self._top_down_map: Optional[np.ndarray] = None
        self._shortest_path_points: Optional[List[Tuple[int, int]]] = None
        self.line_thickness = int(
            np.round(self._map_resolution * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "top_down_map"

    def get_original_map(self):
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=self._config.draw_border,
        )

        if self._config.fog_of_war.draw:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def _draw_goals_view_points(self, episode):
        if self._config.draw_view_points:
            for goal in episode.goals:
                if self._is_on_same_floor(goal):
                    try:
                        if goal.view_points is not None:
                            for view_point in goal.view_points:
                                self._draw_point(
                                    view_point.agent_state.position,
                                    maps.MAP_VIEW_POINT_INDICATOR,
                                )
                    except AttributeError:
                        pass

    def _draw_goals_positions(self, episode):
        if self._config.draw_goal_positions:

            for goal in episode.goals:
                if self._is_on_same_floor(goal): 
                    try:
                        self._draw_point(
                            goal.position, maps.MAP_TARGET_POINT_INDICATOR
                        )
                    except AttributeError:
                        pass

    def _draw_goals_aabb(self, episode): # not support yet
        if self._config.draw_goal_aabbs:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(
                        sem_scene.objects[object_id].id.split("_")[-1]
                    ) == int(
                        goal.object_id
                    ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = (
                        sem_scene.objects[object_id].aabb.sizes / 2.0
                    )
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                        if self._is_on_same_floor(center[1])
                    ]

                    map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            (
                                self._top_down_map.shape[0],
                                self._top_down_map.shape[1],
                            ),
                            sim=self._sim,
                        )
                        for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass
    
    def get_straight_shortest_path_points(self, position_a, position_bs):
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = position_a
        path.requested_ends = position_bs
        if self._sim.pathfinder.find_path(path):
            return path.points
        else:
            return []

    def _draw_shortest_path(
        self, episode: NavigationEpisode, agent_position: AgentState
    ):
        if self._config.draw_shortest_path:
            candidates_points = []
            for goal in episode.goals:
                if goal.view_points is not None:
                    for item in goal.view_points:
                        candidates_points.append(item.agent_state.position)
            _shortest_path_points = self.get_straight_shortest_path_points(agent_position, candidates_points)                         
            
            self._shortest_path_points = [
                maps.to_grid(
                    p[2],
                    p[0],
                    (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                    sim=self._sim,
                )
                for p in _shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

    def _is_on_same_floor(
        self, goal, ref_floor_height=None, ceiling_height=2.0, relax_margin = 0.5
    ):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]

        if goal.view_points is not None:
            for item in goal.view_points:
                if ref_floor_height - relax_margin <= item.agent_state.position[1] < ref_floor_height + ceiling_height:
                    return True
        return False

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        if hasattr(episode, "goals"):
            # draw source and target parts last to avoid overlap
            self._draw_goals_view_points(episode)
            self._draw_goals_aabb(episode)
            self._draw_goals_positions(episode)
            self._draw_shortest_path(episode, agent_position)

        if self._config.draw_source:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )
        self.update_metric(episode, None, *args, **kwargs)
        
    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": (map_agent_x, map_agent_y),
            "agent_angle": self.get_polar_angle(),
        }

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.max_episode_steps, 245
            )

            thickness = self.line_thickness
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        if self._config.fog_of_war.draw:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(),
                fov=self._config.fog_of_war.fov,
                max_line_len=self._config.fog_of_war.visibility_dist
                / maps.calculate_meters_per_pixel(
                    self._map_resolution, sim=self._sim
                ),
            )



@registry.register_measure
class OVONDistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "distance_to_goal"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[List[Tuple[float, float, float]]] = None
        self._episode_goal_points: Optional[List[Tuple[float, float, float]]] = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        if self._config.distance_to == "VIEW_POINTS":
            goals = task._dataset.goals_by_category[episode.goals_key]
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in goals
                for view_point in goal.view_points
            ]
            self._episode_goal_points = [goal.position for goal in goals]
            if episode.children_object_categories is not None:
                for children_category in episode.children_object_categories:
                    scene_id = episode.scene_id.split("/")[-1]
                    goal_key = f"{scene_id}_{children_category}"

                    # Ignore if there are no valid viewpoints for goal
                    if goal_key not in task._dataset.goals_by_category:
                        continue
                    self._episode_goal_points.extend(
                        [goal.position for goal in task._dataset.goals_by_category[goal_key]]
                    )
                    self._episode_view_points.extend(
                        [
                            vp.agent_state.position
                            for goal in task._dataset.goals_by_category[goal_key]
                            for vp in goal.view_points
                        ]
                    )

        self.update_metric(episode=episode, task=task, *args, **kwargs)


    def _is_on_same_floor(
        self, height, ref_floor_height=None, ceiling_height=2.0, relax_margin = 0.5
    ):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height - relax_margin <= height < ref_floor_height + ceiling_height 

    def update_metric(
        self,
        episode: NavigationEpisode,
        task: NavigationTask,
        *args: Any,
        **kwargs: Any,
    ):
        current_position = self._sim.get_agent_state().position

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._config.distance_to == "POINT":
                goals = task._dataset.goals_by_category[episode.goals_key]
                distance_to_target = self._sim.geodesic_distance(
                    current_position,
                    [goal.position for goal in goals],
                    episode,
                )
            elif self._config.distance_to == "VIEW_POINTS":
                distance_to_target = self._sim.geodesic_distance(
                    current_position, self._episode_view_points, episode
                )
                # 修复部分缺少viewpoint问题
                for goal_point in self._episode_goal_points:
                    if self._is_on_same_floor(goal_point[1]): 
                        distance_to_target = min(distance_to_target, np.linalg.norm(current_position[[0,2]] - np.array(goal_point)[[0,2]]))

            else:
                logger.error(
                    "Non valid distance_to parameter was provided"
                    f"{self._config.distance_to}"
                )

            self._previous_position = (
                current_position[0],
                current_position[1],
                current_position[2],
            )
            self._metric = distance_to_target


@registry.register_measure
class OVONObjectGoalID(Measure):
    cls_uuid: str = "object_goal_id"

    def __init__(self, config: "DictConfig", *args: Any, **kwargs: Any):
        cache = load_pickle(config.cache)
        self.vocab = sorted(list(cache.keys()))
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = self.vocab.index(episode.object_category)

    def update_metric(
        self,
        episode: NavigationEpisode,
        task: NavigationTask,
        *args: Any,
        **kwargs: Any,
    ):
        pass


@registry.register_measure
class FailureModeMeasure(Measure):
    """
    Last Mile Navigation failure measures.
    """

    cls_uuid: str = "failure_modes"

    def __init__(self, config: "DictConfig", *args: Any, **kwargs: Any):
        self._config = config
        self._goal_seen = False
        self._elapsed_steps = 0
        self._ovon_categories = load_json(config.categories_file)
        self.cat_map = ObjectCategoryMapping(
            config.mapping_file,
            coverage_meta_file="data/coverage_meta/train.pkl",
            frame_coverage_threshold=0.05,
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, observations, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [OVONDistanceToGoal.cls_uuid, Success.cls_uuid]
        )
        self._goal_seen = False
        self._max_area = 0
        self._elapsed_steps = 0
        self._reached_within_success_area = False
        self.update_metric(episode=episode, task=task, observations=observations, *args, **kwargs)  # type: ignore

    def visible_goal_area(self, observations, episode, task):
        scene_id = episode.scene_id.split("/")[-1]
        object_category = episode.object_category
        goal_key = f"{scene_id}_{object_category}"
        goals = task._dataset.goals_by_category[goal_key]
        object_ids = [g.object_id for g in goals]
        semantic_scene = task._sim.semantic_annotations()
        objs = [o for o in semantic_scene.objects if o.id in object_ids]

        semantic_observation = observations["semantic"]
        mask = np.zeros_like(semantic_observation)
        for obj in objs:
            mask += (semantic_observation == obj.semantic_id).astype(np.int32)
        area = np.sum(mask) / np.prod(semantic_observation.shape)
        return area

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: NavigationTask, observations, *args: Any, **kwargs: Any
    ):
        try:
            area = self.visible_goal_area(observations, episode, task)
            self._max_area = max(self._max_area, area)
            if area >= 0.01:
                self._goal_seen = True

            distance_to_target = task.measurements.measures[
                OVONDistanceToGoal.cls_uuid
            ].get_metric()
            is_success = task.measurements.measures[Success.cls_uuid].get_metric()

            if distance_to_target < 0.25:
                self._reached_within_success_area = True

            metrics = {
                "stop_too_far": False,
                "stop_failure": False,
                "recognition_failure": False,
                "misidentification": False,
                "exploration": False,
            }

            metrics["area_seen"] = self._max_area
            if not is_success:
                if self._goal_seen:
                    if task.is_stop_called:
                        metrics["stop_too_far"] = True
                    else:
                        metrics["stop_failure"] = self._reached_within_success_area
                        metrics["recognition_failure"] = (
                            not self._reached_within_success_area
                        )
                else:
                    if task.is_stop_called:
                        metrics["misidentification"] = True
                    else:
                        metrics["exploration"] = True
            metrics["num_steps"] = self._elapsed_steps

            self._elapsed_steps += 1
            self._metric = metrics
        except Exception as e:
            print("Error ", e)
