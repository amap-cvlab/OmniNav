from typing import List, Optional, Tuple

import habitat_sim
import numpy as np
from habitat_sim._ext.habitat_sim_bindings import SemanticObject
from habitat_sim.agent.agent import AgentState, SixDOFPose
from habitat_sim.simulator import Simulator
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_two_vectors
from numpy import ndarray

EPS_ARRAY = np.array([1e-5, 0.0, 1e-5])


class PoseSampler:
    sim: Simulator
    radius_min: float
    radius_max: float
    radius_step: float
    rot_deg_delta: float
    height_min: float
    height_max: float
    max_vdelta: float
    max_hdelta: float
    pitch_bounds: Tuple[float]
    do_sample_height: bool
    sample_lookat_deg_delta: bool

    def __init__(
        self,
        sim: Simulator,
        r_min: float,
        r_max: float,
        r_step: float,
        rot_deg_delta: float,
        max_vdelta: float = 0.25,
        max_hdelta: float = 0.001,
        h_min: Optional[float] = None,
        h_max: Optional[float] = None,
        pitch_bounds: Tuple[float] = (-np.inf, np.inf),
        sample_lookat_deg_delta: Optional[float] = None,
        np_rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.sim = sim
        self.radius_min = r_min
        self.radius_max = r_max
        self.radius_step = r_step
        self.rot_deg_delta = rot_deg_delta
        self.height_min = h_min
        self.height_max = h_max
        self.max_vdelta = max_vdelta
        self.max_hdelta = max_hdelta
        self.pitch_bounds = pitch_bounds
        self.do_sample_height = h_max is not None and h_min is not None
        self.sample_lookat_deg_delta = sample_lookat_deg_delta
        if np_rng is None:
            np_rng = np.random.default_rng(4)
        self.np_rng = np_rng

    def _get_floor_height(self, search_center: np.ndarray) -> float:
        """Floor height estimation: snap the bbox centroid to the navmesh"""
        point = np.asarray(search_center)[:, None]
        snapped = self.sim.pathfinder.snap_point(point)

        # the centroid should not be lower than the floor
        tries = 0
        while point[1, 0] < snapped[1]:
            point[1, 0] -= 0.05
            snapped = self.sim.pathfinder.snap_point(point)
            tries += 1
            if tries > 40:  # trace 2.0m down.
                break

        return snapped[1]

    def _vsnap_and_validate(self, p: ndarray) -> Tuple[ndarray, bool]:
        """Snap to the navmesh, check horizontal & vertical displacements."""
        snapped = np.asarray(self.sim.pathfinder.snap_point(p))

        is_navigable = self.sim.pathfinder.is_navigable(snapped)
        h_delta = np.linalg.norm(np.array([p[0] - snapped[0], p[2] - snapped[2]]))
        v_delta = abs(p[1] - snapped[1])

        # three conditions:
        valid = is_navigable and v_delta < self.max_vdelta and h_delta < self.max_hdelta

        return snapped, valid

    def sample_agent_poses_radially(
        self,
        search_center: np.ndarray = None,
        obj: SemanticObject = None,
    ) -> List[AgentState]:
        """Generates AgentState.position and AgentState.rotation for all
        navigable agent poses given a radial sampling method about the
        search_center.
        """

        floor_height = self._get_floor_height(search_center)
        search_center = np.array([search_center[0], floor_height, search_center[2]])

        poses: List[AgentState] = []

        for i in range(
            1 + int(np.floor((self.radius_max - self.radius_min) / self.radius_step))
        ):
            r = self.radius_min + i * self.radius_step
            for j in range(1 + int(np.floor(360 / self.rot_deg_delta))):
                theta = np.deg2rad(j * self.rot_deg_delta)
                x_diff = r * np.cos(theta)
                z_diff = r * np.sin(theta)

                pos, valid = self._vsnap_and_validate(
                    np.array(
                        [
                            search_center[0] + x_diff,
                            search_center[1],
                            search_center[2] + z_diff,
                        ]
                    )
                )
                if not valid:
                    continue

                # face the object
                cam_normal = (search_center - pos) + EPS_ARRAY
                cam_normal[1] = 0
                cam_normal = cam_normal / np.linalg.norm(cam_normal)
                rot = quat_from_two_vectors(habitat_sim.geo.FRONT, cam_normal)

                # sample a left-right turn angle
                if self.sample_lookat_deg_delta is not None:
                    deg = self.np_rng.uniform(
                        -self.sample_lookat_deg_delta,
                        self.sample_lookat_deg_delta,
                    )
                    rot = rot * quat_from_angle_axis(
                        np.deg2rad(deg), habitat_sim.geo.GRAVITY
                    )

                poses.append(AgentState(position=pos, rotation=rot))

        return poses

    def sample_camera_poses(
        self, agent_states: List[AgentState], obj: SemanticObject
    ) -> List[AgentState]:
        """Generates agent states that have the existing agent
        position+rotation with new sensor states.
        """
        new_states = []
        for agent_state in agent_states:
            # infer sensor states
            self.sim.agents[0].set_state(agent_state)
            ss = self.sim.agents[0].get_state().sensor_states

            if self.do_sample_height:
                h = agent_state.position[1] + self.np_rng.uniform(
                    self.height_min, self.height_max
                )
                ss = {
                    k: SixDOFPose(
                        position=np.array([p.position[0], h, p.position[2]]),
                        rotation=p.rotation,
                    )
                    for k, p in ss.items()
                }

            p = ss["color_sensor"].position - obj.aabb.center
            phi = (np.pi / 2) - np.arccos(p[1] / np.linalg.norm(p))

            # sample an up-down angle
            if self.sample_lookat_deg_delta is not None:
                deg = self.np_rng.uniform(
                    -self.sample_lookat_deg_delta, self.sample_lookat_deg_delta
                )
                phi = (phi + np.deg2rad(deg)) % (2 * np.pi)

            look_up_angle = np.clip(-phi, self.pitch_bounds[0], self.pitch_bounds[1])
            quat = ss["color_sensor"].rotation * quat_from_angle_axis(
                look_up_angle, habitat_sim.geo.RIGHT
            )

            new_states.append(
                AgentState(
                    position=agent_state.position,
                    rotation=agent_state.rotation,
                    sensor_states={
                        k: SixDOFPose(position=p.position, rotation=quat)
                        for k, p in ss.items()
                    },
                )
            )

        return new_states

    def sample_poses(self, obj: SemanticObject) -> List[AgentState]:
        return self.sample_camera_poses(self.sample_agent_poses_radially(obj), obj)
