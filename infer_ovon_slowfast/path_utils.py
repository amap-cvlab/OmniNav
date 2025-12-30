import habitat_sim
import numpy as np
from numba import njit

from numba import njit

@njit
def wrap_heading(heading):
    """Ensures input heading is between -180 an 180; can be float or np.ndarray"""
    return (heading + np.pi) % (2 * np.pi) - np.pi

@njit
def completion_time_heuristic(
    sim_waypoints, agent_position, agent_heading, lin_vel, ang_vel
):
    """An admissible, consistent heuristic for finding the completion time of an
    obstacle-free path to a waypoint. For each waypoint, this returns the shortest
    amount of time it would take to reach that waypoint from the agent's current
    position and heading assuming NO obstacles, serving as a lower bound on the actual
    completion time. Assumes agent has point-turn dynamics."""
    euclidean_dists = euclidean_heuristic(sim_waypoints, agent_position)
    heading_to_waypoints = np.arctan2(
        sim_waypoints[:, 2] - agent_position[2],
        sim_waypoints[:, 0] - agent_position[0],
    )
    heading = wrap_heading(np.pi / 2.0 - agent_heading)
    heading_errors = np.abs(wrap_heading(heading_to_waypoints - heading))
    completion_times = heading_errors / ang_vel + euclidean_dists / lin_vel
    return completion_times


@njit
def euclidean_heuristic(sim_waypoints, agent_position):
    """An admissible, consistent heuristic for finding the length of the shortest
    obstacle-free path to a waypoint. For each waypoint, this returns its euclidean
    distance from the agent's position, serving as a lower bound on the actual
    length of the shortest obstacle-free path."""
    euclid_dists = np.sqrt(
        (sim_waypoints[:, 2] - agent_position[2]) ** 2
        + (sim_waypoints[:, 0] - agent_position[0]) ** 2
    )
    return euclid_dists


@njit
def shortest_path_completion_time(path, max_lin_vel, max_ang_vel, yaw_diff):
    time = 0
    cur_pos = path[0]
    cur_yaw = None
    for i in range(1, path.shape[0]):
        target_pos = path[i]
        target_yaw = np.arctan2(target_pos[2] - cur_pos[2], target_pos[0] - cur_pos[0])

        distance = np.sqrt(
            (target_pos[2] - cur_pos[2]) ** 2 + (target_pos[0] - cur_pos[0]) ** 2
        )
        if cur_yaw is not None:
            yaw_diff = np.abs(wrap_heading(target_yaw - cur_yaw))

        lin_time = distance / max_lin_vel
        ang_time = yaw_diff / max_ang_vel
        time += lin_time + ang_time

        cur_pos = target_pos
        cur_yaw = target_yaw

    return time


def heading_error(
    agent_position: np.ndarray, position: np.ndarray, agent_heading: np.ndarray
) -> float:
    heading_to_waypoint = np.arctan2(
        position[2] - agent_position[2], position[0] - agent_position[0]
    )
    agent_heading = wrap_heading(np.pi / 2.0 - agent_heading)
    heading_error = wrap_heading(heading_to_waypoint - agent_heading)
    return heading_error


def get_path(start, end, sim):
    shortest_path = habitat_sim.nav.ShortestPath()
    shortest_path.requested_start = start
    shortest_path.requested_end = end
    if not sim.pathfinder.find_path(shortest_path):
        return None
    return shortest_path


def path_dist_cost(start, end, sim):
    path = get_path(start, end, sim)
    if path is None:
        return np.inf
    cost = path.geodesic_distance
    return cost


def path_time_cost(end, start, start_heading, lin_vel, ang_vel, sim):
    path = get_path(start, end, sim)
    if path is None:
        return np.inf
    path_points = np.array(path.points)
    if len(path_points) < 2:
        return np.inf
    heading_err = heading_error(start, path_points[1], start_heading)
    cost = shortest_path_completion_time(
        path_points, lin_vel, ang_vel, np.abs(heading_err)
    )
    return cost


def a_star_search(candidates, heuristic_fn, cost_fn):
    heuristics = heuristic_fn(candidates)
    sorted_inds = np.argsort(heuristics)
    min_cost = np.inf
    best_idx = None
    for idx in sorted_inds:
        if heuristics[idx] > min_cost:
            break
        cost = cost_fn(candidates[idx])
        if cost < min_cost:
            min_cost = cost
            best_idx = idx
    return best_idx, min_cost


def is_in_2d_array(arr_1d, arr_2d):
    for row in arr_2d:
        if np.array_equal(arr_1d, row):
            return True
    return False
