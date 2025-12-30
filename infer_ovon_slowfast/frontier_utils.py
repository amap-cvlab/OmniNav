import os
from typing import List, Optional, Tuple
from habitat.utils.visualizations import maps
import cv2
import numpy as np
from numba import njit
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)

from path_utils import a_star_search, completion_time_heuristic, euclidean_heuristic, path_dist_cost, path_time_cost


def convert_meters_to_pixel(meters: float, map_resolution, sim) -> int:
    return int(meters/ maps.calculate_meters_per_pixel(map_resolution, sim=sim))

def map_coors_to_pixel(position, top_down_map, sim) -> np.ndarray:
        a_x, a_y = maps.to_grid(
            position[2],
            position[0],
            (top_down_map.shape[0], top_down_map.shape[1]),
            sim=sim,
        )
        return np.array([a_x, a_y])

def pixel_to_map_coors(pixel: np.ndarray, agent_position, top_down_map, sim) -> np.ndarray:
        if pixel.ndim == 1:
            x, y = pixel
        else:
            x, y = pixel[:, 0], pixel[:, 1]
        realworld_x, realworld_y = maps.from_grid(
            x,
            y,
            (top_down_map.shape[0], top_down_map.shape[1]),
            sim,
        )
        if pixel.ndim == 1:
            return sim.pathfinder.snap_point(
                [realworld_y,agent_position[1], realworld_x]
            )
        snapped = [
            sim.pathfinder.snap_point([y, agent_position[1], x])
            for y, x in zip(realworld_y, realworld_x)  # noqa
        ]
        return np.array(snapped)
    
def get_polar_angle(agent_state):
    # quaternion is in x, y, z, w format
    ref_rotation = agent_state.rotation

    heading_vector = quaternion_rotate_vector(
        ref_rotation.inverse(), np.array([0, 0, -1])
    )

    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    z_neg_z_flip = np.pi
    return np.array(phi) + z_neg_z_flip
    
@njit
def wrap_heading(heading):
    """Ensures input heading is between -180 an 180; can be float or np.ndarray"""
    return (heading + np.pi) % (2 * np.pi) - np.pi

def vectorize_get_line_points(current_point, points, max_line_len):
    angles = np.arctan2(
        points[..., 1] - current_point[1], points[..., 0] - current_point[0]
    )
    endpoints = np.stack(
        (
            points[..., 0] + max_line_len * np.cos(angles),
            points[..., 1] + max_line_len * np.sin(angles),
        ),
        axis=-1,
    )
    endpoints = endpoints.astype(np.int32)

    line_points = np.stack([points.reshape(-1, 2), endpoints.reshape(-1, 2)], axis=1)
    return line_points

def get_two_farthest_points(source, cnt, agent_yaw):
    """Returns the two points in the contour cnt that form the smallest and largest
    angles from the source point."""
    pts = cnt.reshape(-1, 2)
    pts = pts - source
    rotation_matrix = np.array(
        [
            [np.cos(-agent_yaw), -np.sin(-agent_yaw)],
            [np.sin(-agent_yaw), np.cos(-agent_yaw)],
        ]
    )
    pts = np.matmul(pts, rotation_matrix)
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    # Get the two points that form the smallest and largest angles from the source
    min_idx = np.argmin(angles)
    max_idx = np.argmax(angles)
    return cnt[min_idx], cnt[max_idx]

def reveal_fog_of_war(
    top_down_map: np.ndarray,
    current_fog_of_war_mask: np.ndarray,
    current_point: np.ndarray,
    current_angle: float,
    fov: float = 90,
    max_line_len: float = 100,
    enable_debug_visualization: bool = False,
) -> np.ndarray:
    curr_pt_cv2 = current_point[::-1].astype(int)
    angle_cv2 = np.rad2deg(wrap_heading(-current_angle + np.pi / 2))

    cone_mask = cv2.ellipse(
        np.zeros_like(top_down_map),
        curr_pt_cv2,
        (int(max_line_len), int(max_line_len)),
        0,
        angle_cv2 - fov / 2,
        angle_cv2 + fov / 2,
        1,
        -1,
    )

    # Create a mask of pixels that are both in the cone and NOT in the top_down_map
    obstacles_in_cone = cv2.bitwise_and(cone_mask, 1 - top_down_map)

    # Find the contours of the obstacles in the cone
    obstacle_contours, _ = cv2.findContours(
        obstacles_in_cone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if enable_debug_visualization:
        vis_top_down_map = top_down_map * 255
        vis_top_down_map = cv2.cvtColor(vis_top_down_map, cv2.COLOR_GRAY2BGR)
        vis_top_down_map[top_down_map > 0] = (60, 60, 60)
        vis_top_down_map[top_down_map == 0] = (255, 255, 255)
        cv2.circle(vis_top_down_map, tuple(curr_pt_cv2), 3, (255, 192, 15), -1)
        # cv2.imshow("vis_top_down_map", vis_top_down_map)
        # cv2.waitKey(0)
        # cv2.destroyWindow("vis_top_down_map")
        cv2.imwrite("vis_top_down_map.png", vis_top_down_map)

        cone_minus_obstacles = cv2.bitwise_and(cone_mask, top_down_map)
        vis_cone_minus_obstacles = vis_top_down_map.copy()
        vis_cone_minus_obstacles[cone_minus_obstacles == 1] = (127, 127, 127)
        cv2.imwrite("vis_cone_minus_obstacles.png", vis_cone_minus_obstacles)
        # cv2.imshow("vis_cone_minus_obstacles", vis_cone_minus_obstacles)
        # cv2.waitKey(0)
        # cv2.destroyWindow("vis_cone_minus_obstacles")

        vis_obstacles_mask = vis_cone_minus_obstacles.copy()
        cv2.drawContours(vis_obstacles_mask, obstacle_contours, -1, (0, 0, 255), 1)
        # # cv2.imshow("vis_obstacles_mask", vis_obstacles_mask)
        # cv2.waitKey(0)
        cv2.imwrite("vis_obstacles_mask.png", vis_obstacles_mask)

    if len(obstacle_contours) == 0:
        return cv2.bitwise_or(current_fog_of_war_mask, cone_mask), vis_obstacles_mask # fill entire cone

    # Find the two points in each contour that form the smallest and largest angles
    # from the current position
    points = []
    for cnt in obstacle_contours:
        if cv2.isContourConvex(cnt):
            pt1, pt2 = get_two_farthest_points(curr_pt_cv2, cnt, angle_cv2)
            points.append(pt1.reshape(-1, 2))
            points.append(pt2.reshape(-1, 2))
        else:
            # Just add every point in the contour
            points.append(cnt.reshape(-1, 2))
    points = np.concatenate(points, axis=0)

    # Fragment the cone using obstacles and two lines per obstacle in the cone
    visible_cone_mask = cv2.bitwise_and(cone_mask, top_down_map)
    line_points = vectorize_get_line_points(curr_pt_cv2, points, max_line_len * 1.05)
    # Draw all lines simultaneously using cv2.polylines
    cv2.polylines(visible_cone_mask, line_points, isClosed=False, color=0, thickness=2)

    # Identify the contour that is closest to the current position
    final_contours, _ = cv2.findContours(
        visible_cone_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    visible_area = None
    min_dist = np.inf
    for cnt in final_contours:
        pt = tuple([int(i) for i in curr_pt_cv2])
        dist = abs(cv2.pointPolygonTest(cnt, pt, True))
        if dist < min_dist:
            min_dist = dist
            visible_area = cnt

    if enable_debug_visualization:
        vis_points_mask = vis_obstacles_mask.copy()
        for point in points.reshape(-1, 2):
            cv2.circle(vis_points_mask, tuple(point), 3, (0, 255, 0), -1)
        cv2.imwrite("vis_points_mask.png", vis_points_mask)
        # cv2.imshow("vis_points_mask", vis_points_mask)
        # cv2.waitKey(0)
        # cv2.destroyWindow("vis_points_mask")

        vis_lines_mask = vis_points_mask.copy()
        cv2.polylines(
            vis_lines_mask, line_points, isClosed=False, color=(0, 0, 255), thickness=2
        )
        # cv2.imshow("vis_lines_mask", vis_lines_mask)
        # cv2.waitKey(0)
        # cv2.destroyWindow("vis_lines_mask")
        cv2.imwrite("vis_lines_mask.png", vis_lines_mask)

        vis_final_contours = vis_top_down_map.copy()
        # Draw each contour in a random color
        for cnt in final_contours:
            color = tuple([int(i) for i in np.random.randint(0, 255, 3)])
            cv2.drawContours(vis_final_contours, [cnt], -1, color, -1)
        # cv2.imshow("vis_final_contours", vis_final_contours)
        # cv2.waitKey(0)
        # cv2.destroyWindow("vis_final_contours")
        cv2.imwrite("vis_final_contours.png", vis_final_contours)

        vis_final = vis_top_down_map.copy()
        # Draw each contour in a random color
        if visible_area is not None and len(visible_area) > 0:
            visible_area = visible_area.astype(np.int32)
            cv2.drawContours(vis_final, [visible_area], -1, (127, 127, 127), -1)
        # cv2.imshow("vis_final", vis_final)
        # cv2.waitKey(0)
        # cv2.destroyWindow("vis_final")
        cv2.imwrite("vis_final.png", vis_final)

    if min_dist > 3:
        return current_fog_of_war_mask,vis_obstacles_mask  # the closest contour was too far away

    new_fog = cv2.drawContours(current_fog_of_war_mask, [visible_area], 0, 1, -1)

    return new_fog,vis_obstacles_mask

DEBUG = False
VISUALIZE= False

def filter_out_small_unexplored(
    full_map: np.ndarray, explored_mask: np.ndarray, area_thresh: int
):
    """Edit the explored map to add small unexplored areas, which ignores their
    frontiers."""
    if area_thresh == -1:
        return explored_mask

    unexplored_mask = full_map.copy()
    unexplored_mask[explored_mask > 0] = 0

    if VISUALIZE:
        img = cv2.cvtColor(unexplored_mask * 255, cv2.COLOR_GRAY2BGR)
        cv2.imshow("unexplored mask", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Find contours in the unexplored mask
    contours, _ = cv2.findContours(
        unexplored_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if VISUALIZE:
        img = cv2.cvtColor(unexplored_mask * 255, cv2.COLOR_GRAY2BGR)
        # Draw the contours in red
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        cv2.imshow("unexplored mask with contours", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Add small unexplored areas to the explored map
    small_contours = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < area_thresh:
            mask = np.zeros_like(explored_mask)
            mask = cv2.drawContours(mask, [contour], 0, 1, -1)
            masked_values = unexplored_mask[mask.astype(bool)]
            values = set(masked_values.tolist())
            if 1 in values and len(values) == 1:
                small_contours.append(contour)
    new_explored_mask = explored_mask.copy()
    cv2.drawContours(new_explored_mask, small_contours, -1, 255, -1)

    if VISUALIZE and len(small_contours) > 0:
        # Draw the full map and the new explored mask, then outline the contours that
        # were added to the explored mask
        img = cv2.cvtColor(full_map * 255, cv2.COLOR_GRAY2BGR)
        img[new_explored_mask > 0] = (127, 127, 127)
        cv2.drawContours(img, small_contours, -1, (0, 0, 255), 3)
        cv2.imshow("small unexplored areas", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return new_explored_mask

@njit
def contour_to_frontiers(contour, unexplored_mask):
    """Given a contour from OpenCV, return a list of numpy arrays. Each array contains
    contiguous points forming a single frontier. The contour is assumed to be a set of
    contiguous points, but some of these points are not on any frontier, indicated by
    having a value of 0 in the unexplored mask. This function will split the contour
    into multiple arrays that exclude such points."""
    bad_inds = []
    num_contour_points = len(contour)
    for idx in range(num_contour_points):
        x, y = contour[idx][0]
        if unexplored_mask[y, x] == 0:
            bad_inds.append(idx)
    frontiers = np.split(contour, bad_inds)
    # np.split is fast but does NOT remove the element at the split index
    filtered_frontiers = []
    front_last_split = (
        0 not in bad_inds
        and len(bad_inds) > 0
        and max(bad_inds) < num_contour_points - 2
    )
    for idx, f in enumerate(frontiers):
        # a frontier must have at least 2 points (3 with bad ind)
        if len(f) > 2 or (idx == 0 and front_last_split):
            if idx == 0:
                filtered_frontiers.append(f)
            else:
                filtered_frontiers.append(f[1:])
    # Combine the first and last frontier if the first point of the first frontier and
    # the last point of the last frontier are the first and last points of the original
    # contour. Only check if there are at least 2 frontiers.
    if len(filtered_frontiers) > 1 and front_last_split:
        last_frontier = filtered_frontiers.pop()
        filtered_frontiers[0] = np.concatenate((last_frontier, filtered_frontiers[0]))
    return filtered_frontiers

def _bresenhamline_nslope(slope):
    """
    Normalize slope for Bresenham's line algorithm.

    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])

    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])

    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope

def _bresenhamlines(start, end, max_iter):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension)

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)

def bresenhamline(start, end, max_iter=5):
    """
    Returns a list of points from (start, end) by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed

    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])


def interpolate_contour(contour):
    """Given a cv2 contour, this function will add points in between each pair of
    points in the contour using the bresenham algorithm to make the contour more
    continuous.
    :param contour: A cv2 contour of shape (N, 1, 2)
    :return:
    """
    # First, reshape and expand the frontier to be a 2D array of shape (N-1, 2, 2)
    # representing line segments between adjacent points
    line_segments = np.concatenate((contour[:-1], contour[1:]), axis=1).reshape(
        (-1, 2, 2)
    )
    # Also add a segment connecting the last point to the first point
    line_segments = np.concatenate(
        (line_segments, np.array([contour[-1], contour[0]]).reshape((1, 2, 2)))
    )
    pts = []
    for (x0, y0), (x1, y1) in line_segments:
        pts.append(
            bresenhamline(np.array([[x0, y0]]), np.array([[x1, y1]]), max_iter=-1)
        )
    pts = np.concatenate(pts).reshape((-1, 1, 2))
    return pts

def detect_frontiers(
    full_map: np.ndarray, explored_mask: np.ndarray, area_thresh: Optional[int] = -1
) -> List[np.ndarray]:
    """Detects frontiers in a map.

    Args:
        full_map (np.ndarray): White polygon on black image, where white is navigable.
        Mono-channel mask.
        explored_mask (np.ndarray): Portion of white polygon that has been seen already.
        This is also a mono-channel mask.
        area_thresh (int, optional): Minimum unexplored area (in pixels) needed adjacent
        to a frontier for that frontier to be valid. Defaults to -1.

    Returns:
        np.ndarray: A mono-channel mask where white contours represent each frontier.
    """
    # Find the contour of the explored area
    filtered_explored_mask = filter_out_small_unexplored(
        full_map, explored_mask, area_thresh
    )
    contours, _ = cv2.findContours(
        filtered_explored_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if VISUALIZE:
        img = cv2.cvtColor(full_map * 255, cv2.COLOR_GRAY2BGR)
        img[explored_mask > 0] = (127, 127, 127)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        cv2.imshow("contours", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    unexplored_mask = np.where(filtered_explored_mask > 0, 0, full_map)
    unexplored_mask = cv2.blur(  # blurring for some leeway
        np.where(unexplored_mask > 0, 255, unexplored_mask), (3, 3)
    )
    frontiers = []
    # TODO: There shouldn't be more than one contour (only one explored area on map)
    for contour in contours:
        frontiers.extend(
            contour_to_frontiers(interpolate_contour(contour), unexplored_mask)
        )
    return frontiers

def closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    segment = b - a
    t = np.einsum("ij,ij->i", p - a, segment) / np.einsum("ij,ij->i", segment, segment)
    t = np.clip(t, 0, 1)
    return a + t[:, np.newaxis] * segment

def closest_line_segment(
    coord: np.ndarray, segments: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    closest_points = closest_point_on_segment(coord, segments[:, 0], segments[:, 1])
    # Identify the segment that yielded the closest point
    min_idx = np.argmin(np.linalg.norm(closest_points - coord, axis=1))
    closest_segment, closest_point = segments[min_idx], closest_points[min_idx]

    return closest_segment, closest_point

def get_closest_frontier_point(xy, frontier):
    """Returns the point on the frontier closest to the given coordinate."""
    # First, reshape and expand the frontier to be a 2D array of shape (X, 2)
    # representing line segments between adjacent points
    line_segments = np.concatenate([frontier[:-1], frontier[1:]], axis=1).reshape(
        (-1, 2, 2)
    )
    closest_segment, closest_point = closest_line_segment(xy, line_segments)
    return closest_point

@njit
def get_frontier_midpoint(frontier) -> np.ndarray:
    """Given a list of contiguous points (numpy arrays) representing a frontier, first
    calculate the total length of the frontier, then find the midpoint of the
    frontier"""
    # First, reshape and expand the frontier to be a 2D array of shape (X, 2, 2)
    # representing line segments between adjacent points
    line_segments = np.concatenate((frontier[:-1], frontier[1:]), axis=1).reshape(
        (-1, 2, 2)
    )
    # Calculate the length of each line segment
    line_lengths = np.sqrt(
        np.square(line_segments[:, 0, 0] - line_segments[:, 1, 0])
        + np.square(line_segments[:, 0, 1] - line_segments[:, 1, 1])
    )
    cum_sum = np.cumsum(line_lengths)
    total_length = cum_sum[-1]
    # Find the midpoint of the frontier
    half_length = total_length / 2
    # Find the line segment that contains the midpoint
    line_segment_idx = np.argmax(cum_sum > half_length)
    # Calculate the coordinates of the midpoint
    line_segment = line_segments[line_segment_idx]
    line_length = line_lengths[line_segment_idx]
    # Use the difference between the midpoint length and cumsum
    # to find the proportion of the line segment that the midpoint is at
    length_up_to = cum_sum[line_segment_idx - 1] if line_segment_idx > 0 else 0
    proportion = (half_length - length_up_to) / line_length
    # Calculate the midpoint coordinates
    midpoint = line_segment[0] + proportion * (line_segment[1] - line_segment[0])
    return midpoint

def frontier_waypoints(
    frontiers: List[np.ndarray], xy: Optional[np.ndarray] = None
) -> np.ndarray:
    """For each given frontier, returns the point on the frontier closest (euclidean
    distance) to the given coordinate. If coordinate is not given, will just return
    the midpoints of each frontier.

    Args:
        frontiers (List[np.ndarray]): list of arrays of shape (X, 1, 2), where each
        array is a frontier and X is NOT the same across arrays
        xy (np.ndarray): the given coordinate

    Returns:
        np.ndarray: array of waypoints, one for each frontier
    """
    if xy is None:
        return np.array([get_frontier_midpoint(i) for i in frontiers])
    return np.array([get_closest_frontier_point(xy, i) for i in frontiers])

# 假设必要的库已导入，如 np, cv2, os, Optional
# 假设 wrap_heading, detect_frontiers, frontier_waypoints 等函数已定义
import cv2
import numpy as np

def draw_fancy_arrow(image, center, radius, angle_deg, color_arrow, color_bg):
    """
    在图像上绘制一个精致的、带背景的箭头图标。

    Args:
        image: 要绘制的OpenCV图像。
        center: 图标的中心点 (x, y)。
        radius: 背景圆的半径。
        angle_deg: 箭头指向的角度 (0度朝向右方，顺时针增加)。
        color_arrow: 箭头的颜色 (B, G, R)。
        color_bg: 背景圆的颜色 (B, G, R)。
    """
    center = tuple(map(int, center))
    radius = int(radius)
    
    # 1. 绘制白色的圆形背景，并使用抗锯齿让边缘更平滑
    cv2.circle(image, center, radius, color_bg, -1, cv2.LINE_AA)

    # 2. 定义箭头的基本形状（一个等腰三角形），假设它朝向 0 度（右方）
    # 这些比例可以调整以改变箭头的形状
    arrow_tip = (radius * 0.9, 0)
    arrow_left = (-radius * 0.4, -radius * 0.6)
    arrow_right = (-radius * 0.4, radius * 0.6)
    
    points = np.array([arrow_tip, arrow_left, arrow_right], dtype=np.float32)

    # 3. 创建旋转矩阵
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    # 4. 旋转并平移顶点
    # @ 是矩阵乘法运算符
    rotated_points = points @ rotation_matrix.T
    translated_points = rotated_points + center
    
    # 5. 绘制填充后的蓝色箭头
    pts_for_poly = [np.int32(translated_points)]
    cv2.fillPoly(image, pts_for_poly, color_arrow, cv2.LINE_AA)


def detect_frontier_waypoints(
    full_map: np.ndarray,
    explored_mask: np.ndarray,
    current_point: np.ndarray,
    current_angle: float,
    fov: float = 90,
    max_line_len: float = 100,
    area_thresh: Optional[int] = -1,
    xy: Optional[np.ndarray] = None,
    enable_visualization: bool = False,
    tag:Optional[str] = None,
):
    if DEBUG:
        import time

        os.makedirs("map_debug", exist_ok=True)
        cv2.imwrite(
            f"map_debug/{int(time.time())}_debug_full_map_{area_thresh}.png", full_map
        )
        cv2.imwrite(
            f"map_debug/{int(time.time())}_debug_explored_mask_{area_thresh}.png",
            explored_mask,
        )

    if VISUALIZE:
        inverted_map = (1 - full_map) * 255 
        img = cv2.cvtColor(inverted_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img[explored_mask > 0] = (127, 127, 127)

        cv2.imshow("inputs", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    explored_mask[full_map == 0] = 0
    frontiers = detect_frontiers(full_map, explored_mask, area_thresh)
    waypoints = frontier_waypoints(frontiers, None)

    
    img_with_fov = None # 初始化返回值
    if enable_visualization:
        inverted_map = (1 - full_map) * 255 
        img = cv2.cvtColor(inverted_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img[explored_mask > 0] = (127, 127, 127)
        # (绘制边界点和路径点的代码保持不变)
        for idx, frontier in enumerate(frontiers):
            # ... (这部分绘制彩色边界的代码不变) ...
            color = cv2.applyColorMap(np.uint8([255 * (idx + 1) / len(frontiers)]), cv2.COLORMAP_RAINBOW)[0][0]
            color = tuple(int(i) for i in color)
            for idx2, p in enumerate(frontier):
                if idx2 < len(frontier) - 1:
                    cv2.line(img, p[0], frontier[idx2 + 1][0], color, 3)
            waypoint = waypoints[idx]
            cv2.putText(img, str(idx), tuple(waypoint.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(waypoint.astype(int)), 5, color, -1)
        
        overlay = img.copy()

        # 1. 确定中心点和扇形/圆形尺寸
        curr_pt_cv2 = tuple(current_point[::-1].astype(int))
        axes = (int(max_line_len), int(max_line_len))

        # 2. 根据 tag 决定是画圆还是扇形
        if tag == 'spin':
            cv2.ellipse(overlay, curr_pt_cv2, axes, 0, 0, 360, (255, 255, 255), -1)
        else:
            angle_cv2 = np.rad2deg(wrap_heading(-current_angle + np.pi / 2)) 
            start_angle = angle_cv2 - fov / 2
            end_angle = angle_cv2 + fov / 2
            cv2.ellipse(overlay, curr_pt_cv2, axes, 0, start_angle, end_angle, (255, 255, 255), -1)
        
        # 3. 将覆盖层和原图进行半透明混合
        alpha = 0.3
        img_with_fov = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # ==================== 【主要修改部分】绘制精致箭头或红点 ====================
        # (注意：我们在混合后的图像 img_with_fov 上绘制，以确保它们在最顶层且不透明)

        if tag == 'spin':
            # --- 如果是圆形，在中心画一个大的红色实心圆点 ---
            SPIN_DOT_COLOR = (230, 176, 170) 
            SPIN_DOT_RADIUS = 15         # 半径可以稍大一些
            cv2.circle(img_with_fov, curr_pt_cv2, SPIN_DOT_RADIUS, SPIN_DOT_COLOR, -1, cv2.LINE_AA)
        else:
            # --- 如果是扇形，调用新函数绘制精致的箭头 ---
            ARROW_RADIUS = 15                 # 箭头的整体大小 (背景圆半径)
            ARROW_BG_COLOR = (255, 255, 255)  # 背景色：白色
            ARROW_FG_COLOR = (230, 176, 170)    # 箭头色：漂亮的蓝色 (BGR)
            
            # 这里的 angle_cv2 已经计算好了，可以直接使用
            draw_fancy_arrow(
                img_with_fov,
                curr_pt_cv2,
                ARROW_RADIUS,
                angle_cv2,
                ARROW_FG_COLOR,
                ARROW_BG_COLOR
            )
        # ==============================================================================

        # 绘制原始的 xy 点 (如果存在)
        # if xy is not None:
        #     cv2.circle(img_with_fov, tuple(xy.astype(int)), 5, (255, 255, 255), -1)

        cv2.imwrite("frontiers.png", img_with_fov)
    
    # 确保有返回值
    if img_with_fov is None and enable_visualization:
        img_with_fov = cv2.cvtColor(full_map * 255, cv2.COLOR_GRAY2BGR)

    return waypoints, img_with_fov

def astar_search(sim_waypoints, start_position, sim):
  
    def heuristic_fn(x):
        return euclidean_heuristic(x, start_position)

    def cost_fn(x):
        return path_dist_cost(x, start_position, sim)

    return a_star_search(sim_waypoints, heuristic_fn, cost_fn)
    
def get_closest_waypoint(frontier_waypoints, agent_position, top_down_map, sim):
    if len(frontier_waypoints) == 0:
        return None
    sim_waypoints = pixel_to_map_coors(frontier_waypoints, agent_position, top_down_map, sim)
    idx, _ = astar_search(sim_waypoints, agent_position, sim)
    if idx is None:
        return None

    return frontier_waypoints[idx]