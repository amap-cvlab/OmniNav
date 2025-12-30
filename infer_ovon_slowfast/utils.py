import cv2
import subprocess
import numpy as np
import argparse
import json
import os
import gzip
from omegaconf import OmegaConf
import datetime 
import habitat_sim
from frontier_utils import map_coors_to_pixel
from common.embodied_utils.simulator import HabitatSimulator
def make_video(goals,start_position,path_finder,global_color_list, global_frontier_list,video_path,split,sr,scene_id,cur_episode,object_category,log_file_path,global_state_list,top_down_map,sim):
    all_view_points = [
        vp["agent_state"]["position"]
        for goal in goals
        for vp in goal["view_points"]
    ]
    best_target_position = None
    min_geo_dist = float('inf')
    reachable_view_points = []
    if all_view_points:
        for current_view_point in all_view_points:
            path = habitat_sim.ShortestPath()
            path.requested_start = start_position
            path.requested_end = current_view_point
            if path_finder.find_path(path):
                current_geo_dist = path.geodesic_distance
                reachable_view_points.append((current_view_point, current_geo_dist))
                if current_geo_dist < min_geo_dist:
                    min_geo_dist = current_geo_dist
                    best_target_position = current_view_point
        if best_target_position is None and all_view_points:
            best_target_position = all_view_points[0]

    # --- Part 2: 检查和设置视频编写器 (逻辑不变) ---
    if not all([global_color_list, global_frontier_list]):
        print("错误: 列表为空，无法生成视频。")
        # continue
    
    height, width, _ = global_color_list[0][0].shape 
    
    # 【修改 1】将视频宽度从 3 倍改为 2 倍
    combined_width = width * 2
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(video_path, exist_ok=True)
    original_avi_path = f'{video_path}/split_{split}_sr_{sr}_process_{scene_id}_{cur_episode["episode_id"]}_{object_category}.avi'
    if 'generated_avi_files' not in locals():
        generated_avi_files = []
    generated_avi_files.append(original_avi_path)
    # 使用新的 combined_width 初始化视频编写器
    video = cv2.VideoWriter(original_avi_path, cv2.VideoWriter_fourcc(*'DIVX'), 2, (combined_width, height))
    log_message(f"Best target position for visualization: {str(best_target_position)}",log_file_path,timestamp)

    trajectory_points_scaled = []            
    for i in range(len(global_color_list)):
        color_frame, frame_type = global_color_list[i]
        frontier_frame, spin_type, label = global_frontier_list[i]
        current_agent_state = global_state_list[i]

        if spin_type == 'spin' and label == 'part':
            continue 
        
        color_view_bgr = color_frame
        
        frontier_frame_copy = frontier_frame.copy()
        map_h_frontier, map_w_frontier, _ = frontier_frame_copy.shape
        resized_frontier_map_bgr, scale, x_offset, y_offset = resize_with_padding(
            frontier_frame_copy, width, height
        )
        OTHER_VIEWPOINT_COLOR = (255, 165, 0)
        OTHER_VIEWPOINT_RADIUS = 3
        for vp, dist in reachable_view_points:
            if not np.array_equal(vp, best_target_position):
                map_y, map_x = map_coors_to_pixel(vp, top_down_map, sim)

                final_x = int(map_x * scale + x_offset)
                final_y = int(map_y * scale + y_offset)
                cv2.circle(resized_frontier_map_bgr, (final_x, final_y), radius=OTHER_VIEWPOINT_RADIUS, color=OTHER_VIEWPOINT_COLOR, thickness=-1)
        BEST_TARGET_COLOR = (0, 0, 255)
        BEST_TARGET_RADIUS = 5
        if best_target_position is not None:
            map_y, map_x = map_coors_to_pixel(best_target_position, top_down_map, sim) 
            

            final_x = int(map_x * scale + x_offset)
            final_y = int(map_y * scale + y_offset)
            cv2.circle(resized_frontier_map_bgr, (final_x, final_y), radius=BEST_TARGET_RADIUS, color=BEST_TARGET_COLOR, thickness=-1)

        


        map_y_agent, map_x_agent = map_coors_to_pixel(current_agent_state.position, top_down_map, sim)
        

        scaled_x_agent = int(map_x_agent * scale + x_offset)
        scaled_y_agent = int(map_y_agent * scale + y_offset)
        trajectory_points_scaled.append((scaled_x_agent, scaled_y_agent))

        
        if len(trajectory_points_scaled) > 1:
            overlay = resized_frontier_map_bgr.copy()
            
            TRAJECTORY_COLOR = (255, 0, 0) 
            TRAJECTORY_THICKNESS = 2
            ALPHA = 0.7

            for j in range(1, len(trajectory_points_scaled)):
                draw_dashed_line(
                    overlay, 
                    trajectory_points_scaled[j-1], 
                    trajectory_points_scaled[j], 
                    TRAJECTORY_COLOR, 
                    TRAJECTORY_THICKNESS,
                    dash_length=5,
                    gap_length=3
                )
            
            resized_frontier_map_bgr = cv2.addWeighted(overlay, ALPHA, resized_frontier_map_bgr, 1 - ALPHA, 0)


        combined_frame = cv2.hconcat([color_view_bgr, resized_frontier_map_bgr])
        
        video.write(combined_frame)
            
    video.release()
    print(f"save video path: {original_avi_path}")

    files_to_convert = generated_avi_files

    if not files_to_convert:
        print("no video file。")
    else:
        for file in files_to_convert:
            if not os.path.exists(file):
                continue

            base_name = os.path.splitext(file)[0]
            new_filename = f'{base_name}_convert.mp4'
            command = f'ffmpeg -y -i "{file}" -vf "setpts=0.25*PTS" -vcodec libx264 -an "{new_filename}"'
            command = f'{command} > /dev/null 2>&1'
            
            try:
                return_code = os.system(command)
                if return_code == 0:
                    print(f"convert sucessfully: {new_filename}")
                    log_message(f"Video conversion successful: {new_filename}",log_file_path,timestamp)
                    os.remove(file)
                else:
                    print(f" ffmpeg fail: {return_code}")
            except Exception as e:
                print(f" ffmpeg fail: {e}")
def get_sim_agent(sim_setting_path,goat_agent_setting_path,scene_path,cur_episode):
        sim_settings = OmegaConf.load(sim_setting_path)
        goat_agent_setting = OmegaConf.load(goat_agent_setting_path)
        sim_settings['scene'] = scene_path
        abstract_sim = HabitatSimulator(sim_settings, goat_agent_setting)
        sim = abstract_sim.simulator
        agent = abstract_sim.agent
        agent_state = habitat_sim.AgentState()
        start_position = cur_episode['start_position']
        start_rotation = cur_episode['start_rotation']
        agent_state.position = start_position
        agent_state.rotation = start_rotation
        agent.set_state(agent_state)
        path_finder = sim.pathfinder   
        return sim,agent,path_finder,start_position,start_rotation

def make_log_file(log_dir,split,name,scene_id,cur_episode_id,object_category):
        log_file_path = os.path.join(log_dir,name, f"split_{split}_process_{scene_id}_{cur_episode_id}_{object_category}.log")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(os.path.join(log_dir,name), exist_ok=True)
        return log_file_path,timestamp
def get_info(cur_data,hm3d_data_base_path,navigation_data_dict,split):
        scene_id = cur_data['scan_id']
        clean_scene_id = scene_id.split("-")[-1]
        scene_path = os.path.join(hm3d_data_base_path, scene_id, f"{clean_scene_id}.basis.glb")
        episode_index = cur_data['episode_index']
        object_category = cur_data['object_category']
        cur_episode = navigation_data_dict[split][scene_id]['episodes'][episode_index]
        assert cur_episode['object_category'] == object_category
        return scene_id,scene_path,episode_index,object_category,cur_episode

def load_navigation_data(embodied_scan_dir,navigation_data_path):
    navigation_data_dict = {'val_seen': {}, 'val_seen_synonyms': {}, 'val_unseen': {}}
    split_list = ['val_seen', 'val_seen_synonyms', 'val_unseen']
    train_val_split = json.load(open(os.path.join(embodied_scan_dir, 'HM3D', 'hm3d_annotated_basis.scene_dataset_config.json')))
    raw_scan_ids = set([pa.split('/')[1] for pa in train_val_split['scene_instances']['paths']['.json']])
    for split in split_list:
        data_dir = os.path.join(navigation_data_path, split, 'content')
        file_list = [f for f in os.listdir(data_dir) if f[0] != '.']
        for file_name in file_list:
            file_path = os.path.join(data_dir, file_name)
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f) 
                simplified_scan_id = file_name.split('.')[0]
                raw_scan_id = [pa for pa in raw_scan_ids if simplified_scan_id in pa][0]
                new_data = {}
                new_data['episodes'] = data['episodes']
                new_data['goals_by_category'] = dict([(k.split('glb_')[-1], v) for k, v in data['goals_by_category'].items()])
                navigation_data_dict[split][raw_scan_id] = new_data
    return split_list,navigation_data_dict


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', type=str, default='all')
    parser.add_argument('--model_path', type=int, default=' ')
    parser.add_argument('--name', type=int, default='omni')
    args = parser.parse_args()
    return args
def log_message(message,log_file_path,timestamp):
    with open(log_file_path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=5):
    """
    在图像上绘制虚线。
    :param img: 要绘制的图像
    :param pt1: 起始点 (x, y)
    :param pt2: 结束点 (x, y)
    :param color: 线的颜色 (B, G, R)
    :param thickness: 线的粗细
    :param dash_length: 虚线中每一小段的长度
    :param gap_length: 虚线中每一段间隔的长度
    """
    dist = ((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)**0.5
    if dist == 0: return 
    
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist


    current_pos = np.array(pt1, dtype=float)
    total_length_drawn = 0


    while total_length_drawn < dist:

        dash_end_x = current_pos[0] + dx * dash_length
        dash_end_y = current_pos[1] + dy * dash_length
        

        if total_length_drawn + dash_length > dist:
            dash_end_x, dash_end_y = pt2[0], pt2[1]


        cv2.line(img, (int(current_pos[0]), int(current_pos[1])), (int(dash_end_x), int(dash_end_y)), color, thickness)
        

        total_length_drawn += dash_length + gap_length
        current_pos[0] += dx * (dash_length + gap_length)
        current_pos[1] += dy * (dash_length + gap_length)

def resize_with_padding(image, target_width, target_height, bg_color=(255, 255, 255)):
    """
    保持原始长宽比缩放图像，并用背景色填充以达到目标尺寸。

    Args:
        image: 要处理的OpenCV图像 (BGR)。
        target_width: 目标宽度。
        target_height: 目标高度。
        bg_color: 填充的背景颜色 (BGR)。

    Returns:
        A tuple containing:
        - padded_image: 处理后的最终图像。
        - scale: 应用的缩放比例。
        - x_offset: 图像在画布上的水平偏移量。
        - y_offset: 图像在画布上的垂直偏移量。
    """
    src_h, src_w, _ = image.shape
    scale = min(target_width / src_w, target_height / src_h)


    new_w = int(src_w * scale)
    new_h = int(src_h * scale)


    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


    padded_image = np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)


    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2


    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return padded_image, scale, x_offset, y_offset