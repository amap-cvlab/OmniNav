from collections import defaultdict
import os
from habitat.utils.visualizations import maps
from habitat_sim import Simulator as Sim
import json
import habitat_sim
import numpy as np
from habitat.tasks.nav.nav import TopDownMap

from frontier_utils import convert_meters_to_pixel, detect_frontier_waypoints, get_closest_waypoint, get_polar_angle, map_coors_to_pixel, pixel_to_map_coors, reveal_fog_of_war
from sim_utils import get_simulator
import cv2
import random
from utils import log_message,add_arguments,load_navigation_data,get_info,make_log_file,get_sim_agent,make_video
from qwen_utils import getresult,Bank,get_result_fast
import torch
import numpy as np
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
# hyperparameter
data_set_path = "./data/dataset/embodied_bench_data/embodied_bench_data/our-set/ovon_full_set.json"
navigation_data_path = "./data/dataset/embodied_bench_data/embodied_bench_data/ovon"
hm3d_data_base_path = "./data/dataset/hm3d/val"
embodied_scan_dir = "./data/dataset/embodied_scan"
output_path = "./data/output/mturesult/jsonl/ovon-test.json"
log_dir="./data/output/mturesult/log"
sim_setting_path='configs/habitat/goat_sim_config.yaml'
goat_agent_setting_path='configs/habitat/goat_agent_config.yaml'
video_path='./data/output/mturesult/video'

decision_num_min = 3
visible_radius = 10
generated_avi_files = []
min_pixels = 56 * 56
max_pixels = 4480 * 4480
map_resolution = 512
REPRODUCIBLE_SEED = 42

args = add_arguments()
pattern = args.pattern
model_path=args.model_path
name=args.name
fast_type=args.type
# type
enable_visualization = True
# pattern = 'all'
# model_path='./data/ckpts/Qwen2.5-VL-3B_mh_data_more_image_5_3_3epoch_vit'
# name='args_name'
# fast_type='A-star'
# load navigation data
split_list,navigation_data_dict=load_navigation_data(embodied_scan_dir,navigation_data_path)
omni_nav=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",attn_implementation="flash_attention_2"
    )
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

# record result
if os.path.exists(output_path):
    result_dict = json.load(open(output_path, "r"))
else:
    result_dict = {'val_seen': [], 'val_seen_synonyms': [], 'val_unseen': []}
# Filter out data in data_set which is already in result_dict
data_set = json.load(open(data_set_path, "r"))
for split in split_list:
    existing_episodes = {(result['scan_id'], result['episode_index']) for result in result_dict[split]}
    data_set[split] = [episode for episode in data_set[split] if (episode['scan_id'], episode['episode_index']) not in existing_episodes]


for split in split_list:
    random.seed(REPRODUCIBLE_SEED)
    if pattern =='all':
        split_list=[]
        for key,scene in navigation_data_dict[split].items():
            num=0
            for item in scene['episodes']:
                split_list.append(
                    {"scan_id":key,
                    "episode_index":num,
                    "object_category":item['object_category'],
                    }
                )
                num=num+1
        episode_list_to_process = list(split_list)
    else:
        episode_list_to_process = list(data_set[split])

    random.shuffle(episode_list_to_process)
    for cur_data in episode_list_to_process:

        scene_id,scene_path,episode_index,object_category,cur_episode=get_info(cur_data,hm3d_data_base_path,navigation_data_dict,split)
        log_file_path,timestamp=make_log_file(log_dir,split,name,scene_id,cur_episode['episode_id'],object_category)
        # load target
        goals = navigation_data_dict[split][scene_id]['goals_by_category'][object_category]
        # get simulator
        sim,agent,path_finder,start_position,start_rotation=get_sim_agent(sim_setting_path,goat_agent_setting_path,scene_path,cur_episode)
        # get fronier param
        top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=map_resolution,  draw_border=False)
        fog_of_war_mask = np.zeros_like(top_down_map)
        area_thres_in_pixels =  convert_meters_to_pixel(9, map_resolution, sim)
        visibility_dist_in_pixels = convert_meters_to_pixel(visible_radius, map_resolution, sim)

        # start decision
        decision_num = 0
        total_steps = 0
        global_color_list,global_frontier_list,global_state_list = [],[],[]
        goto_color_list,goto_agent_state_list  = [],[]

        prev_agent_state = agent.get_state()
        episode_cum_distance = 0
        visited_frontier_set = set()


        # visited frontier
        bank=Bank(goto_sample_interval=1)
        color_past=None
        agent_past=None
        while total_steps < 2000:
            if goto_color_list:
                bank.add(goto_color_list, goto_agent_state_list, data_type='goto')
                goto_color_list,goto_agent_state_list = [], []
            # spin around
            color_list ,agent_state_list= [],[]
            action_list = ['turn_left'] * 12

            for i, action in enumerate(action_list):
                obervations = sim.step(action=action)
                color = obervations['color_sensor'][:, :, :3]
                color_list.append(color)
                global_color_list.append((color, 'spin'))
                agent_state = agent.get_state()
                agent_state_list.append(agent_state)
                fog_of_war_mask,vis_obstacles_mask = reveal_fog_of_war(top_down_map=top_down_map, current_fog_of_war_mask=fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=110, max_line_len=visibility_dist_in_pixels, enable_debug_visualization=enable_visualization)
                frontier_waypoints,frontier = detect_frontier_waypoints(top_down_map, fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=110, max_line_len=visibility_dist_in_pixels,area_thresh=area_thres_in_pixels, xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1], enable_visualization=enable_visualization,tag='spin')
                tag = 'all' if i == len(action_list) - 1 else 'part'
                global_frontier_list.append((frontier, 'spin',tag))
                total_steps += 1
            if total_steps==12:
                rgb_image_data = sim.get_sensor_observations()['color_sensor'][:, :, :3]
                rgb_image_data = cv2.cvtColor(rgb_image_data, cv2.COLOR_BGR2RGB)
                color_list.append(rgb_image_data)
                agent_state = agent.get_state()
                agent_state_list.append(agent_state)
            else:
                 color_list.append(color_past)
                 agent_state_list.append(agent_past)
            agent_state = agent.get_state()
            bank.add(color_list[::3], agent_state_list[::3], data_type='spin')
            global_state_list.extend(agent_state_list[0:12])
            # compute frontier
            frontier_waypoints,frontier = detect_frontier_waypoints(top_down_map, fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=110, max_line_len=visibility_dist_in_pixels,area_thresh=area_thres_in_pixels, xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1], enable_visualization=enable_visualization,tag='spin')
            if len(frontier_waypoints) == 0:
                frontier_waypoints = []
            else:
                frontier_waypoints = frontier_waypoints[:, ::-1]
                frontier_waypoints = pixel_to_map_coors(frontier_waypoints, agent_state.position, top_down_map, sim)
            # filter out visited frontier
            frontier_waypoints = [waypoint for waypoint in frontier_waypoints if tuple(np.round(waypoint, 1)) not in visited_frontier_set]
            target_position, is_final_decision,output_texts = getresult(
                                                                omni_nav,
                                                                processor,
                                                                bank,
                                                                current_frontiers=frontier_waypoints,
                                                                decision_agent_state=agent_state, # 传入决策时的状态作为参考系
                                                                object_category=object_category,
                                                                decision_num=decision_num,
                                                                visited_frontier_set=visited_frontier_set
                                                            )

            print(f"Model decision: target_position={target_position}, is_final_decision={is_final_decision}")
            bank.clear()
            log_message(f"Model decision: target_position={target_position}, is_final_decision={is_final_decision},\n output:{output_texts}",log_file_path,timestamp)
            decision_num += 1
            if fast_type=='A-star':
                try:
                    if not is_final_decision:
                        visited_frontier_set.add(tuple(np.round(target_position, 1)))
                    agent_island = path_finder.get_island(agent_state.position)
                    target_on_navmesh = path_finder.snap_point(point=target_position, island_index=agent_island)
                    follower = habitat_sim.GreedyGeodesicFollower(path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right")

                    action_list = follower.find_path(target_on_navmesh)
                except Exception as e:
                    log_message(f"start backed path...",log_file_path,timestamp)

                    path_found = False
                    action_list = []
                    alternative_waypoints = list(frontier_waypoints)
                    random.shuffle(alternative_waypoints)

                    for alternative_target in alternative_waypoints:
                        if (alternative_target == target_position).all():
                            continue
                        print(f"alternative_target {alternative_target}")
                        try:
                            alt_target_on_navmesh = path_finder.snap_point(point=alternative_target, island_index=agent_island)
                            action_list = follower.find_path(alt_target_on_navmesh)
                            log_message(f"sucess backed path: {alternative_target} ",log_file_path,timestamp)


                            target_position = alternative_target
                            target_on_navmesh = alt_target_on_navmesh

                            path_found = True
                            break

                        except Exception as fallback_e:
                            log_message(f"fail alternative_target:{alternative_target} : {fallback_e}",log_file_path,timestamp)
                            continue

                    if not path_found:
                        break

                goto_color_list ,goto_agent_state_list= [],[]
                if action_list==None:
                    break
                for action in action_list:
                    if action:
                        obervations = sim.step(action=action)
                        agent_state = agent.get_state()
                        color = obervations['color_sensor'][:, :, :3]
                        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                        goto_color_list.append(color)
                        global_color_list.append((color, 'goto'))
                        goto_agent_state_list.append(agent_state)
                        fog_of_war_mask,vis_obstacles_mask = reveal_fog_of_war(top_down_map=top_down_map, current_fog_of_war_mask=fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=110, max_line_len=visibility_dist_in_pixels, enable_debug_visualization=enable_visualization)
                        total_steps += 1
                        episode_cum_distance += np.linalg.norm(agent_state.position - prev_agent_state.position)
                        prev_agent_state = agent_state
                        frontier_waypoints,frontier = detect_frontier_waypoints(top_down_map, fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=110, max_line_len=visibility_dist_in_pixels,area_thresh=area_thres_in_pixels, xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1], enable_visualization=enable_visualization,tag='goto')
                        global_frontier_list.append((frontier, 'goto','part'))
                global_state_list.extend(goto_agent_state_list)
                # break on final decision
                if goto_color_list==[]:
                    break
                color_past=goto_color_list[-1]
                agent_past=goto_agent_state_list[-1]
                if is_final_decision:
                    break
            elif fast_type=='point-goal':
                goto_color_list ,goto_agent_state_list= [],[]
                if action_list==None:
                    break
                count = 0

                while float(np.linalg.norm(np.array(agent.get_state().position) - np.array(target_position)))>0.1 and count < 60:
                    count = count + 1

                    obervations,positions,rot = get_result_fast(omni_nav,
                                                    processor,
                                                    bank,
                                                    global_color_list,
                                                    global_state_list,
                                                    sim,
                                                    agent,
                                                    target_position,
                                                    log_file_path,
                                                    current_frontiers=frontier_waypoints,
                                                    decision_agent_state=agent_state, # 传入决策时的状态作为参考系
                                                    object_category=object_category,
                                                    decision_num=decision_num,
                                                    rot=rot
                                                    )
                    log_message(f'count:{count}')
                    # global_color_list.append((obervations['color_sensor'][:, :, :3], 'goto')) # 添加 
                    if obervations=='000':
                        split='stop'
                        is_final_decision = True
                        break
                    agent_state = agent.get_state()
                    log_message(f"Model position={agent_state.position}")
                # except Exception as e:
                    color = obervations['color_sensor'][:, :, :3] # (h,w,4) 0-255
                    # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    goto_color_list.append(color)
                    global_color_list.append((color, 'goto'))
                    depth = obervations['depth_sensor'][:, :] # (h,w) float
                    goto_agent_state_list.append(agent_state)
                    global_state_list.append(agent_state)
                    fog_of_war_mask,vis_obstacles_mask = reveal_fog_of_war(top_down_map=top_down_map, current_fog_of_war_mask=fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=110, max_line_len=visibility_dist_in_pixels, enable_debug_visualization=enable_visualization)
                    total_steps += 1
                    episode_cum_distance += np.linalg.norm(agent_state.position - prev_agent_state.position)
                    prev_agent_state = agent_state
                    frontier_waypoints,frontier = detect_frontier_waypoints(top_down_map, fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=110, max_line_len=visibility_dist_in_pixels,area_thresh=area_thres_in_pixels, xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1], enable_visualization=enable_visualization,tag='goto')
                    OTHER_VIEWPOINT_COLOR = (155, 10, 0)
                    OTHER_VIEWPOINT_RADIUS = 2
                    map_y, map_x = map_coors_to_pixel(target_position, top_down_map, sim)
                    cv2.circle(frontier, (map_x, map_y), radius=5, color=OTHER_VIEWPOINT_COLOR, thickness=-1)
                    for vp in positions:
                        map_y, map_x = map_coors_to_pixel(vp, top_down_map, sim)
                        cv2.circle(frontier, (map_x, map_y), radius=OTHER_VIEWPOINT_RADIUS, color=OTHER_VIEWPOINT_COLOR, thickness=-1)
                    global_frontier_list.append((frontier, 'goto','part'))
                    bank.add(goto_color_list, goto_agent_state_list, data_type='goto')
                    goto_color_list=[]
                    goto_agent_state_list = []

                if not is_final_decision and float(np.linalg.norm(np.array(agent.get_state().position) - np.array(target_position)))<0.25:
                    visited_frontier_set.add(tuple(np.round(target_position, 1)))
                    rot=0
                if is_final_decision :
                    break
        agent_state = agent.get_state()
        view_points = [
                    view_point["agent_state"]["position"]
                    for goal in goals
                    for view_point in goal["view_points"]
        ]
        # computer start end geodesic distance
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = start_position
        path.requested_ends = view_points
        # log_message(view_points)
        if path_finder.find_path(path):
            start_end_geo_distance = path.geodesic_distance
        else:
            log_message("goal is not navigatable",log_file_path,timestamp)
            # print("goal is not navigatable")
            start_end_geo_distance = np.inf
        # compute agent current distance
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = agent_state.position
        path.requested_ends = view_points
        if path_finder.find_path(path):
            agent_end_geo_distance = path.geodesic_distance
        else:
            log_message("agent_end_geo_distance is inf",log_file_path,timestamp)
            agent_end_geo_distance = np.inf
        # compute success rate
        if start_end_geo_distance == np.inf:
            sr = 1
            spl = 1
        elif agent_end_geo_distance == np.inf:
            sr = 0
            spl = 0
        else:
            log_message(f'agent_end_geo_distance:{agent_end_geo_distance} 1',log_file_path,timestamp)
            sr = agent_end_geo_distance <= 1
            spl = sr * start_end_geo_distance / max(start_end_geo_distance, episode_cum_distance)

        if enable_visualization:
             make_video(goals,start_position,path_finder,global_color_list, global_frontier_list,video_path,split,sr,scene_id,cur_episode,object_category,log_file_path,global_state_list,top_down_map,sim)

        result_dict[split].append({'scan_id': scene_id, 'episode_index': episode_index, 'sr': sr, 'spl': spl, 'object_category': object_category})
        log_message(f"Episode finished. SR: {sr}, SPL: {spl}, Steps: {total_steps}, Decision Num: {decision_num}",log_file_path,timestamp)
        with open(output_path, 'w') as f:
            json.dump(result_dict, f)

for split in split_list:
    total_sr = 0
    total_spl = 0
    category_sr_spl = defaultdict(lambda: {'sr': 0, 'spl': 0, 'count': 0})
    for result in result_dict[split]:
        total_sr += result['sr']
        total_spl += result['spl']
        category = result['object_category']
        category_sr_spl[category]['sr'] += result['sr']
        category_sr_spl[category]['spl'] += result['spl']
        category_sr_spl[category]['count'] += 1

    avg_sr = total_sr / len(result_dict[split])
    avg_spl = total_spl / len(result_dict[split])
    print(f"Split: {split}, Average SR: {avg_sr}, Average SPL: {avg_spl}")

    for category, metrics in category_sr_spl.items():
        avg_category_sr = metrics['sr'] / metrics['count']
        avg_category_spl = metrics['spl'] / metrics['count']
        print(f"Category: {category}, Average SR: {avg_category_sr}, Average SPL: {avg_category_spl}")
