
import numpy as np
from PIL import Image        
from qwen_vl_utils import smart_resize
min_pixels = 56 * 56
max_pixels = 4480 * 4480


import numpy as np
from scipy.spatial.transform import Rotation as R
import re
import json

def parse_response(response_text):
    pattern = re.compile(r"coordinate (\[.*?\])")
    match = pattern.search(response_text)
    if match:
        coord_str = match.group(1)
        try:
            coord_list = json.loads(coord_str)
            
            if isinstance(coord_list, list):
                if "found" in response_text:
                    return coord_list, True
                else:
                    return coord_list, False
            else:
                return None, False
        except json.JSONDecodeError:
            return None, False
    return None, False

def process_vision(colorlist):
    image_inputs = []
    video_inputs = []
    for image in colorlist:  
            image = Image.fromarray(image)
            image = image.resize((486,420))
            width, height = image.size
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=14,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            image = image.resize((resized_width, resized_height))
            image_inputs.append(image)

    return image_inputs, video_inputs
import numpy as np
import quaternion

from scipy.spatial.transform import Rotation as R

def transform_to_local_frame(world_point, agent_world_coord, agent_world_quat):
    world_point = np.array(world_point)
    agent_world_coord = np.array(agent_world_coord)
    relative_point = world_point - agent_world_coord
    quat_xyzw = [agent_world_quat['x'], agent_world_quat['y'], agent_world_quat['z'], agent_world_quat['w']]
    agent_rotation = R.from_quat(quat_xyzw)
    local_point = agent_rotation.apply(relative_point, inverse=True)
    local_point = local_point.round(2).tolist()
    res = [local_point[0] ,local_point[1]* -1, local_point[2] * -1]
    return res

def transform_from_local_frame(local_point, agent_world_coord, agent_world_quat):

    local_point = np.array([
        local_point[0] ,
        local_point[1]* -1,
        local_point[2] * -1
    ])
    agent_world_coord = np.array(agent_world_coord)
    quat_xyzw = [agent_world_quat['x'], agent_world_quat['y'], agent_world_quat['z'], agent_world_quat['w']]
    agent_rotation = R.from_quat(quat_xyzw)
    relative_point_in_world = agent_rotation.apply(local_point)
    reconstructed_world_point = relative_point_in_world + agent_world_coord

    return reconstructed_world_point



obj_goal_template = [
"Find a {} in your immediate surroundings and stop when you see one.", 
"Explore the area until you locate a {}. Stop when you've reached its location.",
"Move through the environment to discover a {}. Your task is complete when you're directly facing it.",
"Navigate to any visible {}. Stop immediately upon successful discovery.",
"Search for an instance of {} within this space. Terminate navigation once you've positioned yourself within arm's reach of it.",
"Survey the surroundings until you identify a {}. Stop navigating as soon as you are positioned directly in front of it",
"Roam through the space until a {} is spotted. Terminate navigation the moment you’re certain you’re facing it.",
"Go to the {}, then stop at the front of it.",
"Move to the nearst {}, then stop",
"Navigate to a nearst {}, then stop over there.",
"Get close to the {}, then stop",
"Could you help me find a {}? Show me the way"]

import random



def getresult(qwen, processor,bank, current_frontiers, decision_agent_state, object_category, decision_num,
               visited_frontier_set):
    ref_coord = decision_agent_state.position
    ref_quat_dict = {
        'x': decision_agent_state.rotation.x, 'y': decision_agent_state.rotation.y,
        'z': decision_agent_state.rotation.z, 'w': decision_agent_state.rotation.w
    }
    all_prompt_images = []
    spin_image,spin_state=bank.get_spin_data()  
    all_prompt_images.extend(spin_image)
    spin_images_content ="1: These four images show a 360-degree panoramic view around Observer's perspective,position is all [0.00,0.00], taken at 90-degree intervals: <image><image><image><image>"
    main_images_content ="2: This is the reference image from Observer's perspective for all coordinates: <image>"




    local_to_global_map = {}
    frontier_parts = ['3: The coordinates of the explorable frontiers are: ']
    for frontier_coord in current_frontiers:
        local_coord = transform_to_local_frame(frontier_coord, ref_coord, ref_quat_dict)
        x, z = local_coord[0], local_coord[2]
        local_to_global_map[(x, z)] = frontier_coord
        frontier_parts.append(f"[{x:.2f}, {z:.2f}]")

    frontier_content = ''.join(frontier_parts)


    task_sentence = "instruction: "+random.choice(obj_goal_template).format(object_category)
    user_content = "\n".join([spin_images_content,main_images_content,frontier_content,task_sentence])
    
    current_messages=[]
 
    msg_copy={}

    if len(current_messages) == 0:  
        content = []
        content.append({"type": "image", "image": all_prompt_images})
        content.append({"type": "text", "text":user_content})
        msg_copy['content'] = content
        current_messages.append(msg_copy)
    current_messages[0]['role']='user'
    text = processor.apply_chat_template(
        current_messages, tokenize=False, add_generation_prompt=True
    )
    text=text.replace('<|vision_start|><|image_pad|><|vision_end|>','')
    text=text.replace('<image>','<|vision_start|><|image_pad|><|vision_end|>')

    image_inputs, video_inputs = process_vision(all_prompt_images)
    inputs = processor(
        text=text,
        images=image_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    device = qwen.device 
    inputs = inputs.to(device)
    print(text.count('image_pad'))
    with torch.no_grad():
        generated_ids = qwen.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        parsed_local_target,is_final_decision=parse_response(output_texts[0])
    print(output_texts)
    if parsed_local_target is None:
        if  local_to_global_map is None and current_frontiers is not None:
            return current_frontiers[0] if current_frontiers else None, False
    model_choice_coord = np.array(parsed_local_target)
    
    min_dist = float('inf')
    best_match_global_coord = None
    if is_final_decision:
        model_choice_coord= np.insert(model_choice_coord, 1, 0)
        best_match_global_coord=transform_from_local_frame(model_choice_coord,ref_coord, ref_quat_dict)
    else:
        for local_key, global_coord_val in local_to_global_map.items():
            dist = np.linalg.norm(model_choice_coord - np.array(local_key))
            
            if dist < min_dist:
                min_dist = dist
                best_match_global_coord = global_coord_val
    target_position0 = best_match_global_coord

    return target_position0, is_final_decision,output_texts

def get_result_fast(qwen, processor,bank,global_color_list,global_list,sim,agent,target_position,log_file_path, current_frontiers, decision_agent_state, object_category, decision_num,rot
               ):
    def log_message(message):
        with open(log_file_path, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n") 
    ref_coord = decision_agent_state.position
    ref_quat_dict = {
        'x': decision_agent_state.rotation.x, 'y': decision_agent_state.rotation.y,
        'z': decision_agent_state.rotation.z, 'w': decision_agent_state.rotation.w
    }

    input_pos = []
    all_prompt_images = []
    REPRODUCIBLE_SEED = 42

    all_fwd_indices_before_start = [i for i, act in enumerate(global_color_list) if act[1] == 'goto']
    if rot >50:
        log_message('too many interrupt')
        return '000',global_list[-1],rot
    if check_small_position_change(
        all_fwd_indices_before_start, 
        global_list, 
        threshold=0.1,  
        lookback=20
    ):  
        import habitat_sim
        agent_state = habitat_sim.AgentState()
        rot=rot+1
        while True:
            noise_scale=0.3
            dx = np.random.uniform(-noise_scale, noise_scale)
            dy = np.random.uniform(-noise_scale, noise_scale)


            new_pos=agent.state.position
            new_pos[0]=new_pos[0]+dy
            new_pos[2]=new_pos[2]+dx
            new_pos=np.array(sim.step_filter(agent.state.position,new_pos ))

            if sim.pathfinder.is_navigable(new_pos) and np.linalg.norm(new_pos-agent.state.position)>0.01:
                break
        print('interrupt')
        log_message('interrupt')
        agent_state.position=new_pos
        agent_state.rotation = [decision_agent_state.rotation.x,decision_agent_state.rotation.y,decision_agent_state.rotation.z,decision_agent_state.rotation.w]
        agent.set_state(agent_state)            
        return sim.get_sensor_observations(agent_ids=[0])[0],[transform_from_local_frame(np.insert(agent_state.position, 1, 0), ref_coord, ref_quat_dict)] ,rot
    range_start_idx = 0 
    if len(all_fwd_indices_before_start) >= 12:
        range_start_idx = all_fwd_indices_before_start[-12]
    range_end_idx = len(global_color_list)-2

    indices = np.linspace(range_start_idx, range_end_idx, 4, dtype=int).tolist()
    random.seed(REPRODUCIBLE_SEED)
    goto_image_select = [global_color_list[i][0] for i in indices]
    goto_state_select = [global_list[i] for i in indices]


    all_prompt_images.extend(goto_image_select)
    obervations =  sim.get_sensor_observations(agent_ids=[0])[0]
    all_prompt_images.extend([obervations['color_sensor_left'][:, :, :3]])
    all_prompt_images.extend([obervations['color_sensor'][:, :, :3]])
    all_prompt_images.extend([obervations['color_sensor_right'][:, :, :3]])
    input_pos.extend(goto_state_select)
    input_pos.extend([global_list[-1]])
    input_pos.extend([target_position])   
    content_text ="The following are observation images from the past 4 frames:<image>,<image>,<image>,<image>\n\
        The current tri-view is shown below: leftside:<image>,frontside:<image>,rightside:<image>\n\
        Position coordinates for the past 4 frames:<input_pos1><input_pos2><input_pos3><input_pos4>\n\
        The current observation represents the coordinate: <input_pos5>\n\
        Target position coordinate: <input_target>\n\
        Please predict the position coordinates for the next 5 frames based on the above information.<|NAV|>\nOutput the waypoint"    # --- 3. 构建当前图像部分 <image> ---
    import numpy

    input_pos_local=[]
    for frontier_coord in input_pos:

        local_coord = transform_to_local_frame(frontier_coord.position if type(frontier_coord) != numpy.ndarray else frontier_coord, ref_coord, ref_quat_dict)
        x, z = local_coord[0], local_coord[2]
        input_pos_local.append([x,z])

    current_messages=[]
 
    msg_copy={}

    if len(current_messages) == 0:  
        content = []
        content.append({"type": "image", "image": all_prompt_images})
        content.append({"type": "text", "text":content_text})
        msg_copy['content'] = content
        current_messages.append(msg_copy)
    current_messages[0]['role']='user'
    text = processor.apply_chat_template(
        current_messages, tokenize=False, add_generation_prompt=True
    )
    text=text.replace('<|vision_start|><|image_pad|><|vision_end|>','')
    text=text.replace('<image>','<|vision_start|><|image_pad|><|vision_end|>')

    image_inputs, video_inputs = process_vision(all_prompt_images)

    from transformers import AutoTokenizer

    image_inputs=preprocess(image_inputs)
    inputs = processor(
        text=text,
        images=image_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    device = qwen.device 
    step_scale = 0.3
    input_positions = torch.tensor(input_pos_local,dtype=torch.float32)
    input_positions_scaled = (input_positions / step_scale)[None].to(device)
    inputs['input_waypoints'] = input_positions_scaled


    inputs = inputs.to(device)
    # print(frontier_content)
    print(text.count('image_pad'))
    with torch.no_grad():
        wp_pred, arrive_pred,sin_angle,cos_angle = qwen.forward(**inputs, action_former=True, gt_waypoints=0,train=False,train_branch=['continue'])
    wp_pred = wp_pred.cpu().type(torch.float32).numpy().squeeze()

    recover_angle = torch.atan2(sin_angle, cos_angle).detach().cpu().type(torch.float32).numpy().squeeze()

    select_way_point_idx = 0
    way_point_loc = wp_pred[select_way_point_idx]
    # way_point_coord= np.insert(way_point_loc, 1, 0)
    # pos=transform_from_local_frame(way_point_coord,ref_coord, ref_quat_dict) 
    r=np.linalg.norm(way_point_loc)*step_scale
    pos = rtheta_to_global_coordinates(
        sim, agent,r, recover_angle[0], y_delta=0, dimensionality=3
    )    
    agent_pos = agent.get_state().position
    new_rot = agent.get_state().rotation 

    new_pos = np.array(sim.step_filter(agent_pos, pos))


    if np.any(np.isnan(new_pos)) or not sim.pathfinder.is_navigable(new_pos):
        new_pos = agent_pos
        new_rot, _ = compute_heading_to(agent_pos, pos)
    else:
        new_pos = np.array(sim.pathfinder.snap_point(new_pos))
        if np.any(np.isnan(new_pos)) or not sim.pathfinder.is_navigable(new_pos):
            new_pos = agent_pos

        new_rot, _ = compute_heading_to(agent_pos, pos)

    import habitat_sim
    agent_state = habitat_sim.AgentState()
    agent_state.position = new_pos
    agent_state.rotation = new_rot
    agent.set_state(agent_state)
    # obs = sim.get_observations()
    log_message(f"target_position:{input_pos_local[-1]},wp_pred:{wp_pred[0]},recover_angle:{recover_angle[0]}")
    return sim.get_sensor_observations(agent_ids=[0])[0],[transform_from_local_frame(np.insert(way_point_coord, 1, 0), ref_coord, ref_quat_dict) for  way_point_coord in wp_pred],rot


class Bank:
    def __init__(self, goto_sample_interval=3):
        self.images_spin = []
        self.agent_states_spin = []
        self.images_goto = []
        self.agent_states_goto = []

        self.unsampled_goto_images = []
        self.unsampled_goto_agent_states = []

        self.goto_sample_interval = goto_sample_interval

    def add(self, images, agent_states, data_type='spin'):
        if not images:
            return

        if data_type == 'goto':

            sampled_indices = set(self._get_sampled_indices(len(images)))


            for i, (img, state) in enumerate(zip(images, agent_states)):
                if i in sampled_indices:

                    self.images_goto.append(img)
                    self.agent_states_goto.append(state)
                else:

                    self.unsampled_goto_images.append(img)
                    self.unsampled_goto_agent_states.append(state)
        
        else: 

            self.images_spin.extend(images)
            self.agent_states_spin.extend(agent_states)


    def get_spin_data(self):
        return self.images_spin, self.agent_states_spin
    def get_goto_data(self):
        return self.images_goto, self.agent_states_goto

    def get_unsampled_goto_data(self):
        return self.unsampled_goto_images, self.unsampled_goto_agent_states

    def clear(self):
        self.images_spin.clear()
        self.agent_states_spin.clear()
        self.images_goto.clear()
        self.agent_states_goto.clear()
        self.unsampled_goto_images.clear()
        self.unsampled_goto_agent_states.clear()
        print("Bank memory cleared.")

    def _get_sampled_indices(self, num_images):
        if self.goto_sample_interval==1:
            return list(range(num_images))
        if num_images <= 2:
            return list(range(num_images))

        sampled_indices = {0, num_images - 1}
        step = self.goto_sample_interval + 1
        for i in range(step, num_images - 1, step):
            sampled_indices.add(i)
        
        return sorted(list(sampled_indices))

    def __len__(self):
        return len(self.images)






from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
    quaternion_to_list,
)
def compute_heading_to(
    pos_from, pos_to
):
    """Compute the heading that points from position `pos_from` to position `pos_to`
    in the global XZ coordinate frame.

    Args:
        pos_from: [x,y,z] or [x,z]
        pos_to: [x,y,z] or [x,z]

    Returns:
        heading quaternion as [x, y, z, w]
        heading scalar angle
    """
    delta_x = pos_to[0] - pos_from[0]
    delta_z = pos_to[-1] - pos_from[-1]
    xz_angle = np.arctan2(delta_x, delta_z)
    xz_angle = (xz_angle + np.pi) % (2 * np.pi)
    quat = quaternion_to_list(
        quaternion.from_euler_angles([0.0, xz_angle, 0.0])
    )
    return quat, xz_angle
from datetime import datetime
import torch

def find_first_waypoint_beyond_threshold(wp_pred, threshold=0.15):
    wp_pred = torch.from_numpy(wp_pred)
    if wp_pred.dim() == 3:
        # shape: [batch_size, num_waypoints, 2]
        batch_size = wp_pred.shape[0]
        
        distances = torch.norm(wp_pred, dim=-1)  # [batch_size, num_waypoints]
        

        indices = []
        for i in range(batch_size):
            dist = distances[i]
            mask = dist > threshold
            
            if mask.any():
                idx = torch.nonzero(mask, as_tuple=True)[0][0].item()
            else:
                idx = wp_pred.shape[1] - 1
            
            indices.append(idx)
        
        indices = torch.tensor(indices, device=wp_pred.device)
        return indices
    
    elif wp_pred.dim() == 2:
        # shape: [num_waypoints, 2]
        distances = torch.norm(wp_pred, dim=-1)
        mask = distances > threshold
        
        if mask.any():
            idx = torch.nonzero(mask, as_tuple=True)[0][0].item()
        else:
            idx = wp_pred.shape[0] - 1
        
        return idx
    
    else:
        raise ValueError(f"Unexpected wp_pred shape: {wp_pred.shape}")
import math
import habitat_sim
def rtheta_to_global_coordinates(
    sim,
    agent,
    r,
    theta,
    y_delta,
    dimensionality,
):
    """Maps relative polar coordinates from an agent position to an updated
    agent position. The returned position is not validated for navigability.
    """
    assert dimensionality in [2, 3]
    scene_node = sim.get_agent(0).scene_node
    forward_ax = (
        np.array(scene_node.absolute_transformation().rotation_scaling())
        @ habitat_sim.geo.FRONT
    )
    agent_state = agent.get_state()
    rotation = habitat_sim.utils.quat_from_angle_axis(
        theta, habitat_sim.geo.UP
    )
    move_ax = habitat_sim.utils.quat_rotate_vector(rotation, forward_ax)
    position = agent_state.position + (move_ax * r)
    position[1] += y_delta

    if dimensionality == 2:
        return [position[0], position[2]]
    return position
import numpy as np
def rescale_image_with_long_edge_and_random_scale(image: Image.Image, 
                                                 long_edge=640, 
                                                 scale=1.0):
    w, h = image.size
    
    if w >= h:
        new_w = long_edge
        new_h = int(h * (long_edge / w))
    else:
        new_h = long_edge
        new_w = int(w * (long_edge / h))
    
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    final_w = int(new_w * scale)
    final_h = int(new_h * scale)
    
    if final_w < 1 or final_h < 1:
        final_w = max(1, final_w)
        final_h = max(1, final_h)
    
    image = image.resize((final_w, final_h), Image.Resampling.LANCZOS)
    
    return image
import random
import math
def rescale_image_to_fixed_size(img: Image.Image, height: int, width: int) -> Image.Image:
    import torchvision.transforms as T
    return T.Resize((int(height), int(width)))(img)
def crop_resize_image_magic_resolution(img: Image.Image) -> Image.Image:
    import torchvision.transforms as T
    width = img.width
    height = img.height
    if width !=720 and height !=640:
        return img
    top, bottom = 140, 500

    left, right = 0, img.width

    cropped_img = img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((width, height), Image.BILINEAR)
    
    return resized_img
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor
def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor
def smart_resize(
    height, width, factor, min_pixels, max_pixels
):
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar
def preprocess(images):
    images = [
        crop_resize_image_magic_resolution(img) 
            if idx not in {len(images) - 1, len(images) - 3} 
            else img 
            for idx, img in enumerate(images)
        ]
    current_img_num = 3
    LONG_EDGE = 640
    SCALE_RANGE = (0.7, 1.0)
    # ✅ 在 batch 级别随机选择一个 scale（所有图像共用）
    s = 1.0
    images = [
        rescale_image_with_long_edge_and_random_scale(img, LONG_EDGE, scale = s)
        for img in images
    ]
    if True:
        images = [rescale_image_to_fixed_size(img,int(img.height/4),int(img.width/4)) if idx < len(images)-current_img_num else img for idx,img in enumerate(images)]
    images_new  = []
    size_factor = 28 
    min_pixels = 3136
    max_pixels = 12845056
    for image in images:
        width, height = image.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = image.resize((resized_width, resized_height))
        images_new.append(image)
    return   images_new             

def check_small_position_change(all_fwd_indices_before_start, positions, threshold=0.2, lookback=50):
    if len(all_fwd_indices_before_start) < lookback:
        return False

    recent_indices = all_fwd_indices_before_start[-lookback:]
    

    for i in range(len(recent_indices) - 1):
        idx1 = recent_indices[i]
        idx2 = recent_indices[i + 1]
        
        pos1 = positions[idx1].position
        pos1[1]=0
        pos2 = positions[idx2].position
        pos2[1]=0

        distance = np.linalg.norm(pos2 - pos1)
        

        if distance > threshold:
            return False
    

    return True

import torch
