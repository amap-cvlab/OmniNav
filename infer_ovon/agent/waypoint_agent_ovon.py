import json
import numpy as np

try:
    from habitat import Env
    from habitat.core.agent import Agent
    import imageio
    from habitat.utils.visualizations import maps
except:
    pass
from tqdm import trange
import os
import torch
import cv2
import time
from scipy.spatial.transform import Rotation as R
from safetensors.torch import load_file
import random
from PIL import Image
from copy import deepcopy
from transformers import AutoProcessor, AutoTokenizer, AutoConfig, Qwen2VLForConditionalGeneration, \
    Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

SAVE_RANDER_IMG = True
PREDICT_SCALE = 0.3
MAX_HISTORY_FRAMES = 20
flow_match = False
NUM_CURRENT_IMAGE = 3
INPUT_IMG_SIZE = (640, 569)
HISTORY_RESIZE_RATIO = 1 / 4

MODEL_TYPE = 'Waypoint'

NUM_ACTION_TRUNK = 5
NUM_EXCUTE_ACTION_IN_TRUNK = 1

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


def get_model_name_from_path(model_path):
    return '/'.join(model_path.split('/')[-3:])


def evaluate_agent_ovon(config, split_id, dataset, model_path, result_path) -> None:
    env = Env(config, dataset)
    # obs = env.reset()
    model_name = get_model_name_from_path(model_path)
    result_path = os.path.join(result_path, model_name)

    agent = Waypoint_Agent(model_path, result_path)

    num_episodes = len(env.episodes)

    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS

    target_key = {"distance_to_goal", "success", "spl"}

    count = 0
    is_collision = False
    for _ in trange(num_episodes, desc=config.EVAL.IDENTIFICATION + "-{}".format(split_id)):
        obs = env.reset()
        scene_name = os.path.basename(env.current_episode.scene_id).split('.')[0]
        env.current_episode.episode_id = f'{scene_name}_{env.current_episode.episode_id}'
        iter_step = 0
        agent.reset()

        target_object = env.current_episode.object_category

        instruction = random.choice(obj_goal_template).format(target_object)

        continuse_rotation_count = 0
        continuse_collision_count = 0
        last_dtg = 999
        while not env.episode_over:
            info = env.get_metrics()

            if info['collisions'] is not None and info['collisions']['is_collision']:
                is_collision = True
                # break

            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count = 0
                continuse_collision_count = 0
            else:
                continuse_rotation_count += 1

            obs['pose'] = {'position': env._sim.get_agent_state().position.tolist(),
                           'rotation': [env._sim.get_agent_state().rotation.w,
                                        env._sim.get_agent_state().rotation.x,
                                        env._sim.get_agent_state().rotation.y,
                                        env._sim.get_agent_state().rotation.z]}
            obs["instruction"] = {"text": instruction}
            with torch.no_grad():
                action = agent.act(obs, info, env.current_episode.episode_id)
            if MODEL_TYPE == 'Action' or MODEL_TYPE == 'ActionTrunk' or MODEL_TYPE == 'ActionTrunkV2':
                if continuse_rotation_count > 0 and action['action'] == 1:
                    continuse_collision_count += 1
                else:
                    continuse_collision_count = 0

                if continuse_collision_count > 5:
                    # 随机2，3一次
                    action = {"action": random.randint(2, 3)}
                    print(
                        f'episode: {env.current_episode.episode_id} because of continuse_rotation_count={continuse_rotation_count} and continuse_collision_count={continuse_collision_count}, use random action: {action}')
                    continuse_collision_count = 0
                pass

            else:
                if action['arrive_pred'] > 0:
                    action = {"action": 0}
                else:
                    select_way_point_idx = 0
                    way_point_loc = action['action'][select_way_point_idx, :]
                    recover_angle = action['recover_angle'][select_way_point_idx]
                    distance = np.linalg.norm(way_point_loc)
                    action = {"action": "GO_TOWARD_POINT", "action_args": {"theta": recover_angle, "r": distance}}
                print(f'step: {iter_step}, action: {action}')

            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step > EARLY_STOP_STEPS:
                action = {"action": 0}
            obs = env.step(action)
            iter_step += 1
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["is_collision"] = is_collision
        result_dict["id"] = env.current_episode.episode_id
        count += 1

        with open(
                os.path.join(os.path.join(result_path, "log"), "stats_{}.json".format(env.current_episode.episode_id)),
                "w") as f:
            json.dump(result_dict, f, indent=4)


class QwenModel():
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        self.nav_version = 'special_token'
        if config.model_type == 'qwen2_vl':
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
            safe_model_path = os.path.join(model_path, 'model.safetensors')
            state_dict = load_file(safe_model_path)
            weight = state_dict['query_action']
            qma_in_bias = state_dict['query_multihead_attn.in_proj_bias']
            qma_in_proj = state_dict['query_multihead_attn.in_proj_weight']
            qma_out_bias = state_dict['query_multihead_attn.out_proj.bias']
            qma_out_proj = state_dict['query_multihead_attn.out_proj.weight']
            self.model.query_action.data.copy_(weight)
            self.model.query_multihead_attn.in_proj_bias.data.copy_(qma_in_bias)
            self.model.query_multihead_attn.in_proj_weight.data.copy_(qma_in_proj)
            self.model.query_multihead_attn.out_proj.bias.data.copy_(qma_out_bias)
            self.model.query_multihead_attn.out_proj.weight.data.copy_(qma_out_proj)
        elif config.model_type == 'qwen2_5_vl':
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
            # self.model = self.model.cuda()
            for name in os.listdir(model_path):
                if name.endswith('safetensors'):
                    safe_model_path = os.path.join(model_path, name)
                    state_dict = load_file(safe_model_path)
                    self.model.load_state_dict(state_dict, strict=False)

    @staticmethod
    def qwen_data_pack(images, user_content):
        content = []
        for idx, image in enumerate(images):
            if idx >= len(images) - NUM_CURRENT_IMAGE:
                cur_json = {
                    "type": "image",
                    "image": image,
                    "resized_height": INPUT_IMG_SIZE[1],
                    "resized_width": INPUT_IMG_SIZE[0],
                }
            else:
                cur_json = {
                    "type": "image",
                    "image": image,
                    "resized_height": INPUT_IMG_SIZE[1] * HISTORY_RESIZE_RATIO,
                    "resized_width": INPUT_IMG_SIZE[0] * HISTORY_RESIZE_RATIO,
                }
            content.append(cur_json)
        content.append({
            "type": "text",
            "text": user_content,
        })
        messages = [
            {
                "role": "user",
                "content": content
            },
        ]
        return messages

    def qwen_infer(self, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + "<|im_end|>"
        if self.nav_version == 'special_token':
            text = text.replace('<|vision_start|><|image_pad|><|vision_end|>', '')
            num_image = len(messages[0]['content']) - 1
            num_current_image = NUM_CURRENT_IMAGE
            num_history_image = num_image - num_current_image

            history_img_str = ''.join(['<|vision_start|><|image_pad|><|vision_end|>'] * num_history_image)
            history_str_pos = text.rfind('Your historical pictures are: ') + len('Your historical pictures are: ')
            text = text[:history_str_pos] + history_img_str + text[history_str_pos:]

            text = text.replace('leftside: ', 'leftside: <|vision_start|><|image_pad|><|vision_end|>')
            text = text.replace('rightside: ', 'rightside: <|vision_start|><|image_pad|><|vision_end|>')
            text = text.replace('frontside: ', 'frontside: <|vision_start|><|image_pad|><|vision_end|>')
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        if flow_match == True:
            norm = [{"min": [
                         [-0.49142804741859436, -0.018926994875073433, -0.5000011853675626, 0.8660246981163404, 0.0],
                         [-0.8506758809089646, -0.11684392392635345, -0.5176391471000088, -0.36602701582911296, 0.0],
                         [-0.9391180276870728, -0.262770414352417, -0.5176390363234377, -0.5000015591363245, 0.0],
                         [-0.9319084137678146, -0.5872985124588013, -0.5176390363234377, -0.5176391890893195, 0.0],
                         [-0.9333658218383789, -0.8579317331314087, -0.5176390363233605, -0.5176391431200632, 0.0]],
                     "max": [[0.8222980499267578, 1.1485368013381958, 0.5000012222510074, 1.0, 1.0],
                             [0.8579317331314087, 1.0390557050704985, 0.5176391335634103, 0.13397477820902748, 1.0],
                             [0.9584183096885622, 0.9541159868240356, 0.5176391335632885, 0.3660255949191993, 1.0],
                             [0.9442337155342072, 0.9441415071487427, 0.5176391335631186, 0.5000004173778672, 1.0],
                             [0.9610724449157715, 0.9491362571716309, 0.5176391335630393, 0.5176390878671062, 1.0]]}]
            wp_pred, arrive_pred, sin_angle, cos_angle = self.model.forward(**inputs, norm=norm, action_former=True,
                                                                            gt_waypoints=0, train=False,
                                                                            train_branch=['continue'])
        else:
            wp_pred, arrive_pred, sin_angle, cos_angle = self.model.forward(**inputs, action_former=True,
                                                                            gt_waypoints=0, train=False,
                                                                            train_branch=['continue'])
        return wp_pred * PREDICT_SCALE, arrive_pred, sin_angle, cos_angle


class Waypoint_Agent():
    def __init__(self, model_path, result_path, require_map=True):

        print("Initialize Qwen")

        self.result_path = result_path
        self.require_map = require_map

        if not self.result_path is None:
            os.makedirs(self.result_path, exist_ok=True)
            os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
            os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)
            os.makedirs(os.path.join(self.result_path, "map_vis"), exist_ok=True)
            os.makedirs(os.path.join(self.result_path, "render_img"), exist_ok=True)

        self.model = QwenModel(model_path)
        if flow_match:
            self.promt_template = """You are an autonomous navigation robot. You will get a task with historical pictures and current pictures you see.
            Based on these information, you need to decide your next {num_action_trunck} actions, which could involve <|left|>,<|right|>,<|forward|>. If you finish your mission, output <|stop|>. Here are some examples: <|left|><|forward|><|forward|><|stop|>, <|forward|><|forward|><|forward|><|left|><|forward|> or <|stop|>
            # Your historical pictures are: {history_img_string}
            # {current_img_string}
            # Your mission is: {instruction}<|NAV|>"""
        else:
            self.promt_template = """You are an autonomous navigation robot. You will get a task with historical pictures and current pictures you see.
            Based on these information, you need to decide your next {num_action_trunck} actions, which could involve <|left|>,<|right|>,<|forward|>. If you finish your mission, output <|stop|>. Here are some examples: <|left|><|forward|><|forward|><|stop|>, <|forward|><|forward|><|forward|><|left|><|forward|> or <|stop|>
            # Your historical pictures are: {history_img_string}
            # {current_img_string}
            # Your mission is: {instruction}<|NAV|>\nOutput the waypoint"""
        print("Initialization Complete")

        self.last_infer_result = ''
        self.history_rgb_tensor = None

        self.rgb_list = []
        self.pose_list = []
        self.topdown_map_list = []

        self.count_id = 0
        self.reset()

    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1] + 5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)

        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image

    def reset(self):

        if self.require_map:
            if len(self.topdown_map_list) != 0:
                output_video_path = os.path.join(self.result_path, "video", "{}.gif".format(self.episode_id))

                imageio.mimsave(output_video_path, self.topdown_map_list)

        self.history_rgb_tensor = None
        self.transformation_list = []
        self.rgb_list = []
        self.pose_list = []
        self.image_indices = []
        self.topdown_map_list = []
        self.total_frame_count = 0
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []

        self.first_forward = False

    def pose_to_matrix(self, pose):
        ## 高斯仿真坐标系 前坐上
        if isinstance(pose, np.ndarray):
            rotation_matrix = pose[:3, :3]
            position = pose[:3, 3]
            rot_normal_raw = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            rotation_matrix = rotation_matrix @ rot_normal_raw
            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix
            pose[:3, 3] = position

        else:  ## habitat仿真坐标系
            position = np.array(pose['position'])
            rotation = np.array(pose['rotation'])

            rotation_matrix = R.from_quat(rotation[[1, 2, 3, 0]]).as_matrix()

            rot_normal_raw = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            rotation_matrix = rotation_matrix @ rot_normal_raw
            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix
            pose[:3, 3] = position
        return pose

    def transform_poses_to_local(self, current_pose, input_poses):
        current_pose = self.pose_to_matrix(current_pose)
        current_pose_inv = np.linalg.inv(current_pose)

        output_poses = [current_pose_inv @ self.pose_to_matrix(pose) for pose in input_poses]
        return output_poses

    def generate_infer_prompt(self, instruction):
        cur_prompt = deepcopy(self.promt_template)

        input_poses = deepcopy(self.pose_list)
        local_poses = self.transform_poses_to_local(self.pose_list[-1], input_poses)

        input_positions = [[pose[0, 3], pose[2, 3]] for pose in local_poses]
        images = self.rgb_list

        history_pose_strings = ['<{:.3f},{:.3f}>'.format(pose[0], pose[1]) for pose in input_positions]
        history_pose_string = ",".join(history_pose_strings)

        history_img_string = ''
        current_img_string = "Your current observations is leftside: , frontside: , rightside: "

        cur_prompt = cur_prompt.format(instruction=instruction, history_pose_string=history_pose_string,
                                       step_scale=PREDICT_SCALE, num_action_trunck=NUM_ACTION_TRUNK,
                                       current_img_string=current_img_string, history_img_string=history_img_string)

        return self.model.qwen_data_pack(images, cur_prompt)

    def add_frame(self, rgbs, pose):
        """
        添加新的RGB图像和pose，实现智能的历史帧管理
        1. 限制最大图片数量为MAX_HISTORY_FRAMES
        2. 保证第一张图像一直在队列中
        3. 超过MAX_HISTORY_FRAMES时，进行类似均匀采样的重新采样
        """

        # TODO: pose需要按照外参管理
        # 计时
        start_time = time.time()
        # 将新的rgb和pose添加到完整列表中
        rgbs_new = []
        for rgb in rgbs:
            if isinstance(rgb, np.ndarray):
                rgb_img = Image.fromarray(rgb)
                # 统一调整大小到模型输入尺寸
                rgb = rgb_img.resize((INPUT_IMG_SIZE[0], INPUT_IMG_SIZE[1]))
            else:
                rgb = rgb.resize((INPUT_IMG_SIZE[0], INPUT_IMG_SIZE[1]))
            rgbs_new.append(rgb)

        rgbs_new = [rgbs_new[idx] for idx in [0, 2, 1]]  # 0720更新，调整为左中右

        if len(self.rgb_list) >= NUM_CURRENT_IMAGE:
            # 历史帧只保留front TODO:目前只支持3张，如果变成多张需要修改
            pop_idx = [-1, -2]
            for idx in pop_idx:
                self.rgb_list.pop(idx)
                self.pose_list.pop(idx)
                self.image_indices.pop(idx)

        self.rgb_list.extend(rgbs_new)
        self.pose_list.extend([pose] * len(rgbs_new))
        self.image_indices.extend([self.total_frame_count] * len(rgbs_new))
        self.total_frame_count += 1
        if len(self.rgb_list) > NUM_CURRENT_IMAGE:
            self.rgb_list[-1 - NUM_CURRENT_IMAGE] = self.rgb_list[-1 - NUM_CURRENT_IMAGE].resize(
                (int(INPUT_IMG_SIZE[0] * HISTORY_RESIZE_RATIO), int(INPUT_IMG_SIZE[1] * HISTORY_RESIZE_RATIO)))

        # 如果超过最大历史帧数，需要重新采样
        if len(self.rgb_list) > MAX_HISTORY_FRAMES + NUM_CURRENT_IMAGE:
            # 基于self.image_indices 移除第一个间距最小的帧
            min_interval_idx = np.argmin(np.diff(self.image_indices[:-NUM_CURRENT_IMAGE]))
            self.rgb_list.pop(min_interval_idx + 1)
            self.pose_list.pop(min_interval_idx + 1)
            self.image_indices.pop(min_interval_idx + 1)

        print('current image_indices: {}'.format(self.image_indices))
        end_time = time.time()
        print(f"add_frame 耗时: {end_time - start_time} 秒")

    def save_rgb(self, path):
        output_dir = os.path.join(path, str(self.count_id), str(self.total_frame_count))
        os.makedirs(output_dir, exist_ok=True)
        for idx, rgb in enumerate(self.rgb_list):
            output_img_path = os.path.join(output_dir, "{}.png".format(idx))
            rgb.save(output_img_path)

        with open(os.path.join(output_dir, "last_infer_result.txt"), "w") as f:
            f.write(self.last_infer_result)

    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        cur_episode_folder = os.path.join(self.result_path, "render_img", str(episode_id))
        os.makedirs(cur_episode_folder, exist_ok=True)

        cur_episode_vis_folder = os.path.join(self.result_path, "map_vis", str(episode_id))
        os.makedirs(cur_episode_vis_folder, exist_ok=True)

        if self.model.nav_version == 'special_token':
            rgb = observations["front"]
            pose = observations["pose"]
        else:
            rgb = observations["rgb"]
            pose = observations["pose"]

        # 保存当前图像（使用即将分配的原始帧索引）
        current_frame_index = self.total_frame_count
        output_img_path = os.path.join(cur_episode_folder, "{}.png".format(current_frame_index))
        if SAVE_RANDER_IMG:
            cv2.imwrite(output_img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # 使用add_frame方法添加新的图像和pose
        if self.model.nav_version == 'special_token':
            self.add_frame([observations['left'], observations['right'], observations['front']], pose)
        else:
            self.add_frame([rgb], pose)

        if self.require_map:
            if 'top_down_map_vlnce' in info:
                top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_vlnce"], rgb.shape[0])
            else:
                top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0:
            temp_action = self.pending_action_list.pop(0)

            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]["text"],
                                   "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
                cv2.imwrite(os.path.join(cur_episode_vis_folder, "{}.png".format(current_frame_index)),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            return {"action": temp_action}

        start_time = time.time()
        navigation_qs = self.generate_infer_prompt(observations["instruction"]["text"])
        end_time = time.time()
        print(f"generate_infer_prompt 耗时: {end_time - start_time} 秒")

        start_time = time.time()
        wp_pred_, src_arrive_pred, sin_angle, cos_angle = self.model.qwen_infer(navigation_qs)
        end_time = time.time()
        print(f"qwen_infer 耗时: {end_time - start_time} 秒")
        cnt = 0
        if flow_match:
            cnt = 0
            for cur_arrive in src_arrive_pred.squeeze():
                if cur_arrive.item() > 0.5:
                    cnt += 1
        else:

            for cur_arrive in src_arrive_pred.squeeze():
                if cur_arrive.item() >= 0:
                    cnt += 1
        print(cnt)
        if cnt == 5:
            arrive_pred = 1
        else:
            arrive_pred = 0
        wp_pred_ = wp_pred_.cpu().type(torch.float32).numpy().squeeze()
        recover_angle = torch.atan2(sin_angle, cos_angle).detach().cpu().type(torch.float32).numpy().squeeze()

        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]["text"], '{},arrive_pred_{},{},{}'.format(cnt,
                                                                                                                src_arrive_pred.detach().cpu().type(
                                                                                                                    torch.float32).numpy().squeeze(),
                                                                                                                wp_pred_[
                                                                                                                    0],
                                                                                                                recover_angle))

            self.topdown_map_list.append(img)
            cv2.imwrite(os.path.join(cur_episode_vis_folder, "{}.png".format(current_frame_index)),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return {"action": wp_pred_, "arrive_pred": arrive_pred, "recover_angle": recover_angle}
