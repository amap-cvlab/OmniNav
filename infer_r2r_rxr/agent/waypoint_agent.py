flow_match = False
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

from PIL import Image
from copy import deepcopy

from transformers import AutoProcessor, AutoTokenizer, AutoConfig, Qwen2VLForConditionalGeneration, \
    Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

SAVE_RANDER_IMG = True
PREDICT_SCALE = 0.3
MAX_HISTORY_FRAMES = 20
magic_crop = False
NUM_CURRENT_IMAGE = 3
INPUT_IMG_SIZE = (640, 569)
HISTORY_RESIZE_RATIO = 1 / 4

MODEL_TYPE = 'Waypoint'
NUM_ACTION_TRUNK = 5
NUM_EXCUTE_ACTION_IN_TRUNK = 1


def get_model_name_from_path(model_path):
    return '/'.join(model_path.split('/')[-3:])


def evaluate_agent(config, split_id, dataset, model_path, result_path) -> None:
    env = Env(config.TASK_CONFIG, dataset)

    model_name = get_model_name_from_path(model_path)
    result_path = os.path.join(result_path, model_name)

    agent = Waypoint_Agent(model_path, result_path)

    num_episodes = len(env.episodes)

    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS

    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}

    count = 0

    for _ in trange(num_episodes, desc=config.EVAL.IDENTIFICATION + "-{}".format(split_id)):
        obs = env.reset()
        iter_step = 0
        agent.reset()

        continuse_rotation_count = 0
        continuse_collision_count = 0
        last_dtg = 999
        while not env.episode_over:

            info = env.get_metrics()

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
            with torch.no_grad():
                action = agent.act(obs, info, env.current_episode.episode_id)
            if action[
                'arrive_pred'] > 0:  # or np.max(np.linalg.norm(action['action'],axis=1)) < 0.2:
                action = {"action": "STOP"}
            elif action[
                'arrive_pred'] >= 0.5:  # or np.max(np.linalg.norm(action['action'],axis=1)) < 0.2:
                action = {"action": "STOP"}
            else:
                select_way_point_idx = 0
                print(action['action'])
                print(action['recover_angle'])
                way_point_loc = action['action'][select_way_point_idx, :]
                recover_angle = action['recover_angle'][select_way_point_idx]
                print("way_point_loc", "recover_angle")
                print(way_point_loc)
                print(recover_angle)
                distance = np.linalg.norm(way_point_loc)
                print(distance)
                theta = np.arctan2(-way_point_loc[0], way_point_loc[1])

                action = {"action": "GO_TOWARD_POINT", "action_args": {"theta": recover_angle, "r": distance}}
                print(f'step: {iter_step}, action: {action}')

            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step > EARLY_STOP_STEPS:
                action = {"action": "STOP"}
            obs = env.step(action)
            iter_step += 1
        info = env.get_metrics()
        result_dict = {k: info[k] for k in target_key if k in info}
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
        self.nav_version = 'special_token'
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        if flow_match:
            self.model = self.model.cuda()
            for name in os.listdir(model_path):
                if name.endswith('safetensors'):
                    safe_model_path = os.path.join(model_path, name)
                    state_dict = load_file(safe_model_path)
                    self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model = self.model.cuda()
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
            num_current_image = 3
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

        print("good")
        self.model = QwenModel(model_path)
        self.promt_template = "\n{instruction}"
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
        # current_img_string = "Your current observations is leftside: , frontside: , rightside: "
        current_img_string = "Your current observations is leftside: , rightside: , frontside: "

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
                import sys
                sys.exit(0)
                rgb = rgb.resize((INPUT_IMG_SIZE[0], INPUT_IMG_SIZE[1]))
            rgbs_new.append(rgb)

        if len(self.rgb_list) >= NUM_CURRENT_IMAGE:
            # 历史帧只保留front
            for _ in range(NUM_CURRENT_IMAGE - 1):
                self.rgb_list.pop(-2)
                self.pose_list.pop(-2)
                self.image_indices.pop(-2)
            # pop_idx = [-1, -2]
            # for idx in pop_idx:
            #     self.rgb_list.pop(idx)
            #     self.pose_list.pop(idx)
            #     self.image_indices.pop(idx)

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
            print("??????????")
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
            # self.add_frame([observations['left'], observations['front'], observations['right']], pose)
        else:
            self.add_frame([rgb], pose)

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_vlnce"], rgb.shape[0])
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
        print("question")
        print(navigation_qs)
        wp_pred_, src_arrive_pred, sin_angle, cos_angle = self.model.qwen_infer(navigation_qs)
        end_time = time.time()
        print(f"qwen_infer 耗时: {end_time - start_time} 秒")
        if flow_match:
            print(src_arrive_pred.squeeze())
            cnt = 0
            for cur_arrive in src_arrive_pred.squeeze():
                if cur_arrive.item() > 0.5:
                    cnt += 1
            print(cnt)
            if cnt == 5:
                arrive_pred = 1
            else:
                arrive_pred = 0
        else:
            cnt = 0
            for cur_arrive in src_arrive_pred.squeeze():
                if cur_arrive.item() >= 0:
                    cnt += 1
            # print(cnt)
            if cnt == 5:
                arrive_pred = 1
            else:
                arrive_pred = 0
            # arrive_pred = arrive_pred.item()
            # arrive_pred = arrive_pred.cpu().type(torch.float32).numpy().squeeze()
        wp_pred_ = wp_pred_.cpu().type(torch.float32).numpy().squeeze()
        recover_angle = torch.atan2(sin_angle, cos_angle).detach().cpu().type(torch.float32).numpy().squeeze()

        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]["text"], '{},arrive_pred_{},{},{}'.format(cnt,
                                                                                                                src_arrive_pred.detach().cpu().type(
                                                                                                                    torch.float32).numpy().squeeze(),
                                                                                                                wp_pred_[
                                                                                                                    0],
                                                                                                                recover_angle))
            # img = output_im
            self.topdown_map_list.append(img)
            cv2.imwrite(os.path.join(cur_episode_vis_folder, "{}.png".format(current_frame_index)),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return {"action": wp_pred_, "arrive_pred": arrive_pred, "recover_angle": recover_angle}
