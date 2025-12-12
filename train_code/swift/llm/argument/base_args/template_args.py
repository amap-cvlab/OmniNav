# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from swift.llm import TEMPLATE_MAPPING
from swift.utils import get_logger

logger = get_logger()


@dataclass
class TemplateArguments:
    """
    TemplateArguments class is a dataclass that holds various arguments related to template configuration and usage.

    Args:
        template (Optional[str]): Template type. Default is None, meaning use the template of the model_type.
        system (Optional[str]): Override the default system in the template. Default is None.
        max_length (Optional[int]): Maximum length for the template. Default is None.
        truncation_strategy (Literal): Strategy for truncating the template. Default is 'delete'.
        max_pixels (Optional[int]): Maximum number of pixels for the template. Default is None.
        padding_side: The padding_side when the training batch_size >= 2
        loss_scale (str): Loss scale for training. Default is 'default',
            meaning only calculate the loss of the assistant.
        sequence_parallel_size (int): Size of sequence parallelism. Default is 1.
        use_chat_template (str): Use chat template or default generation template, default True
        template_backend (str): Use swift template or jinja
    """
    template: Optional[str] = field(
        default=None, metadata={'help': f'template choices: {list(TEMPLATE_MAPPING.keys())}'})
    system: Optional[str] = None  # Override the default_system in the template.
    max_length: Optional[int] = None

    truncation_strategy: Literal['delete', 'left', 'right', None] = None
    max_pixels: Optional[int] = None
    resized_img_downsample: bool = False #mark
    resized_history_image: bool = False
    resized_img_fixed: bool = False#mark
    current_img_num: int = 1#mark
    dynamic_resolution: bool = False #mark
    magic_resolution: bool = False
    fixed_height: Optional[int] = None#mark
    fixed_width: Optional[int] = None#mark
    input_waypoint_augment: bool = False#mark
    action_former: bool = False
    if_arrive_list_drop_input_waypoint: bool =False
    if_arrive_list_need_arrive_loss: bool = False
    predict_angle: bool = False
    ar_loss_weight: float = 1.0
    arrive_loss_weight: float = 1.0
    use_arrive_list: bool = False
    cross_attention_flow_match: bool = False
    waypoint_direction_loss: bool = True
    norm_method: Optional[str]=None
    query_action_layer : int = 1
    waypoint_number: int = 5
    action_dim: int = 5
    use_input_waypoint: bool = False
    dataset_not_concat: bool = False
    agent_template: Optional[str] = None
    norm_bbox: Literal['norm1000', 'none', None] = None
    use_chat_template: Optional[bool] = None
    # train
    padding_free: bool = False
    padding_side: Literal['left', 'right'] = 'right'
    loss_scale: str = 'default'
    sequence_parallel_size: int = 1
    # infer/deploy
    response_prefix: Optional[str] = None
    template_backend: Literal['swift', 'jinja'] = 'swift'

    def __post_init__(self):
        if self.template is None and hasattr(self, 'model_meta'):
            self.template = self.model_meta.template
        if self.use_chat_template is None:
            self.use_chat_template = True
        if self.system is not None:
            if self.system.endswith('.txt'):
                assert os.path.isfile(self.system), f'self.system: {self.system}'
                with open(self.system, 'r') as f:
                    self.system = f.read()
            else:
                self.system = self.system.replace('\\n', '\n')
        if self.response_prefix is not None:
            self.response_prefix = self.response_prefix.replace('\\n', '\n')
        if self.truncation_strategy is None:
            self.truncation_strategy = 'delete'

    def get_template_kwargs(self):
        truncation_strategy = self.truncation_strategy
        if truncation_strategy == 'delete':
            truncation_strategy = 'raise'
        remove_unused_columns = self.remove_unused_columns
        if hasattr(self, 'rlhf_type') and self.rlhf_type == 'grpo':
            remove_unused_columns = True
        return {
            'default_system': self.system,
            'max_length': self.max_length,
            'truncation_strategy': truncation_strategy,
            'max_pixels': self.max_pixels,
            'resized_img_downsample': self.resized_img_downsample,#mark
            'resized_history_image': self.resized_history_image,
            'resized_img_fixed': self.resized_img_fixed,#mark
            'dynamic_resolution': self.dynamic_resolution,
            'magic_resolution': self.magic_resolution,
            'current_img_num': self.current_img_num,#mark
            'fixed_height' :self.fixed_height,#mark
            'fixed_width' : self.fixed_width,#mark
            'input_waypoint_augment': self.input_waypoint_augment,#mark
            'action_former': self.action_former,#mark
            'if_arrive_list_drop_input_waypoint': self.if_arrive_list_drop_input_waypoint,
            'if_arrive_list_need_arrive_loss': self.if_arrive_list_need_arrive_loss,
            'predict_angle': self.predict_angle,#mark
            'ar_loss_weight': self.ar_loss_weight,#mark
            'arrive_loss_weight': self.arrive_loss_weight,
            'use_arrive_list': self.use_arrive_list,
            'cross_attention_flow_match' : self.cross_attention_flow_match,
            'waypoint_direction_loss': self.waypoint_direction_loss,#mark
            'norm_method': self.norm_method,
            'query_action_layer': self.query_action_layer,
            'waypoint_number': self.waypoint_number,
            'action_dim' : self.action_dim,
            'use_input_waypoint': self.use_input_waypoint,
            'dataset_not_concat': self.dataset_not_concat,
            'agent_template': self.agent_template,
            'norm_bbox': self.norm_bbox,
            'use_chat_template': self.use_chat_template,
            'remove_unused_columns': remove_unused_columns,
            # train
            'padding_free': self.padding_free,
            'padding_side': self.padding_side,
            'loss_scale': self.loss_scale,
            'sequence_parallel_size': self.sequence_parallel_size,
            # infer/deploy
            'response_prefix': self.response_prefix,
            'template_backend': self.template_backend,
        }
