# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, Literal, Optional

from ..utils import Processor
from .base import Template
from .template_meta import TemplateMeta

TEMPLATE_MAPPING: Dict[str, TemplateMeta] = {}


def register_template(template_meta: TemplateMeta, *, exist_ok: bool = False) -> None:
    template_type = template_meta.template_type
    if not exist_ok and template_type in TEMPLATE_MAPPING:
        raise ValueError(f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.')
    TEMPLATE_MAPPING[template_type] = template_meta


def get_template(
    template_type: str,
    processor: Processor,
    default_system: Optional[str] = None,
    max_length: Optional[int] = None,
    *,
    truncation_strategy: Literal['raise', 'left', 'right'] = 'raise',
    max_pixels: Optional[int] = None,  # h * w
    resized_img_fixed: bool = False,#mark
    dynamic_resolution: bool = False,
    magic_resolution: bool = False,
    resized_img_downsample: bool = False,#mark
    resized_history_image: bool = False,#mark
    fixed_height: int = None,#mark
    current_img_num: int = 1,
    fixed_width: int = None,#mark
    input_waypoint_augment: bool = False,#mark
    action_former: bool = False,#mark
    if_arrive_list_drop_input_waypoint: bool = False,
    if_arrive_list_need_arrive_loss: bool = False,
    predict_angle: bool = False,
    ar_loss_weight: float = 1.0,
    arrive_loss_weight: float = 1.0,
    use_arrive_list: bool = False,
    cross_attention_flow_match: bool = False,
    waypoint_direction_loss: bool = True,
    norm_method: Optional[str] =None,
    query_action_layer: int = 1,
    waypoint_number: int = 5,
    action_dim: int = 5,
    use_input_waypoint: bool = False,
    dataset_not_concat: bool = False,
    agent_template: Optional[str] = None,
    norm_bbox: Literal['norm1000', 'none', None] = None,
    use_chat_template: bool = True,
    remove_unused_columns: bool = True,
    # train
    padding_free: bool = False,
    padding_side: Literal['left', 'right'] = 'right',
    loss_scale: str = 'default',
    sequence_parallel_size: int = 1,
    # infer/deploy
    response_prefix: Optional[str] = None,
    template_backend: Literal['swift', 'jinja'] = 'swift',
) -> 'Template':
    template_meta = TEMPLATE_MAPPING[template_type]
    template_cls = template_meta.template_cls
    return template_cls(
        processor,
        template_meta,
        default_system,
        max_length,
        truncation_strategy=truncation_strategy,
        max_pixels=max_pixels,
        resized_img_downsample = resized_img_downsample,#mark
        resized_history_image = resized_history_image,#mark
        resized_img_fixed= resized_img_fixed,#mark
        dynamic_resolution = dynamic_resolution,
        magic_resolution = magic_resolution,
        fixed_height = fixed_height,#mark
        current_img_num = current_img_num,#mark
        fixed_width = fixed_width,#mark
        input_waypoint_augment = input_waypoint_augment,#mark
        action_former = action_former,#mark
        if_arrive_list_drop_input_waypoint = if_arrive_list_drop_input_waypoint,
        if_arrive_list_need_arrive_loss = if_arrive_list_need_arrive_loss,
        use_input_waypoint = use_input_waypoint,
        predict_angle = predict_angle,
        ar_loss_weight = ar_loss_weight,
        arrive_loss_weight = arrive_loss_weight,
        use_arrive_list = use_arrive_list,
        cross_attention_flow_match = cross_attention_flow_match,
        waypoint_direction_loss = waypoint_direction_loss,
        norm_method = norm_method,
        query_action_layer = query_action_layer,
        waypoint_number = waypoint_number,
        action_dim = action_dim,
        dataset_not_concat = dataset_not_concat,
        agent_template=agent_template,
        norm_bbox=norm_bbox,
        use_chat_template=use_chat_template,
        remove_unused_columns=remove_unused_columns,
        # train
        padding_free=padding_free,
        padding_side=padding_side,
        loss_scale=loss_scale,
        sequence_parallel_size=sequence_parallel_size,
        # infer/deploy
        response_prefix=response_prefix,
        template_backend=template_backend,
    )


def get_template_meta(template_type: str) -> TemplateMeta:
    return TEMPLATE_MAPPING[template_type]
