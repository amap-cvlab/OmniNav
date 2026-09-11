#!/usr/bin/env bash

# 路径仅作示例，请替换为本地模型和包含 norm 统计量的 Flow 训练数据。
data="./data/waypoint_train_demo_flow.jsonl"
model_path="./data/OmniNav_Flow"
output_dir="./data/output_flow"

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model "$model_path" \
    --train_type full \
    --dataset "$data" \
    --torch_dtype bfloat16 \
    --ddp_find_unused_parameters true \
    --freeze_aligner false \
    --freeze_llm false \
    --freeze_vit true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --norm_method min_max \
    --action_dim 5 \
    --cross_attention_flow_match true \
    --resized_img_downsample false \
    --resized_history_image true \
    --predict_angle true \
    --waypoint_number 5 \
    --use_arrive_list true \
    --resized_img_fixed false \
    --current_img_num 3 \
    --dynamic_resolution true \
    --dataset_not_concat true \
    --use_input_waypoint false \
    --action_former false \
    --waypoint_direction_loss false \
    --lazy_tokenize true \
    --learning_rate 8.62e-06 \
    --split_dataset_ratio 0.0 \
    --dataset_num_proc 8 \
    --truncation_strategy delete \
    --gradient_accumulation_steps 1 \
    --save_steps 500 \
    --logging_steps 5 \
    --max_length 8192 \
    --lr_scheduler_type cosine \
    --output_dir "$output_dir" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --add_version \
    --save_only_model true \
    --attn_impl flash_attn
