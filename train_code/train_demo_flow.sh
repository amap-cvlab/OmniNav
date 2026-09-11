#!/usr/bin/env bash
set -euo pipefail

# 根据 expert completion 训练配置整理；仅加载模型权重，不恢复优化器和数据游标。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/data/OmniNav_Flow}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/data/output_flow}"

# 单数据集通过 DATA_PATH 指定；多数据集作为脚本参数传入，支持带引号的路径#采样数。
# 必须使用包含 norm 统计量的 Flow 数据，不能直接使用 waypoint_train_demo.json。
if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    : "${DATA_PATH:?请设置 DATA_PATH 为 Flow 训练数据路径，或通过脚本参数传入多个数据集。}"
    DATASETS=("$DATA_PATH")
fi

# 优先使用仓库内的定制 swift；多卡启动时 NPROC_PER_NODE 应与可见 GPU 数量一致。
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# demo 默认每卡 batch size 为 2；源脚本为 16，可通过环境变量覆盖。
"${PYTHON_BIN:-python}" -m swift.cli.main sft \
    --model "$MODEL_PATH" \
    --train_type full \
    --dataset "${DATASETS[@]}" \
    --torch_dtype bfloat16 \
    --ddp_find_unused_parameters true \
    --freeze_aligner false \
    --freeze_llm false \
    --freeze_vit "${FREEZE_VIT:-true}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
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
    --learning_rate "${LEARNING_RATE:-8.62e-06}" \
    --split_dataset_ratio 0.0 \
    --dataset_num_proc 8 \
    --truncation_strategy delete \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
    --save_steps 500 \
    --logging_steps 5 \
    --max_length 8192 \
    --lr_scheduler_type cosine \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --add_version \
    --save_only_model true \
    --attn_impl "${ATTN_IMPL:-flash_attn}"
