#!/bin/bash

PATTERN="all"   # 全量测试集"all"或者采样测试集"sampled"
MODEL_PATH="./data/checkpoint" # replace your trained checkpoint
NAME="task_name" # 任务名字
TYPE='A-star' # 点到点导航采用A星 "A-star" or "point-goal"

python ./infer_ovon_slowfast/run_nav_ovon_omni.py \
  --pattern "${PATTERN}" \
  --model_path "${MODEL_PATH}" \
  --name "${NAME}" \
  --fast_type "${TYPE}" \
