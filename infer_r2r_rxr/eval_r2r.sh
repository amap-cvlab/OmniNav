#!/bin/bash

CHUNKS=5
MODEL_PATH="./data/checkpoint-x"  #replace the checkpoint path here
GPUS=(0 2 3 5 6)

#R2R
CONFIG_PATH="./data/r2r_pano.yaml"
SAVE_PATH="./data/result_r2r"

for IDX in $(seq 0 $((CHUNKS-1))); do
    REAL_GPU=${GPUS[$(( IDX % ${#GPUS[@]} ))]}
    echo "Running chunk $IDX on GPU $REAL_GPU"
    sleep $(( IDX * 2 )) 
    CUDA_VISIBLE_DEVICES=$REAL_GPU nohup python -u run_infer.py \
        --exp-config $CONFIG_PATH \
        --split-num $CHUNKS \
        --split-id $IDX \
        --model-path $MODEL_PATH \
        --result-path $SAVE_PATH &
done

wait

