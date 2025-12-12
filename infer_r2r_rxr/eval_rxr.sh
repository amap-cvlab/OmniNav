#!/bin/bash

CHUNKS=6
MODEL_PATH="./data/checkpoint-x"
GPUS=(1 5 6 7 8 9)

#RxR
CONFIG_PATH="./data/rxr_pano.yaml"
SAVE_PATH="./data/result_rxr"

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

