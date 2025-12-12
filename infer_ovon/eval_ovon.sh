#!/bin/bash

#!/bin/bash

CHUNKS=1
GPUS=(0)
MODEL_PATH="./data/checkpoint-x"  #replace the checkpoint path here

#ovon pano
CONFIG_PATH="ovon/configs/ovon_pano.yaml"
SAVE_PATH="./data/result_ovon"


for IDX in $(seq 0 $((CHUNKS-1))); do
    REAL_GPU=${GPUS[$(( IDX % ${#GPUS[@]} ))]}
    echo "Running chunk $IDX on GPU $REAL_GPU"
    sleep $(( IDX * 2 ))
    CUDA_VISIBLE_DEVICES=$REAL_GPU nohup python -u run_infer_ovon.py \
        --exp-config $CONFIG_PATH \
        --split-num $CHUNKS \
        --split-id $IDX \
        --model-path $MODEL_PATH \
        --result-path $SAVE_PATH &
done

wait
