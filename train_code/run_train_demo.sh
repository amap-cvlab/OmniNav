data = './data/waypoint_train_demo.json'
model_path="./data/Qwen2___5-VL-3B-Instruct"
output_dir="./data/output"

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --deepspeed zero0 \
    --model $model_path \
    --train_type full \
    --dataset $data \
    --torch_dtype bfloat16 \
    --freeze_aligner false \
    --freeze_llm false \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --resized_img_downsample false \
    --resized_history_image true \
    --input_waypoint_augment true \
    --if_arrive_list_drop_input_waypoint true \
    --if_arrive_list_need_arrive_loss true \
    --predict_angle true \
    --waypoint_number 5 \
    --use_arrive_list true \
    --resized_img_fixed false \
    --current_img_num 3 \
    --dynamic_resolution true \
    --dataset_not_concat true \
    --use_input_waypoint true \
    --action_former true \
    --waypoint_direction_loss false \
    --lazy_tokenize true \
    --learning_rate 1e-5 \
    --split_dataset_ratio 0.0 \
    --dataset_num_proc 8 \
    --truncation_strategy delete \
    --gradient_accumulation_steps 1 \
    --freeze_vit false \
    --save_steps 10000 \
    --logging_steps 5 \
    --max_length 8096 \
    --lr_scheduler_type cosine \
    --output_dir $output_dir \
    --warmup_ratio 0 \
    --dataloader_num_workers 8 \
    --add_version \
    --save_only_model true \
    --attn_impl flash_attn

