python pretraining.py \
    --model_type auto \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --train_file_dir ./datasets/pretrain \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --fp16 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 10 \
    --block_size 1024 \
    --validation_split_percentage 10 \
    --output_dir outputs-pt-mistral-v1 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 256 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ./cache
