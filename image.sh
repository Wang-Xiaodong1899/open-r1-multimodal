export WANDB_PROJECT=Qwen-2-VL-2B-Image-GRPO
export WANDB_NAME=multimodal-open-r1-8k-verified

mkdir -p output/$WANDB_PROJECT/$WANDB_NAME

CUDA_VISIBLE_DEVICES=1,3,4,5 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12350" \
    src/open_r1/grpo.py \
    --deepspeed local_scripts/zero3_offload.json \
    --output_dir output/$WANDB_PROJECT/$WANDB_NAME \
    --model_name_or_path /data/wangxd/models/Qwen2-VL-2B-Instruct \
    --dataset_name /home/user/wangxd/open-r1-multimodal/data/multimodal-open-r1-8k-verified \
    --max_prompt_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 50 \
    --max_pixels 2359296 \
    --save_total_limit 8 \
    --save_only_model true