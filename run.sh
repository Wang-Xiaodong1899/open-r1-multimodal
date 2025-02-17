export WANDB_PROJECT=Qwen-2-VL-2B-Video-GRPO
export WANDB_NAME=llava-video-gemini-0.8k

mkdir -p output/$WANDB_PROJECT/$WANDB_NAME

CUDA_VISIBLE_DEVICES=1,3,4,5 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12352" \
    src/open_r1/grpo.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$WANDB_NAME \
    --model_name_or_path /data/wangxd/models/Qwen2-VL-2B-Instruct \
    --dataset_name xxx \
    --jsonl_path /home/user/wangxd/open-r1-multimodal/data/LLaVA-Video-large-swift-gemini-1.5-filter.jsonl \
    --max_prompt_length 4096 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 10 \
    --run_name $WANDB_NAME \
    --save_steps 20 \
    --save_only_model true