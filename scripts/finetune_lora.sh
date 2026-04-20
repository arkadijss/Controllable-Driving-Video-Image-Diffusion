# Copyright (c) 2026 Arkādijs Sergejevs
# Adapted from Hugging Face Diffusers (LoRA Training Documentation)
# Original Copyright The HuggingFace Inc. team
# Licensed under the Apache License 2.0.

export WANDB_API_KEY=x

export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="./sddata/finetune/lora/kitti-360"
export DATASET_DIR="/home/user/data/KITTI-360_proc/lora/center_cropped"

accelerate launch --mixed_precision="fp16" src/train_text_to_image_lora.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_DIR \
  --dataloader_num_workers=8 \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="KITTI-360 style, a realistic driving scene, day, clear weather" \
  --seed=1337
