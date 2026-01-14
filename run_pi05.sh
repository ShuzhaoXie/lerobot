REPOID=2511_2601_j7_d4_pi05
TIME=$(date +"%Y_%m_%d_%H_%M_%S")

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=$REPOID \
    --dataset.root=/home/szxie/storage/lerobot/data/baked/$REPOID \
    --policy.type=pi05 \
    --job_name=pi05_j7_dw4_$TIME \
    --policy.pretrained_path=lerobot/pi05_base \
    --output_dir=./outputs/pi05_training_joint7/$TIME \
    --policy.repo_id=$REPOID\_test \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --steps=60000 \
    --policy.device=cuda \
    --batch_size=8 \
    --wandb.enable=true