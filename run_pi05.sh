REPOID=2025-11-12_j6_v4_pi05
TIME=$(date +"%Y_%m_%d_%H_%M_%S")

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=$REPOID \
    --dataset.root=/home/szxie/storage/lerobot/data/baked/$REPOID \
    --policy.type=pi05 \
    --job_name=pi05_training_j6_v4_pi05_$TIME \
    --policy.pretrained_path=lerobot/pi05_base \
    --output_dir=./outputs/pi05_training_joint6/$TIME \
    --policy.repo_id=$REPOID\_test \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --steps=60000 \
    --policy.device=cuda \
    --batch_size=8 \
    --wandb.enable=true