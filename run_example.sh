REPOID=2025-11-12_j6_only
TIME=$(date +"%Y_%m_%d_%H_%M_%S")

CUDA_VISIBLE_DEVICES=5 python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=$REPOID \
    --dataset.root=/home/szxie/storage/lerobot/data/baked/$REPOID \
    --policy.type=pi0 \
    --job_name=pi0_training_j6_only_$TIME \
    --policy.pretrained_path=lerobot/pi0_base \
    --output_dir=./outputs/pi0_training_joint6/$TIME \
    --policy.repo_id=$REPOID\_test \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --steps=60000 \
    --policy.device=cuda \
    --batch_size=8 \
    --wandb.enable=true