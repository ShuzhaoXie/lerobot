REPOID=2025-11-12

CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=$REPOID \
    --dataset.root=/home/szxie/storage/lerobot/data/baked/$REPOID \
    --policy.type=pi0 \
    --output_dir=./outputs/pi0_training \
    --job_name=pi0_training \
    --policy.pretrained_path=lerobot/pi0_base \
    --policy.repo_id=$REPOID\_test \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=2