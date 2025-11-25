REPOID=2025-11-12
TIME=$(date +"%Y_%m_%d_%H_%M_%S")

CUDA_VISIBLE_DEVICES=2 python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=$REPOID \
    --dataset.root=/home/szxie/storage/lerobot/data/baked/$REPOID \
    --policy.type=pi0 \
    --job_name=pi0_training_v2_$TIME \
    --policy.pretrained_path=lerobot/pi0_base \
    --output_dir=./outputs/pi0_training_v2/$TIME \
    --policy.repo_id=$REPOID\_test \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --steps=60000 \
    --policy.device=cuda \
    --batch_size=8 \
    --wandb.enable=true