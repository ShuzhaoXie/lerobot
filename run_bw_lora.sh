REPOID=2025-11-12
TIME=$(date +"%Y_%m_%d_%H_%M_%S")

CUDA_VISIBLE_DEVICES=2,4 python src/lerobot/scripts/abb_bw_lora.py \
    --dataset.repo_id=$REPOID \
    --dataset.root=/home/szxie/storage/lerobot/data/baked/$REPOID \
    --policy.type=pi0 \
    --job_name=pi0_training_lora \
    --policy.pretrained_path=lerobot/pi0_base \
    --output_dir=./outputs/pi0_training_v2/$TIME \
    --policy.repo_id=$REPOID\_test \
    --steps=10000 \
    --policy.device=cuda \
    --batch_size=4 \
    --wandb.enable=true


    #     --policy.compile_model=true \
    # --policy.gradient_checkpointing=true \
    # --policy.dtype=bfloat16 \