SAMPLE_RATE=4

python create_dataset_v30_down.py \
    --input /mnt/storage/users/szxie_data/abb/data/raw_videos/abb0/2025-11-12robotdata \
    --output /home/szxie/storage/lerobot/data/baked/2025-11-12_j7_down$SAMPLE_RATE \
    --task "Pick up the box on the conveyor belt and place it into the blue plastic bin." \
    --sample-rate $SAMPLE_RATE \
    --action-type 0