SAMPLE_RATE=8

python create_dataset_v30_down.py \
    --input /home/szxie/storage/lerobot/data/raw_videos/abb0/2025-11-12robotdata \
    --output /home/szxie/storage/lerobot/data/baked/2511-2601_down$SAMPLE_RATE \
    --task "Pick up the box on the conveyor belt and place it into the blue plastic tray." \
    --sample-rate $SAMPLE_RATE \
    --action-type 0