# ABB

# Data

```
ln -s /mnt/storage/users/szxie_data/abb/data data
```

# Data Preprocess
check `create_dataset_v30_backup.sh`

## Note for Pi 0.5
```
cp /home/szxie/storage/lerobot/data/baked/2025-11-12_j6_v4 /home/szxie/storage/lerobot/data/baked/2025-11-12_j6_v4_pi05
python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py --repo-id=lerobot/1121_j6_v4_pi05 --root=/home/szxie/storage/lerobot/data/baked/2025-11-12_j6_v4_pi05 --overwrite
```

# Environment
## Step 1
Follow [README.md](./README.md)
## Step 2
```bash
pip install -e ".[pi]"
pip install -e ".[async]"
pip install fastapi uvicorn peft
pip install pykinect_azure
pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
```