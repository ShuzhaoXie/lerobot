# ABB

# Data

```
ln -s /mnt/storage/users/szxie_data/abb/data data
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