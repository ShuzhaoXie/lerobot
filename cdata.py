import pandas as pd

paths = [
    "/home/szxie/storage/lerobot/data/baked/2025-11-12_joint6/meta/tasks.parquet",
    "/home/szxie/storage/lerobot/data/baked/2025-11-12/meta/tasks.parquet",
]

for p in paths:
    print("=" * 80)
    print(f"Reading: {p}")
    try:
        df = pd.read_parquet(p)
        print(df)
    except Exception as e:
        print(f"Error reading {p}: {e}")