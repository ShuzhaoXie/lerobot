import pandas as pd

df = pd.read_parquet("/home/szxie/storage/lerobot/data/baked/2025-11-12_j6_v3/meta/tasks.parquet")
# 打印原始 index，看一下它是什么
print("OLD INDEX:", df.index)

# # 新的任务描述
# new_task = "Pick up the box on the conveyor belt and place it into the blue plastic bin."

# # 替换 index：只改文字，不动数据
# df.index = [new_task]
# print(df)

# # 如果你想保存回 parquet
# df.to_parquet("/home/szxie/storage/lerobot/data/baked/2025-11-12_joint6_t2/meta/tasks.parquet")
