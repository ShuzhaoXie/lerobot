import os 
import json 

def check_gripper_pattern(data, min_len=10):
    """
    data: list[dict]，每个元素有 key 'gripper'
    min_len: 连续多少帧才算“长的一段”
    """

    filtered = []
    for item in data:
        if item is None:
            continue
        if not isinstance(item, dict):
            continue
        # 任意 value 为 None 就丢弃
        if any(v is None for v in item.values()):
            continue
        if "gripper" not in item:
            continue
        filtered.append(item)
        
    # 1. 提取状态序列
    states = [item["gripper"] for item in filtered]

    if not states:
        return False

    # 2. run-length encoding
    segments = []
    cur_state = states[0]
    cur_len = 1

    for s in states[1:]:
        if s == cur_state:
            cur_len += 1
        else:
            segments.append((cur_state, cur_len))
            cur_state = s
            cur_len = 1
    segments.append((cur_state, cur_len))
    
    # print(segments)
    # 3. 找 off -> on -> off 的连续三段
    if len(segments) != 3:
        print(segments)
        return False

    (s1, l1), (s2, l2), (s3, l3) = segments
    return (
        s1 == "off" and s2 == "on" and s3 == "off" and
        l1 >= min_len and l2 >= min_len and l3 >= min_len
    )


raw_dir = "/home/szxie/storage/lerobot/data/raw_videos/abb0/2025-11-12robotdata"
names = os.listdir("/home/szxie/storage/lerobot/data/raw_videos/abb0/2025-11-12robotdata")
for name in names:
    cur_dir = os.path.join(raw_dir, name)
    act_path = os.path.join(cur_dir, "action.json")
    if os.path.exists(act_path):
        with open(os.path.join(cur_dir, "action.json"), "r") as f:
            all_data = json.load(f)
            if check_gripper_pattern(all_data["states"]):
                # print('yes')
                pass
            else:
                print('no', name)
    