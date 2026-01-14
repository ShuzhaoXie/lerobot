TASK1="Pick up blocks from conveyor belt and place them in boxes"
TASK2="Pick up the box on the conveyor belt and place it into the blue plastic bin."
TASK3="Vacuum pick the box on the conveyor belt and place it into the blue plastic bin."
TASK4="Pick up the box on the conveyor belt and place it into the blue plastic tray."

PI0_CKPT_V2=outputs/pi0_training_joint6/2025_11_30_18_25_53/checkpoints/060000/pretrained_model
PI05_CKPT_V1=outputs/pi05_training_joint6/2025_11_30_20_24_10/060000/pretrained_model
PI0_JOINT_ONLY=outputs/pi0_training_j6_only/2025_12_04_02_39_29/060000/pretrained_model

PI0_CKPT_V3=outputs/pi0_training_j7/2025_12_17_21_55_35/checkpoints/060000/pretrained_model # d2
PI0_CKPT_V4=outputs/pi0_training_j7/2025_12_17_22_26_02/checkpoints/060000/pretrained_model # d8
PI0_CKPT_V5=outputs/pi0_training_j7/2025_12_18_00_11_32/checkpoints/060000/pretrained_model # d4
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=abb_irb120 \
    --robot.ip=192.168.125.1 \
    --robot.port=5000 \
    --robot.id=abb_irb120 \
    --robot.action_type=other \
    --robot.cameras="{ cam_0: {type: gopro, index_or_path: gopro, width: 448, height: 448, fps: 30}, cam_1: {type: kinectdk, index_or_path: 0, width: 448, height: 448, fps: 30}}" \
    --task="${TASK4}" \
    --policy_type=pi0 \
    --pretrained_name_or_path=$PI0_CKPT_V4 \
    --policy_device=cuda \
    --actions_per_chunk=10 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True