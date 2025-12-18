TASK1="Pick up blocks from conveyor belt and place them in boxes"
TASK2="Pick up the box on the conveyor belt and place it into the blue plastic bin."
TASK3="Vacuum pick the box on the conveyor belt and place it into the blue plastic bin."


PI0_CKPT_V2=outputs/pi0_training_joint6/2025_11_30_18_25_53/checkpoints/060000/pretrained_model
PI05_CKPT_V1=outputs/pi05_training_joint6/2025_11_30_20_24_10/060000/pretrained_model
PI0_JOINT_ONLY=outputs/pi0_training_j6_only/2025_12_04_02_39_29/060000/pretrained_model

python -m lerobot.async_inference.robot_client_abb \
    --server_address=127.0.0.1:8080 \
    --robot.type=abb_irb120 \
    --robot.ip=192.168.125.1 \
    --robot.port=5000 \
    --robot.id=abb_irb120 \
    --robot.action_type=other \
    --robot.cameras="{ cam_0: {type: gopro, index_or_path: gopro, width: 448, height: 448, fps: 16}, cam_1: {type: kinectdk, index_or_path: 0, width: 448, height: 448, fps: 30}}" \
    --task="${TASK2}" \
    --policy_type=pi05 \
    --pretrained_name_or_path=$PI05_CKPT_V1 \
    --policy_device=cuda \
    --actions_per_chunk=10 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True