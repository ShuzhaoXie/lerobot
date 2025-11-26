GORPOSTREAM="udp://10.5.5.100:8554?buffer_size=1000000&fifo_size=1000000&overrun_nonfatal=1"
TASK=""

python -m lerobot.async_inference.robot_client_abb \
    --server_address=127.0.0.1:8080 \
    --robot.type=abb_irb120 \
    --robot.ip=192.168.125.1 \
    --robot.port=5000 \
    --robot.id=abb_irb120_n1 \
    --robot.cameras="{ cam0: {type: opencv, index_or_path: ${GORPOSTREAM}, width: 224, height: 224, fps: 30}, cam1: {type: azurekinectdk, index_or_path: 0, width: 224, height: 224, fps: 30}}" \
    --task="dummy" \
    --policy_type=pi0 \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True