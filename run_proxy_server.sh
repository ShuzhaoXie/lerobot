HF_ENDPOINT=https://hf-mirror.com python -m lerobot.async_inference.policy_server_abb \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=5 \
     --inference_latency=0.1