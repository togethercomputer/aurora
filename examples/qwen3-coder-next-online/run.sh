#!/bin/bash
# Online training with SglEngine — Qwen3-Coder-Next (EAGLE3 speculative decoding)
#
# Target model: Qwen/Qwen3-Coder-Next (80B MoE, 3B active params)
# SglEngine manages inference internally with TP=4.
# No external server needed — Ray manages all inference engines.
#
# GPU allocation (default: 8 GPUs):
#   - 4 GPUs for inference (SglEngine TP=4, target + draft model)
#   - 2 GPUs for training (FSDP/DP: draft model sharded)
#
# Usage:
#   bash examples/qwen3-next-coder-online/run.sh [EXTRA_ARGS...]

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HOME="${HF_HOME:-/scratch/shared/huggingface}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT_DIR/cache/hf_datasets}"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export PYTHONPATH="$ROOT_DIR/_sglang/python:${PYTHONPATH:-}"

CONFIG_FILE="$SCRIPT_DIR/config.yaml"

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPU_ARRAY[@]}

TRAIN_GPUS="${TRAIN_GPUS:-2}"
INFERENCE_GPUS="${INFERENCE_GPUS:-4}"

export AURORA_LOG_LEVEL=INFO

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/qwen3_next_coder_online_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")

echo "=============================================="
echo "Qwen3-Coder-Next Online Training (EAGLE3 Speculative)"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Total GPUs: $TOTAL_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  - Training GPUs: $TRAIN_GPUS (FSDP/DP)"
echo "  - Inference GPUs: $INFERENCE_GPUS (SglEngine with TP)"
echo "Local IP: $LOCAL_IP"
echo "Extra args: $*"
echo "=============================================="

RAY_PORT="${RAY_PORT:-6380}"
export RAY_ADDRESS="${LOCAL_IP}:${RAY_PORT}"
ray stop --force 2>/dev/null || true
echo "Starting Ray on port $RAY_PORT with $TOTAL_GPUS GPUs..."
ray start --head --num-gpus "$TOTAL_GPUS" --port "$RAY_PORT" --disable-usage-stats

python3 -m aurora.train_entry \
    --config "$CONFIG_FILE" \
    dataset.train_data_path="$ROOT_DIR/datasets/onlinesd/merged/merged_code_train_shuffled.jsonl" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
    inference.inference_num_gpus="$INFERENCE_GPUS" \
    inference.inference_num_gpus_per_engine=4 \
    inference.inference_num_gpus_per_node="$TOTAL_GPUS" \
    inference.sglang.tp_size=4 \
    decode.cuda_graph_max_bs=12 \
    decode.max_running_requests=12 \
    mooncake.master_server_address="$LOCAL_IP:50052" \
    ${MOONCAKE_DEVICE_NAME:+mooncake.device_name="$MOONCAKE_DEVICE_NAME"} \
    "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="
