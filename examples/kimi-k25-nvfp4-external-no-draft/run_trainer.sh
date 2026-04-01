#!/bin/bash
# Kimi-K2.5 NVFP4 external training — Node 1: Trainer
#
# Run this on the TRAINING node (8 GPUs for FSDP/DP draft model training).
# It starts Ray, mooncake master, and the training callback server.
# The sglang server on Node 2 connects to this node's callback URL.
#
# Before running:
#   1. Start this script on Node 1 (trainer)
#   2. Wait for "Training callback server is ready"
#   3. Start run_sglang.sh on Node 2 with TRAINER_IP set to this node's IP
#   4. Start send_requests.sh to send traffic
#
# GPU allocation:
#   - All 8 GPUs on this node: training (FSDP/DP)
#
# Usage:
#   bash examples/kimi-k25-nvfp4-external-no-draft/run_trainer.sh [EXTRA_ARGS...]
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES  - GPUs for training (default: 0,1,2,3,4,5,6,7)
#   TRAIN_GPUS            - Number of training GPUs (default: 8)
#   CALLBACK_PORT         - Training callback server port (default: 18080)
#   SGLANG_PORT           - sglang server port on Node 2 (default: 30000)

set -euo pipefail
set -x

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export HF_HOME="${HF_HOME:-/scratch/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT_DIR/cache/hf_datasets}"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export PYTHONPATH="$ROOT_DIR/_sglang/python:${PYTHONPATH:-}"

CONFIG_FILE="$SCRIPT_DIR/config.yaml"

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPU_ARRAY[@]}

TRAIN_GPUS="${TRAIN_GPUS:-8}"
CALLBACK_PORT="${CALLBACK_PORT:-18080}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
MOONCAKE_GRPC_PORT="${MOONCAKE_GRPC_PORT:-50052}"
MOONCAKE_META_PORT="${MOONCAKE_META_PORT:-8090}"

export AURORA_LOG_LEVEL=INFO

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/kimi_k25_nvfp4_trainer_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

pkill -9 mooncake_master || true

LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")

echo "=============================================="
echo "Kimi-K2.5 NVFP4 External Training — Trainer Node"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Training GPUs: $TRAIN_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Local IP: $LOCAL_IP"
echo "Callback port: $CALLBACK_PORT"
echo "Mooncake gRPC: $LOCAL_IP:$MOONCAKE_GRPC_PORT"
echo "Mooncake meta: $LOCAL_IP:$MOONCAKE_META_PORT"
echo ""
echo "After this script prints 'Training callback server is ready', start"
echo "the sglang server on Node 2 with:"
echo "  TRAINER_IP=$LOCAL_IP bash examples/kimi-k25-nvfp4-external-no-draft/run_sglang.sh"
echo "=============================================="

# --- Cleanup on exit ---
cleanup() {
    echo "Stopping training services..."
    if [ -n "${training_pid:-}" ] && kill -0 "$training_pid" 2>/dev/null; then
        kill "$training_pid" 2>/dev/null || true
        wait "$training_pid" 2>/dev/null || true
    fi
    pkill -9 mooncake_master 2>/dev/null || true
    RAY_ADDRESS="${LOCAL_IP:-127.0.0.1}:${RAY_PORT:-6380}" ray stop 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- Step 1: Start Ray ---
RAY_PORT="${RAY_PORT:-6380}"
export RAY_ADDRESS="${LOCAL_IP}:${RAY_PORT}"
ray stop --force 2>/dev/null || true
echo "Starting Ray on port $RAY_PORT with $TOTAL_GPUS GPUs..."
ray start --head --num-gpus "$TOTAL_GPUS" --port "$RAY_PORT" --disable-usage-stats

# --- Step 2: Start training ---
echo "Starting training (mooncake master + callback server will come up)..."
python3 -m aurora.train_entry \
    --config "$CONFIG_FILE" \
    dataset.train_data_path="$ROOT_DIR/datasets/onlinesd/merged/merged_train_data.jsonl" \
    output_dir="$ROOT_DIR/outputs/kimi-k25-nvfp4-external-no-draft" \
    cache_dir="$ROOT_DIR/cache" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
    inference.inference_num_gpus=0 \
    inference.inference_num_gpus_per_node="$TOTAL_GPUS" \
    online_serving.enabled=true \
    online_serving.port="$CALLBACK_PORT" \
    online_serving.sglang_url="http://${SGLANG_IP:-10.179.0.242}:${SGLANG_PORT}" \
    mooncake.master_server_address="$LOCAL_IP:$MOONCAKE_GRPC_PORT" \
    mooncake.metadata_port="$MOONCAKE_META_PORT" \
    ${MOONCAKE_DEVICE_NAME:+mooncake.device_name="$MOONCAKE_DEVICE_NAME"} \
    "$@" &
training_pid=$!
echo "Training PID: $training_pid"

# --- Step 3: Wait for callback server to be ready ---
echo "Waiting for training callback server at http://localhost:${CALLBACK_PORT}/health ..."
MAX_WAIT=600
WAITED=0
while ! curl -s "http://localhost:${CALLBACK_PORT}/health" > /dev/null 2>&1; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Training callback server failed to start within ${MAX_WAIT}s"
        exit 1
    fi
    if ! kill -0 "$training_pid" 2>/dev/null; then
        echo "ERROR: Training process died before callback server was ready"
        exit 1
    fi
done
echo "=============================================="
echo "Training callback server is ready (took ${WAITED}s)"
echo ""
echo "Now start the sglang server on Node 2:"
echo "  TRAINER_IP=$LOCAL_IP bash examples/kimi-k25-nvfp4-external-no-draft/run_sglang.sh"
echo "=============================================="

# --- Step 4: Wait for training to finish ---
wait "$training_pid"
training_exit=$?

echo "=============================================="
echo "Training completed (exit code: $training_exit)"
echo "=============================================="
