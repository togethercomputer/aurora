#!/bin/bash
# Node 1: Training node — external online training WITH draft model (2-node)
#
# Target model: Qwen/Qwen3-Coder-Next (80B MoE, 3B active params)
#
# This script runs on the TRAINING node and:
#   1. Starts Ray cluster (head node)
#   2. Starts mooncake master
#   3. Starts aurora training actors
#   4. Starts the callback HTTP server (receives samples from Node 2's sglang)
#   5. Periodically syncs draft model weights to Node 2 via shared filesystem
#
# Requirements:
#   - Set NODE2_IP to the IP of the inference node
#   - Shared filesystem (NFS) mounted at SHARED_DIR on both nodes
#
# GPU allocation on this node:
#   - 2 GPUs for training (FSDP/DP: draft model sharded)
#
# Usage:
#   NODE2_IP=10.0.0.2 bash examples/qwen3-8b-coder-next-external-with-draft-2node/run_node1_train.sh
#
# Environment variables:
#   NODE2_IP        - (required) IP address of the inference node
#   SHARED_DIR      - Shared filesystem path for weight sync (default: /scratch/shared/aurora)
#   SGLANG_PORT     - sglang server port on Node 2 (default: 30000)
#   CALLBACK_PORT   - Training callback server port (default: 18080)

set -euo pipefail
set -x

if [ -z "${NODE2_IP:-}" ]; then
    echo "ERROR: NODE2_IP must be set to the IP of the inference node (Node 2)"
    echo "Usage: NODE2_IP=10.0.0.2 bash $0"
    exit 1
fi

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
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

# Network settings
SGLANG_PORT="${SGLANG_PORT:-30000}"
CALLBACK_PORT="${CALLBACK_PORT:-18080}"
MOONCAKE_GRPC_PORT="${MOONCAKE_GRPC_PORT:-50052}"
MOONCAKE_META_PORT="${MOONCAKE_META_PORT:-8090}"
# Output dir on shared filesystem (NFS) — must be accessible from both nodes
# so Node 2 can read the scratch draft model and weight sync checkpoints.
OUTPUT_DIR="${OUTPUT_DIR:-/data/bobbie/tmp/aurora/qwen3-next-coder-external-2node}"
SCRATCH_DRAFT_DIR="$OUTPUT_DIR/scratch_draft_model"
WEIGHT_SYNC_DIR="$HOME/weight_sync"
mkdir -p "$OUTPUT_DIR"

export AURORA_LOG_LEVEL=INFO

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/2node_train_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

pkill -9 mooncake_master || true

LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")

echo "=============================================="
echo "Node 1: Training — External WITH Draft (2-node)"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Local IP (Node 1): $LOCAL_IP"
echo "Remote IP (Node 2): $NODE2_IP"
echo "Training GPUs: $TOTAL_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  - Training: $TRAIN_GPUS GPUs"
echo "  - sglang on Node 2 port: $SGLANG_PORT"
echo "Callback port: $CALLBACK_PORT"
echo "Mooncake master: $LOCAL_IP:$MOONCAKE_GRPC_PORT"
echo "Output dir (NFS): $OUTPUT_DIR"
echo "Scratch draft dir: $SCRATCH_DRAFT_DIR"
echo "Extra args: $*"
echo "=============================================="

# --- Cleanup on exit ---
cleanup() {
    echo "Stopping services..."
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
# Training will start mooncake master + callback server automatically.
# The sglang server on Node 2 connects to this node's mooncake master
# and sends callbacks to this node's callback server.
echo "Starting training..."
python3 -m aurora.train_entry \
    --config "$CONFIG_FILE" \
    dataset.train_data_path="$ROOT_DIR/datasets/onlinesd/merged/merged_train_data.jsonl" \
    output_dir="$OUTPUT_DIR" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
    inference.inference_num_gpus_per_node="$TOTAL_GPUS" \
    online_serving.enabled=true \
    online_serving.port="$CALLBACK_PORT" \
    online_serving.sglang_url="http://${NODE2_IP}:${SGLANG_PORT}" \
    decode.weight_sync_checkpoint_path="$WEIGHT_SYNC_DIR" \
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
echo "Training callback server is ready (took ${WAITED}s)"

# --- Step 3.5: Wait for scratch draft model ---
echo "Waiting for auto-created scratch draft model at $SCRATCH_DRAFT_DIR ..."
WAITED=0
while [ ! -f "$SCRATCH_DRAFT_DIR/config.json" ]; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ "$WAITED" -ge 120 ]; then
        echo "ERROR: Scratch draft model not created within 120s"
        exit 1
    fi
    if ! kill -0 "$training_pid" 2>/dev/null; then
        echo "ERROR: Training process died before creating scratch draft model"
        exit 1
    fi
done
echo "Scratch draft model ready at $SCRATCH_DRAFT_DIR (took ${WAITED}s)"

# --- Step 4: Print instructions for Node 2 ---
echo ""
echo "=============================================="
echo "Node 1 is ready. Start Node 2 with:"
echo ""
echo "  NODE1_IP=$LOCAL_IP \\"
echo "    bash examples/qwen3-8b-coder-next-external-with-draft-2node/run_node2_sglang.sh"
echo ""
echo "Then send requests with:"
echo "  SGLANG_URL=http://${NODE2_IP}:${SGLANG_PORT} \\"
echo "    bash examples/qwen3-8b-coder-next-external-with-draft-2node/send_requests.sh"
echo "=============================================="
echo ""

# --- Step 5: Wait for training to finish ---
echo "Waiting for training to complete..."
echo "  Training log: $LOG_FILE"
wait "$training_pid"
training_exit=$?

echo "=============================================="
echo "Training completed (exit code: $training_exit)"
echo "=============================================="
