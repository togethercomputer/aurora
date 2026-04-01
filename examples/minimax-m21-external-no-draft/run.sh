#!/bin/bash
# External online training WITHOUT draft model — MiniMax-M2.1
#
# This script:
#   1. Starts Ray cluster
#   2. Starts aurora training (mooncake + callback server)
#   3. Waits for the training callback server to be ready
#   4. Starts a standalone sglang server WITHOUT speculative decoding
#   5. Waits for the sglang server to be healthy
#   6. Waits for training to complete
#
# GPU allocation (default: 6 GPUs, non-overlapping):
#   - GPUs 0-1: training (FSDP/DP)
#   - GPUs 2-5: external sglang server (TP=4)
#
# Usage:
#   bash examples/minimax-m21-external-no-draft/run.sh [EXTRA_ARGS...]
#
# Environment variables:
#   SGLANG_GPUS           - GPUs for standalone sglang server (default: 2,3,4,5)
#   SGLANG_PORT           - sglang server port (default: 30000)
#   CALLBACK_PORT         - Training callback server port (default: 18080)
#   TARGET_MODEL          - Target model path (default: MiniMaxAI/MiniMax-M2.1)

set -euo pipefail
set -x

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

TRAIN_GPUS=2

# External sglang server settings (separate GPUs)
SGLANG_GPUS="${SGLANG_GPUS:-2,3,4,5}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
CALLBACK_PORT="${CALLBACK_PORT:-18080}"
MOONCAKE_GRPC_PORT="${MOONCAKE_GRPC_PORT:-50052}"
MOONCAKE_META_PORT="${MOONCAKE_META_PORT:-8090}"
TARGET_MODEL="${TARGET_MODEL:-MiniMaxAI/MiniMax-M2.1}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-12}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-12}"

# Auto-detect sglang TP size from SGLANG_GPUS
IFS=',' read -ra SGLANG_GPU_ARRAY <<< "$SGLANG_GPUS"
SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-${#SGLANG_GPU_ARRAY[@]}}"

export AURORA_LOG_LEVEL=INFO

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/minimax_m21_external_no_draft_${TIMESTAMP}.log"
SGLANG_LOG="$LOG_DIR/minimax_m21_sglang_no_draft_${TIMESTAMP}.log"
SGLANG_REQ_LOG_DIR="$LOG_DIR/requests_${TIMESTAMP}"
mkdir -p "$SGLANG_REQ_LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

pkill -9 mooncake_master || true

LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")
CALLBACK_URL="http://${LOCAL_IP}:${CALLBACK_PORT}/push_sample"

echo "=============================================="
echo "MiniMax-M2.1 External Training - No Draft"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Target model: $TARGET_MODEL"
echo "Draft Model: NONE (no speculative decoding)"
echo "Training GPUs: $TOTAL_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  - Training: $TRAIN_GPUS GPUs"
echo "  - sglang server: $SGLANG_GPUS (TP=$SGLANG_TP_SIZE, target model only)"
echo "Callback URL: $CALLBACK_URL"
echo "Local IP: $LOCAL_IP"
echo "Extra args: $*"
echo "=============================================="

# --- Cleanup on exit ---
cleanup() {
    echo "Stopping services..."
    if [ -n "${sglang_pid:-}" ] && kill -0 "$sglang_pid" 2>/dev/null; then
        kill "$sglang_pid" 2>/dev/null || true
        wait "$sglang_pid" 2>/dev/null || true
    fi
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
REQUIRED_GPUS=$TRAIN_GPUS
export RAY_ADDRESS="${LOCAL_IP}:${RAY_PORT}"
ray stop --force 2>/dev/null || true
echo "Starting Ray on port $RAY_PORT with $TOTAL_GPUS GPUs..."
ray start --head --num-gpus "$TOTAL_GPUS" --port "$RAY_PORT" --disable-usage-stats

# --- Step 2: Start training in background ---
echo "Starting training (mooncake master + callback server will come up)..."
python3 -m aurora.train_entry \
    --config "$CONFIG_FILE" \
    dataset.train_data_path="$ROOT_DIR/datasets/onlinesd/merged/merged_train_data_shuffled.jsonl" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
    inference.inference_num_gpus=0 \
    inference.inference_num_gpus_per_node="$TOTAL_GPUS" \
    online_serving.enabled=true \
    online_serving.port="$CALLBACK_PORT" \
    online_serving.sglang_url="http://localhost:${SGLANG_PORT}" \
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

# --- Step 4: Start sglang server WITHOUT speculative decoding ---
export MOONCAKE_MASTER_SERVER="${LOCAL_IP}:${MOONCAKE_GRPC_PORT}"
export MOONCAKE_METADATA_SERVER="http://${LOCAL_IP}:${MOONCAKE_META_PORT}/metadata"
export MOONCAKE_LOCAL_HOSTNAME="${LOCAL_IP}"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((16 * 1024 * 1024 * 1024))}"
export MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-$((2 * 1024 * 1024 * 1024))}"

echo "Starting sglang server on GPUs=$SGLANG_GPUS, port=$SGLANG_PORT (NO draft model)..."
CUDA_VISIBLE_DEVICES=$SGLANG_GPUS python -m sglang.launch_server \
    --model-path "$TARGET_MODEL" \
    --port "$SGLANG_PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --mem-fraction-static "$MEM_FRACTION" \
    --tp-size "$SGLANG_TP_SIZE" \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --disable-radix-cache \
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --enable-spec-training-mooncake \
    --enable-aux-hidden-states \
    --enable-return-hidden-states \
    --spec-training-callback-url "$CALLBACK_URL" \
    --log-requests \
    --log-requests-target "$SGLANG_REQ_LOG_DIR" \
    > "$SGLANG_LOG" 2>&1 &
sglang_pid=$!
echo "sglang server PID: $sglang_pid (log: $SGLANG_LOG)"

# --- Step 5: Wait for sglang to be healthy ---
echo "Waiting for sglang server to be ready..."
MAX_WAIT=600
WAITED=0
while ! curl -s "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: sglang server failed to start within ${MAX_WAIT}s"
        tail -20 "$SGLANG_LOG"
        exit 1
    fi
    if ! kill -0 "$sglang_pid" 2>/dev/null; then
        echo "ERROR: sglang server process died"
        tail -20 "$SGLANG_LOG"
        exit 1
    fi
done
echo "sglang server is ready (took ${WAITED}s)"

# --- Step 6: Wait for training to finish ---
echo "=============================================="
echo "All services running. Waiting for training..."
echo "  Training log: $LOG_FILE"
echo "  sglang log:   $SGLANG_LOG"
echo "  Send requests: bash examples/minimax-m21-external-no-draft/send_requests.sh"
echo "=============================================="

wait "$training_pid"
echo "=============================================="
echo "Training completed!"
echo "=============================================="
