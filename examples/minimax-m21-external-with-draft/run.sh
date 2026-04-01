#!/bin/bash
# External online training WITH draft model — MiniMax-M2.1
#
# This script:
#   1. Starts Ray cluster
#   2. Starts aurora training (mooncake + callback server)
#   3. Waits for the training callback server to be ready
#   4. Waits for auto-created scratch draft model
#   5. Starts a standalone sglang server with EAGLE3 speculative decoding
#   6. Waits for the sglang server to be healthy
#   7. Waits for training to complete
#
# GPU allocation (default: 6 GPUs, non-overlapping):
#   - GPUs 0-1: training (FSDP/DP)
#   - GPUs 2-5: external sglang server (TP=4)
#
# Usage:
#   bash examples/minimax-m21-external-with-draft/run.sh [EXTRA_ARGS...]
#
# Environment variables:
#   SGLANG_GPUS           - GPUs for standalone sglang server (default: 2,3,4,5)
#   SGLANG_PORT           - sglang server port (default: 30000)
#   CALLBACK_PORT         - Training callback server port (default: 18080)
#   TARGET_MODEL          - Target model path (default: MiniMaxAI/MiniMax-M2.1)
#   DRAFT_MODEL           - Draft model path (default: auto-created scratch model)

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

# External sglang server settings
SGLANG_GPUS="${SGLANG_GPUS:-2,3,4,5}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
CALLBACK_PORT="${CALLBACK_PORT:-18080}"
MOONCAKE_GRPC_PORT="${MOONCAKE_GRPC_PORT:-50052}"
MOONCAKE_META_PORT="${MOONCAKE_META_PORT:-8090}"
TARGET_MODEL="${TARGET_MODEL:-MiniMaxAI/MiniMax-M2.1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/minimax-m21-external-with-draft}"
SCRATCH_DRAFT_DIR="$OUTPUT_DIR/scratch_draft_model"
DRAFT_MODEL="${DRAFT_MODEL:-$SCRATCH_DRAFT_DIR}"

# Speculative decoding settings for sglang server
SPEC_ALGORITHM="${SPEC_ALGORITHM:-EAGLE3}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-5}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS:-$((SPEC_NUM_STEPS + 1))}"
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
LOG_FILE="$LOG_DIR/minimax_m21_external_with_draft_${TIMESTAMP}.log"
SGLANG_LOG="$LOG_DIR/minimax_m21_sglang_with_draft_${TIMESTAMP}.log"
SGLANG_REQ_LOG_DIR="$LOG_DIR/requests_${TIMESTAMP}"
mkdir -p "$SGLANG_REQ_LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

pkill -9 mooncake_master || true

LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")
CALLBACK_URL="http://${LOCAL_IP}:${CALLBACK_PORT}/push_sample"

echo "=============================================="
echo "MiniMax-M2.1 External Training - With Draft"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Target model: $TARGET_MODEL"
echo "Draft model path: $DRAFT_MODEL"
echo "Training GPUs: $TOTAL_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  - Training: $TRAIN_GPUS GPUs"
echo "  - sglang server: $SGLANG_GPUS (TP=$SGLANG_TP_SIZE, port=$SGLANG_PORT)"
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
    output_dir="$OUTPUT_DIR" \
    model.draft_model_config="$SCRIPT_DIR/draft_config.json" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
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

# --- Step 3.5: Wait for scratch draft model to be created by training ---
if [ "$DRAFT_MODEL" = "$SCRATCH_DRAFT_DIR" ]; then
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
fi

# --- Step 4: Start sglang server with EAGLE3 ---
export MOONCAKE_MASTER_SERVER="${LOCAL_IP}:${MOONCAKE_GRPC_PORT}"
export MOONCAKE_METADATA_SERVER="http://${LOCAL_IP}:${MOONCAKE_META_PORT}/metadata"
export MOONCAKE_LOCAL_HOSTNAME="${LOCAL_IP}"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((16 * 1024 * 1024 * 1024))}"
export MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-$((2 * 1024 * 1024 * 1024))}"

echo "Starting sglang server on GPUs=$SGLANG_GPUS, port=$SGLANG_PORT..."
CUDA_VISIBLE_DEVICES=$SGLANG_GPUS python -m sglang.launch_server \
    --model-path "$TARGET_MODEL" \
    --port "$SGLANG_PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --mem-fraction-static "$MEM_FRACTION" \
    --speculative-algorithm "$SPEC_ALGORITHM" \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --speculative-num-steps "$SPEC_NUM_STEPS" \
    --speculative-num-draft-tokens "$SPEC_DRAFT_TOKENS" \
    --speculative-eagle-topk "$SPEC_EAGLE_TOPK" \
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
    --log-requests-level 0 \
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
echo "All services running. Waiting for training to complete..."
echo "  Training log: $LOG_FILE"
echo "  sglang log:   $SGLANG_LOG"
echo "  Send requests: bash examples/minimax-m21-external-with-draft/send_requests.sh"
echo "=============================================="
wait "$training_pid"
training_exit=$?

echo "=============================================="
echo "Training completed (exit code: $training_exit)"
echo "=============================================="
