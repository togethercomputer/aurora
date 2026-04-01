#!/bin/bash
# Node 2: Inference node — standalone sglang server with EAGLE3 (2-node)
#
# Target model: Qwen/Qwen3-Coder-Next (80B MoE, 3B active params)
#
# This script runs on the INFERENCE node and:
#   1. Connects to Node 1's mooncake master (for hidden state transfer)
#   2. Starts sglang server with EAGLE3 speculative decoding
#   3. Sends training callbacks to Node 1's callback server on each completion
#
# Requirements:
#   - Set NODE1_IP to the IP of the training node
#   - Node 1 must already be running (run_node1_train.sh)
#   - Shared filesystem (NFS) mounted at same path for draft model access
#
# GPU allocation on this node:
#   - 4 GPUs for sglang (TP=4, target + draft model)
#
# Usage:
#   NODE1_IP=10.0.0.1 bash examples/qwen3-8b-coder-next-external-with-draft-2node/run_node2_sglang.sh
#
# Environment variables:
#   NODE1_IP        - (required) IP address of the training node
#   DRAFT_MODEL     - Path to draft model (default: /scratch/shared/aurora/scratch_draft_model)
#   SGLANG_GPUS     - GPUs for sglang server (default: 0,1,2,3)
#   SGLANG_PORT     - sglang server port (default: 30000)
#   CALLBACK_PORT   - Training callback port on Node 1 (default: 18080)

set -euo pipefail
set -x

if [ -z "${NODE1_IP:-}" ]; then
    echo "ERROR: NODE1_IP must be set to the IP of the training node (Node 1)"
    echo "Usage: NODE1_IP=10.0.0.1 bash $0"
    exit 1
fi

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$ROOT_DIR/_sglang/python:${PYTHONPATH:-}"

# Inference server settings
SGLANG_GPUS="${SGLANG_GPUS:-0,1,2,3}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
CALLBACK_PORT="${CALLBACK_PORT:-18080}"
MOONCAKE_GRPC_PORT="${MOONCAKE_GRPC_PORT:-50052}"
MOONCAKE_META_PORT="${MOONCAKE_META_PORT:-8090}"
TARGET_MODEL="${TARGET_MODEL:-/scratch/bobbie/hf_cache/Qwen3-Coder-Next}"

# Shared filesystem (NFS) — must match OUTPUT_DIR from Node 1
OUTPUT_DIR="${OUTPUT_DIR:-/data/bobbie/tmp/aurora/qwen3-next-coder-external-2node}"
DRAFT_MODEL="${DRAFT_MODEL:-$OUTPUT_DIR/scratch_draft_model}"

# Speculative decoding settings
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

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SGLANG_LOG="$LOG_DIR/2node_sglang_${TIMESTAMP}.log"

LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")
CALLBACK_URL="http://${NODE1_IP}:${CALLBACK_PORT}/push_sample"

echo "=============================================="
echo "Node 2: Inference — sglang with EAGLE3 (2-node)"
echo "=============================================="
echo "Local IP (Node 2): $LOCAL_IP"
echo "Remote IP (Node 1): $NODE1_IP"
echo "Target model: $TARGET_MODEL"
echo "Draft model: $DRAFT_MODEL"
echo "sglang GPUs: $SGLANG_GPUS (TP=$SGLANG_TP_SIZE, port=$SGLANG_PORT)"
echo "Callback URL: $CALLBACK_URL"
echo "Mooncake master: $NODE1_IP:$MOONCAKE_GRPC_PORT"
echo "=============================================="

# --- Pre-flight checks ---
echo "Checking connectivity to Node 1..."
if ! curl -s --max-time 5 "http://${NODE1_IP}:${CALLBACK_PORT}/health" > /dev/null 2>&1; then
    echo "WARNING: Cannot reach Node 1 callback server at http://${NODE1_IP}:${CALLBACK_PORT}/health"
    echo "Make sure run_node1_train.sh is running on Node 1."
    echo "Proceeding anyway (sglang will retry callbacks)..."
fi

echo "Checking draft model at $DRAFT_MODEL ..."
if [ ! -f "$DRAFT_MODEL/config.json" ]; then
    echo "ERROR: Draft model not found at $DRAFT_MODEL/config.json"
    echo "Make sure Node 1 has created the scratch draft model and the shared filesystem is mounted."
    exit 1
fi
echo "Draft model found."

# --- Cleanup on exit ---
cleanup() {
    echo "Stopping sglang server..."
    if [ -n "${sglang_pid:-}" ] && kill -0 "$sglang_pid" 2>/dev/null; then
        kill "$sglang_pid" 2>/dev/null || true
        wait "$sglang_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# --- Connect to Node 1's mooncake master ---
export MOONCAKE_MASTER_SERVER="${NODE1_IP}:${MOONCAKE_GRPC_PORT}"
export MOONCAKE_METADATA_SERVER="http://${NODE1_IP}:${MOONCAKE_META_PORT}/metadata"
export MOONCAKE_LOCAL_HOSTNAME="${LOCAL_IP}"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((16 * 1024 * 1024 * 1024))}"
export MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-$((2 * 1024 * 1024 * 1024))}"

# --- Start sglang server ---
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

# --- Wait for sglang to be healthy ---
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

echo "=============================================="
echo "Node 2 is running. sglang server ready at http://${LOCAL_IP}:${SGLANG_PORT}"
echo ""
echo "Send requests with:"
echo "  SGLANG_URL=http://${LOCAL_IP}:${SGLANG_PORT} \\"
echo "    bash examples/qwen3-8b-coder-next-external-with-draft-2node/send_requests.sh"
echo ""
echo "sglang log: $SGLANG_LOG"
echo "=============================================="

# Keep running until killed or sglang exits
wait "$sglang_pid"
sglang_exit=$?
echo "sglang server exited (code: $sglang_exit)"
