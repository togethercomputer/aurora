# Examples

Runnable training recipes for Aurora across models and deployment modes.

Each example directory contains:

- `config.yaml` — training configuration
- `draft_config.json` — draft model architecture
- `run.sh` — launch script (starts Ray, training, and SGLang server)
- `send_requests.sh` — send traffic to the SGLang server

## Example Matrix

| Example | Model | Mode | Draft required? | Default GPU layout |
|---------|-------|------|:---------------:|--------------------|
| [qwen3-4b-external-with-draft](qwen3-4b-external-with-draft/) | Qwen3-4B | External + EAGLE3 | Yes | train: 0,1 / sglang: 2 |
| [qwen3-4b-external-no-draft](qwen3-4b-external-no-draft/) | Qwen3-4B | External, from scratch | No | train: 0,1 / sglang: 2 |
| [qwen3-8b-external-with-draft](qwen3-8b-external-with-draft/) | Qwen3-8B | External + EAGLE3 | Yes | train: 0,1 / sglang: 2,3,4,5 |
| [qwen3-8b-external-no-draft](qwen3-8b-external-no-draft/) | Qwen3-8B | External, from scratch | No | train: 0,1 / sglang: 2,3,4,5 |
| [qwen3-8b-coder-next-external-with-draft](qwen3-8b-coder-next-external-with-draft/) | Qwen3-Coder-Next 80B | External + EAGLE3 | Yes | train: 0,1 / sglang: 2,3,4,5 |
| [qwen3-8b-coder-next-external-no-draft](qwen3-8b-coder-next-external-no-draft/) | Qwen3-Coder-Next 80B | External, from scratch | No | train: 0,1 / sglang: 2,3,4,5 |
| [minimax-m21-external-with-draft](minimax-m21-external-with-draft/) | MiniMax-M2.1 229B | External + Scratch EAGLE3 | Yes | train: 0,1 / sglang: 2,3,4,5 |
| [minimax-m21-external-no-draft](minimax-m21-external-no-draft/) | MiniMax-M2.1 229B | External, from scratch | No | train: 0,1 / sglang: 2,3,4,5 |
| [qwen3-coder-next-online](qwen3-coder-next-online/) | Qwen3-Coder-Next 80B | Online | No | all GPUs shared |
| [qwen3-8b-coder-next-external-with-draft-2node](qwen3-8b-coder-next-external-with-draft-2node/) | Qwen3-Coder-Next 80B | External + EAGLE3 (2-node) | Yes | Node 1 train: 0,1 / Node 2 sglang: 0,1,2,3 |

**Shared FS** = training and SGLang server must share a filesystem for draft weight sync.

## Paper Concept Mapping

| Paper concept | Example pattern |
|---------------|-----------------|
| Day-0 deployment | `*-external-no-draft` |
| External train-with-decode | `*-external-with-draft` |
| Multi-node external training | `*-external-with-draft-2node` |
| Unified online training | `qwen3-coder-next-online` |

## Training Curves

<!-- TODO: Add training curve figures per model -->

### Qwen3-8B Domain Shift

<p align="center">
  <img src="../docs/training_curves/qwen3_8b_domain_shift.jpg" alt="Qwen3-8B Training Curves Domain Shift" width="80%">
</p>

### Qwen3-8B Ordered Stream

<p align="center">
  <img src="../docs/training_curves/qwen3_8b_ordered.png" alt="Qwen3-8B Training Curves Ordered Stream" width="80%">
</p>


## Running an Example

```bash
# 1. Start training + SGLang server
bash examples/qwen3-4b-external-no-draft/run.sh

# 2. In another terminal, send requests
bash examples/qwen3-4b-external-no-draft/send_requests.sh
```

### Multi-node (2-node) example

The `*-2node` examples split training and inference across two machines. Both nodes must share a filesystem (e.g., NFS) for the draft model checkpoint and weight sync.

A dataset must be provided to the trainer so it can build the vocab mapping before the draft model is created. The SGLang server on Node 2 runs independently — if the trainer on Node 1 crashes, the SGLang server continues serving requests unaffected.

1. **Machine 1:** Run `run_node1_train.sh`
   ```bash
   NODE2_IP=<inference-node-ip> bash examples/qwen3-8b-coder-next-external-with-draft-2node/run_node1_train.sh
   ```
2. **Machine 2:** Run `run_node2_sglang.sh` (after Node 1 prints "Node 1 is ready")
   ```bash
   NODE1_IP=<training-node-ip> bash examples/qwen3-8b-coder-next-external-with-draft-2node/run_node2_sglang.sh
   ```
3. **Machine 2:** Run `send_requests.sh`
   ```bash
   SGLANG_URL=http://<inference-node-ip>:30000 bash examples/qwen3-8b-coder-next-external-with-draft-2node/send_requests.sh
   ```

### Config overrides

The `run.sh` scripts accept OmegaConf-style overrides:

```bash
bash examples/qwen3-4b-external-with-draft/run.sh \
    output_dir=./outputs/my_experiment \
    online_serving.port=19090
```

The request scripts accept environment overrides:

```bash
NUM_SAMPLES=200 NUM_WORKERS=4 MAX_TOKENS=64 \
bash examples/qwen3-4b-external-with-draft/send_requests.sh
```

## GPU Layout

Examples assume non-overlapping GPU sets for training and inference.

| Component | Default GPUs | Environment variable |
|-----------|-------------|---------------------|
| Training (FSDP/DP) | `0,1` | `CUDA_VISIBLE_DEVICES` |
| External SGLang server | `2` (small) or `2,3,4,5` (large) | `SGLANG_GPUS` |

Do not overlap these sets unless you intentionally want contention.

## Outputs and Logs

The `run.sh` scripts write:

- Model checkpoints under `./outputs/...`
- Combined launcher logs under `./running_logs/`
- SGLang request logs under `./running_logs/requests_*`

The script prints exact log paths after startup. Keep that shell running while `send_requests.sh` is generating traffic.
