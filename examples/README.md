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

### Single-node examples (most common)

**Applies to:** all `*-external-no-draft` and `*-external-with-draft` folders except the `*-2node` variant.

**Step 1** — Launch training + sglang (stays in foreground):

```bash
bash examples/<example-folder>/run.sh
```

`run.sh` orchestrates everything in order: Ray cluster → training → mooncake → callback server → (draft model creation, if applicable) → sglang server. It stays running and waits for training to finish.

**Step 2** — In a **separate terminal**, send traffic once sglang is healthy:

```bash
bash examples/<example-folder>/send_requests.sh
```

Only run this after `run.sh` prints that the sglang server is healthy.

### 2-node examples

**Applies to:** `qwen3-8b-coder-next-external-with-draft-2node`, `kimi-k25-nvfp4-external-no-draft`.

Both nodes must share a filesystem (e.g., NFS) for the draft model checkpoint and weight sync.

**Step 1** — On **Node 1** (trainer machine), launch training:

```bash
NODE2_IP=<inference-node-ip> bash examples/<example-folder>/run_node1_train.sh   # or run_trainer.sh
```

Wait until it prints **"Training callback server is ready"**.

**Step 2** — On **Node 2** (inference machine), launch sglang:

```bash
NODE1_IP=<training-node-ip> bash examples/<example-folder>/run_node2_sglang.sh   # or run_sglang.sh
```

This connects back to Node 1's mooncake/callback server. Wait until sglang is healthy.

**Step 3** — Send traffic (from either node):

```bash
SGLANG_URL=http://<inference-node-ip>:30000 \
  bash examples/<example-folder>/send_requests.sh
```

### Online training (no external sglang)

**Applies to:** `qwen3-coder-next-online`.

```bash
bash examples/qwen3-coder-next-online/run.sh
```

That's it — inference runs embedded inside the training process via `SglEngine`, so there is no separate sglang server and no `send_requests.sh`.

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
