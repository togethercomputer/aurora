# Debugging Ray Jobs in Aurora

Reference for diagnosing stuck, slow, or failing distributed training runs.

## Log Locations

### 1. Main training log (driver stdout/stderr)

All recipe scripts tee output to `running_logs/`:

```bash
ls -lt running_logs/ | head -5
tail -f running_logs/<latest>.log
```

The filename encodes the example and timestamp, e.g. `external_sglang_20260301_080012.log`.

### 2. Ray worker logs (per-actor, most detailed)

Each Ray actor gets its own `worker-*.out` (stdout) and `worker-*.err` (stderr) files. Application-level logs (Python `logging`) go to **stderr**.

The key actors by deployment mode:

| Actor | Online training | External training |
|-------|:-:|:-:|
| TrainerActor | Yes | Yes |
| SglEngine | Yes | No |
| AsyncTrainingController | Yes | Yes |
| AsyncInferenceManager | Yes | No |
| TrainingExternalServer | No | Yes |
| MooncakeMaster | Yes | Yes |

The default Ray temp dir is `/tmp/ray`. Example scripts may override this.

```bash
RAY_TEMP_DIR="/tmp/ray"

# List most recently active workers
ls -lt "$RAY_TEMP_DIR/session_latest/logs/worker-*.err" | head -10

# Tail all worker stderr (noisy but complete)
tail -f "$RAY_TEMP_DIR/session_latest/logs/worker-*.err"

# Find a specific actor type
grep -l "TrainerActor\|SglEngine\|MooncakeMaster\|TrainingController\|ExternalServer" \
    "$RAY_TEMP_DIR/session_latest/logs/worker-*.err"
```

### 3. Ray system logs

```bash
RAY_TEMP_DIR="/tmp/ray"

# Raylet health (updates every few seconds when cluster is alive)
ls -lt "$RAY_TEMP_DIR/session_latest/logs/raylet.out"

# GCS server
tail "$RAY_TEMP_DIR/session_latest/logs/gcs_server.out"

# Monitor log (cluster-level events, autoscaling, errors)
tail "$RAY_TEMP_DIR/session_latest/logs/monitor.log"
```

### 4. Environment variable for log verbosity

```bash
export AURORA_LOG_LEVEL=DEBUG   # default is INFO; DEBUG gives per-step detail
export NCCL_DEBUG=INFO             # NCCL connection/transport debugging
```

## Startup Sequence & What to Expect

The training entry point (`aurora/train_entry.py → train_async_no_generation`) proceeds through these phases. If the job appears stuck, identify which phase it's in.

### Online training (internal SglEngine)

| Phase | What happens | Visible in main log | Typical duration |
|-------|-------------|-------------------|-----------------|
| 1. Config & dataset | Parse YAML, load dataset | `Added N samples to controller` | seconds |
| 2. Placement groups | Reserve GPUs, probe node/GPU mapping | `bundle N, actual_bundle_index: M, node: ...` | seconds |
| 3. Mooncake master | Launch mooncake subprocess actor | `Auto-resolved mooncake master_server_address` | seconds |
| 4. TrainerActor init | NCCL process group, FSDP2 model init | `Device mesh (1D)`, `Draft model: N trainable` (in worker-*.err) | ~1 min |
| 5. SglEngine init | **Model shard loading** → KV cache → CUDA graphs | `Multi-thread loading shards: N%` | **5–15 min for large models** |
| 6. Queue wiring | Connect mooncake queues between inference and training | `set_train_queue` | seconds |
| 7. Training loop | Dispatch + train steps | `Training: N%` progress bar | ongoing |

**Phase 5 is the most common "looks stuck" phase.** After shards finish loading (100%), SGLang silently allocates KV cache and captures CUDA graphs — no output for several minutes.

### External training (standalone sglang server)

| Phase | What happens | Visible in main log | Typical duration |
|-------|-------------|-------------------|-----------------|
| 1. Config & dataset | Parse YAML, load dataset | `Added N samples to controller` | seconds |
| 2. Placement groups | Reserve training GPUs only | `bundle N, actual_bundle_index: M, node: ...` | seconds |
| 3. Mooncake master | Launch mooncake subprocess actor | `Auto-resolved mooncake master_server_address` | seconds |
| 4. TrainerActor init | NCCL process group, FSDP2 model init | `Device mesh (1D)`, `Draft model: N trainable` (in worker-*.err) | ~1 min |
| 5. Callback server | Start TrainingExternalServer HTTP server | `TrainingExternalServer started at http://...` | seconds |
| 6. Wait for sglang | External sglang server starts on separate GPUs | Visible in sglang log (separate file) | **5–15 min for large models** |
| 7. Training loop | Waits for samples from external sglang via HTTP callback | `Received sample: data_id=...` | ongoing |

In external mode, phases 5–6 happen outside Aurora. The `run.sh` script starts the sglang server separately and waits for it to be healthy before sending requests.

## Quick Diagnostic Commands

### Is the cluster alive?

```bash
ray status   # or: ray status --address 127.0.0.1:6380
```

### Is anything still running?

```bash
RAY_TEMP_DIR="/tmp/ray"

# If raylet.out modification time is advancing, the cluster is alive
stat "$RAY_TEMP_DIR/session_latest/logs/raylet.out" | grep Modify

# Check if worker logs are still being written
ls -lt "$RAY_TEMP_DIR/session_latest/logs/worker-*.err" | head -3
```

### What phase are we in?

```bash
# Check the main log's last lines
tail -20 running_logs/<latest>.log

# For external training, also check the sglang server log
tail -20 running_logs/<sglang_latest>.log

# Check the most recent worker stderr for actual progress
tail -30 "$(ls -t $RAY_TEMP_DIR/session_latest/logs/worker-*.err | head -1)"
```

### GPU utilization (are GPUs doing work?)

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
# or watch continuously:
watch -n 2 nvidia-smi
```

### Check for OOM or crashes in worker logs

```bash
grep -i "error\|OOM\|killed\|exception\|traceback" "$RAY_TEMP_DIR/session_latest/logs/worker-*.err" | tail -20
```

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Ray cluster GPUs: N/M` stuck waiting | Stale Ray cluster from a previous run with wrong GPU count | Run `ray stop --force` and restart; example `run.sh` scripts now do this automatically |
| Stuck after placement group creation | Workers haven't joined Ray cluster yet | Check `ray status` for expected GPU count; verify worker nodes ran `ray start` |
| `Multi-thread loading shards` stalls at <100% | Slow NFS/disk or OOM during model load | Check `dmesg` for OOM kills; check disk I/O with `iostat` |
| Shards 100% complete but no further output for >10 min | CUDA graph capture hung or KV cache OOM | Check `nvidia-smi` for GPU memory; check worker-*.err for CUDA errors |
| `NCCL timeout` or `dist.init_process_group` hangs | Network interface mismatch across nodes | Ensure `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME` match on all nodes |
| `Failed to establish connection to the metrics exporter agent` | Harmless Ray metrics warning | Ignore — does not affect training |
| `RayActorError` in main log | Actor crashed on remote node | Find the specific worker-*.err by PID from the error message |
| Training runs but loss is NaN | Gradient explosion or data issue | Check `max_grad_norm`, inspect data with `save_debug_train_data` |
| (External) Callback server up but no samples arriving | sglang server not started, wrong callback URL, or mooncake connection issue | Check sglang log for errors; verify `--spec-training-callback-url` matches callback server address |
| (External) `accept_len=1.0` after weight sync | Dtype mismatch between sglang and training | Check `online_serving.hidden_states_dtype` matches the sglang target model dtype (e.g. `float16` for FP8 models) |

## External Training Specifics

- The standalone sglang server runs in a **separate process** with its own log file (e.g. `running_logs/sglang_server_*.log`). Always check both the training log and the sglang log.
- The `send_requests.sh` script sends traffic to the sglang server. Training won't progress until requests start flowing.
- Draft weight sync (with-draft mode) writes checkpoints to a local directory and POSTs the path to sglang. Both processes must share the same filesystem.

## Multi-Node Specifics

- **Worker node logs** are on the worker machines, not the head node. SSH to the worker and check its `$RAY_TEMP_DIR/session_latest/logs/`.
- Ray deduplicates repeated worker output in the driver log (`[repeated Nx across cluster]`), so the main log may hide individual node progress. Always check worker-*.err on each node.
- The PID and IP shown in the main log (e.g. `(SglEngine pid=514428, ip=10.0.0.2)`) let you identify which node to SSH to.
