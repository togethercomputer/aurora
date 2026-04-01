# Ray Architecture in Aurora

Aurora uses Ray as its distributed orchestration layer. In **online training** mode, all GPU workloads (inference, training, mooncake master) run as Ray actors. In **external training** mode, only training and the callback server run as Ray actors — the sglang inference server runs as a separate process outside Ray.

## Package Layout & Actor Hierarchy

`RayActor` is the base class for all GPU-bound actors. It provides GPU setup, IP discovery, and port allocation so each actor doesn't reinvent them.

```
aurora/ray/
├── ray_actor.py                    RayActor base class
├── train_group.py                  RayTrainGroup (training actor group manager)
└── placement_group.py              Placement group creation & GPU resource management

aurora/inference/engine/
├── hf_engine.py                    HFEngine(InferenceEngine, RayActor)
└── sgl_engine.py                   SglEngine(InferenceEngine, RayActor)

aurora/training/
├── trainer.py                      Trainer (ABC base)
├── trainer_actor.py                TrainerActor(RayActor) — wraps Eagle3Trainer
└── eagle3_trainer.py               Eagle3Trainer(Trainer) — FSDP2 training logic

aurora/transfer/mooncake/
└── utils.py                        MooncakeMaster(RayActor)

aurora/controller/
├── training_controller.py          AsyncTrainingController (standalone Ray actor)
├── inference_manager.py            AsyncInferenceManager (standalone Ray actor)
└── training_external_server.py     TrainingExternalServer (Ray actor — HTTP callback server for external sglang)
```

## Placement Groups

Placement groups reserve GPUs for training and inference as a unit and place them on the correct nodes. `create_placement_groups(args)` is the single entry point.

| Mode | Training GPUs | Inference GPUs | Use case |
|------|--------------|----------------|----------|
| Default (separate) | Dedicated PG | Dedicated PG | Production: no GPU contention |
| `colocate` | Shared PG | Shared PG | Dev: share GPUs between train & inference |
| `debug_train_only` | Dedicated PG | Empty | Debug training without inference |
| `debug_inference_only` | Empty | Dedicated PG | Debug inference without training |

Each placement group probes bundles with a temporary `InfoActor` to discover the actual (node IP, GPU ID) mapping, then sorts by (node, GPU ID) for deterministic ordering.

## Ray Cluster Setup

Aurora connects to Ray via `_ensure_ray_initialized()` in `placement_group.py`. It reads the `RAY_ADDRESS` environment variable (defaulting to `"auto"`) and calls `ray.init(address=...)`. If no cluster is found, it falls back to starting a local instance.

### Single-node (local)

No setup needed — the example `run.sh` scripts run `ray stop --force` and start a fresh local Ray instance automatically.

For **online training**, `CUDA_VISIBLE_DEVICES` controls all GPUs (training + inference):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash examples/qwen3-coder-next-online/run.sh
```

For **external training**, `CUDA_VISIBLE_DEVICES` controls training GPUs and `SGLANG_GPUS` controls the standalone sglang server (they must not overlap):

```bash
CUDA_VISIBLE_DEVICES=0,1 SGLANG_GPUS=2,3,4,5 bash examples/qwen3-8b-external-with-draft/run.sh
```

> **Note:** On shared machines, `auto` may attach to another user's cluster. The
> example scripts set `RAY_ADDRESS` explicitly to avoid this. If running manually,
> use `RAY_ADDRESS=local` to force a fresh local instance.

### Multi-node (local cluster)

Start Ray manually before launching Aurora. Run these commands on each node:

**1. Head node** (run first):

```bash
ray start --head \
  --port 6379 \
  --node-ip-address <HEAD_IP> \
  --num-gpus <N> \
  --temp-dir /tmp/ray_$(id -u) \
  --disable-usage-stats
```

**2. Worker nodes** (run after head is up):

```bash
ray start \
  --address <HEAD_IP>:6379 \
  --num-gpus <N> \
  --temp-dir /tmp/ray_$(id -u) \
  --disable-usage-stats
```

**3. Run Aurora on the head node:**

> **Important:** The example `run.sh` scripts are designed for single-node use —
> they run `ray stop --force` and start their own local head node. For multi-node,
> invoke `aurora.train_entry` directly against the pre-existing cluster:

```bash
export RAY_ADDRESS=<HEAD_IP>:6379
python3 -m aurora.train_entry --config <your_config.yaml> [overrides...]
```

Aurora auto-detects the cluster via `RAY_ADDRESS`. Worker nodes don't
need to be up before the script starts — `_wait_for_gpu_resources()` will block
for up to 300 seconds until all expected GPUs are visible in the cluster.

### Kubernetes

On Kubernetes, we recommend using the [KubeRay operator](https://ray-project.github.io/kuberay/)
to manage the Ray cluster lifecycle. KubeRay handles head/worker pod scheduling,
autoscaling, and fault recovery. Once the `RayCluster` resource is running,
point Aurora at it directly (do not use the example `run.sh` scripts, which
manage their own local cluster):

```bash
export RAY_ADDRESS=ray://<kuberay-head-svc>:10001
python3 -m aurora.train_entry --config <your_config.yaml> [overrides...]
```

### NCCL / Gloo networking

On multi-NIC machines, set the network interface explicitly:

```bash
export NCCL_SOCKET_IFNAME=<iface>   # e.g. eth0
export GLOO_SOCKET_IFNAME=<iface>
export TP_SOCKET_IFNAME=<iface>
```

Find your interface with `ip -o addr show | grep <your_node_ip>`.

## Multi-Node Training & Inference Config

### Training across nodes

`RayTrainGroup` creates `training_num_nodes × training_num_gpus_per_node` actors.
The PACK placement strategy spreads them across nodes automatically.

| Key | Default | Description |
|-----|---------|-------------|
| `training.training_num_nodes` | 1 | Number of training nodes |
| `training.training_num_gpus_per_node` | 1 | GPUs per training node |

### Inference across nodes (SglEngine multi-node TP)

When a single model is too large for one node, SglEngine supports multi-node
tensor parallelism via `inference.sglang.nnodes`.

```
Example: 16-GPU TP across 2 nodes, 8 GPUs each

  inference.inference_num_gpus=16, inference.sglang.nnodes=2, inference.inference_num_gpus_per_node=8

  Factory creates 2 SglEngine actors (one per node):
    engine 0: node_rank=0 (head)   — accepts generate() calls
    engine 1: node_rank=1 (worker) — participates in NCCL TP only
```

| Key | Default | Description |
|-----|---------|-------------|
| `inference.sglang.nnodes` | 1 | Nodes per inference replica |
| `inference.inference_num_gpus` | 1 | Total inference GPUs across all nodes |
| `inference.inference_num_gpus_per_node` | 8 | GPUs per inference node |
| `inference.sglang.dist_init_addr` | auto | Override dist init address (auto-negotiated if unset) |
| `inference.sglang.dist_timeout` | 60 | Dist init timeout in seconds |

### Example: single-node online training layout

```
GPUs 0-1: training (FSDP/DP, 2 TrainerActor)
GPUs 2-5: inference (SglEngine TP=4)

training.training_num_gpus_per_node=2
inference.inference_num_gpus=4, inference.sglang.tp_size=4
```

### Example: single-node external training layout

```
GPUs 0-1: training (FSDP/DP, 2 TrainerActor) — managed by Ray
GPUs 2-5: external sglang server (TP=4)      — separate process, not managed by Ray

training.training_num_gpus_per_node=2
inference.inference_num_gpus=0  # no local engines
```

### Example: multi-node online training layout

```
Node 0 (head):   Ray head + 4 training GPUs
Node 1 (worker): 8 inference GPUs (TP node_rank=0, head)
Node 2 (worker): 8 inference GPUs (TP node_rank=1, worker)

training.training_num_nodes=1, training.training_num_gpus_per_node=4
inference.inference_num_gpus=16, inference.sglang.nnodes=2, inference.inference_num_gpus_per_node=8
```
