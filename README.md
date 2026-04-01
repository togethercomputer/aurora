# Aurora

Aurora is a unified training-serving system for online speculative decoding. It closes the loop between speculator training and serving by continuously learning a draft model directly from live inference traces — treating online speculator learning as an asynchronous reinforcement-learning problem. Aurora is built on top of [TorchSpec](https://github.com/xwuShirley/aurora).

Aurora supports **day-0 deployment**: a speculator can be served immediately and rapidly adapted to live traffic, improving system performance while providing immediate utility feedback. Across experiments, Aurora achieves a **1.5x day-0 speedup** on recently released frontier models (e.g., MiniMax-M2.1 and Qwen3-Coder-Next), and adapts effectively to distribution shifts in user traffic, delivering an additional **1.25x speedup** over a well-trained but static speculator on widely used models (e.g., Qwen3).

<p align="center">
  <img src="docs/diagram.png" alt="Aurora Architecture" width="100%">
</p>


## Deployment Modes

| Mode | Description |
|------|-------------|
| **Online** | Training and inference co-located via Ray controller. Draft model updated continuously from live serving traces with hot-swapped weight sync. |
| **External with draft** | Standalone SGLang server with EAGLE3 speculative decoding. Training improves the draft and syncs weights back periodically. |
| **External without draft** | Standalone SGLang server runs target-only inference. Draft model trained from scratch — no pre-existing data or speculator required. |

## Setup

```bash
./tools/build_conda.sh
micromamba activate aurora
```

To install into your current environment instead:

```bash
./tools/build_conda.sh current
```

Optional Flash Attention extras:

```bash
pip install -e ".[fa]"
```

## Quick Start

```bash
# Start training + external SGLang server (Qwen3-4B, day-0 from scratch)
bash examples/qwen3-4b-external-no-draft/run.sh

# In another terminal, send requests to generate training samples
bash examples/qwen3-4b-external-no-draft/send_requests.sh
```

See [`examples/README.md`](examples/README.md) for the full example catalog, per-model training curves, GPU layout, and config overrides.

## Production Notes

- The example `run.sh` scripts are **single-node oriented** — they manage their own local Ray cluster. For multi-node or Kubernetes deployments, start Ray manually and invoke `python3 -m aurora.train_entry` directly. See [docs/ray.md](docs/ray.md).
- **External with-draft** mode requires a **shared filesystem** between training and the SGLang server for draft weight sync.
- `online_serving.hidden_states_dtype` must match the serving model's dtype (e.g., set `float16` when serving an FP8 model).
- Training and inference GPU sets (`CUDA_VISIBLE_DEVICES` vs `SGLANG_GPUS`) **must not overlap**.

## Checkpoint Conversion

Convert an Aurora checkpoint to Hugging Face format:

```bash
python tools/convert_to_hf.py --input-dir ./outputs/my_experiment/iter_0010000/
```

Vocabulary pruning can be applied either during training (`draft_vocab_size` in config) or at conversion time:

```bash
python tools/convert_to_hf.py \
    --input-dir ./outputs/my_experiment/iter_0010000/ \
    --prune-vocab \
    --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
    --draft-vocab-size 32000 \
    --tokenizer Qwen/Qwen3-8B \
    --chat-template qwen \
    --prompt-key conversations
```
## Metrics Reporting

W&B logging is disabled by default (report_to: none). To enable it, set report_to: wandb in your config and supply your API key.

## Troubleshooting

| Issue | Reference |
|-------|-----------|
| Stuck or failing distributed runs, Ray actor errors | [docs/debugging_ray_jobs.md](docs/debugging_ray_jobs.md) |
| Ray cluster setup, actor hierarchy, placement groups | [docs/ray.md](docs/ray.md) |
| Pipeline bottlenecks, slow steps, throughput analysis | [docs/performance_metrics.md](docs/performance_metrics.md) |

Enable verbose logging:

```bash
AURORA_LOG_LEVEL=DEBUG bash examples/qwen3-4b-external-with-draft/run.sh
```

## Citation

```bibtex
@article{wang2026aurora,
  title={When RL Meets Adaptive Speculative Training: A Unified Training--Serving System},
  author={Wang, Junxiong and Bie, Fengxiang and Li, Jisen and Zhou, Zhongzhu and Shao, Zelei and Wang, Yubo and Liu, Yinghui and Wu, Qingyang and May, Avner and Yanamandra, Sri and Zhang, Yineng and Zhang, Ce and Dao, Tri and Liang, Percy and Athiwaratkun, Ben and Song, Shuaiwen Leon and Xu, Chenfeng and Wu, Xiaoxia},
  journal={arXiv preprint arXiv:2602.06932},
  year={2026}
}
```
