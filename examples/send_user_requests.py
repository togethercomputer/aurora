#!/usr/bin/env python3
"""
Simulate user traffic by sending dataset prompts to a sglang server.

Reads a JSONL dataset, applies chat template, and sends prompts as regular
user requests to the sglang server. The server's callback mechanism handles
storing hidden states and pushing training samples automatically.

Usage:
    python recipes/send_user_requests.py \
        --dataset /home/bbie/torchspec-aurora/datasets/onlinesd/merged/merged_code_train_shuffled.jsonl \
        --server-url http://localhost:30000 \
        --num-workers 12 \
        --model Qwen/Qwen3-Coder-Next --max-tokens 512

    # Limit to first 100 samples:
    python scripts/send_user_requests.py --dataset data.jsonl --num-samples 100

    # Continuous mode (loop forever over dataset):
    python scripts/send_user_requests.py --dataset data.jsonl --loop
"""

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(path: str, prompt_key: str = "conversations"):
    """Load JSONL dataset and return list of conversation message lists."""
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            convs = item.get(prompt_key)
            if convs and isinstance(convs, list):
                samples.append({"id": item.get("id", f"sample_{i}"), "messages": convs})
            elif "prompt" in item:
                samples.append(
                    {
                        "id": item.get("id", f"sample_{i}"),
                        "messages": [{"role": "user", "content": item["prompt"]}],
                    }
                )
            elif "text" in item:
                samples.append(
                    {
                        "id": item.get("id", f"sample_{i}"),
                        "messages": [{"role": "user", "content": item["text"]}],
                    }
                )
    return samples


def _normalize_conversation(conversation):
    """Normalize ShareGPT format (from/value) to standard (role/content).
    Mirrors aurora.data.preprocessing._normalize_conversation."""
    ROLE_MAPPING = {"human": "user", "gpt": "assistant"}
    if not conversation:
        return conversation
    first_msg = conversation[0]
    if "role" in first_msg and "content" in first_msg:
        return conversation
    if "from" in first_msg and "value" in first_msg:
        return [
            {"role": ROLE_MAPPING.get(msg["from"], msg["from"]), "content": msg["value"]}
            for msg in conversation
        ]
    return conversation


def _strip_trailing_assistant(messages):
    """Remove trailing assistant messages so the model generates the response.
    For multi-turn, keeps all messages up to and including the last user message."""
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            last_user_idx = i
    if last_user_idx >= 0:
        return messages[: last_user_idx + 1]
    return messages


def format_prompt(messages, tokenizer, chat_template_name="qwen"):
    """Apply chat template matching aurora's GeneralParser.format().

    Steps (mirroring aurora/data/parse.py GeneralParser.format):
    1. Normalize ShareGPT format
    2. Inject system prompt from template if not present
    3. Strip trailing assistant messages (we want the model to generate)
    4. Apply tokenizer chat template with add_generation_prompt=True
    """
    from aurora.data.template import TEMPLATE_REGISTRY

    messages = _normalize_conversation(messages)
    messages = _strip_trailing_assistant(messages)

    template = TEMPLATE_REGISTRY.get(chat_template_name)
    formatted_messages = []

    # Inject system prompt if conversation doesn't start with one
    if messages and messages[0]["role"] == "system":
        formatted_messages.append(messages[0])
        messages = messages[1:]
    elif template.system_prompt:
        formatted_messages.append({"role": "system", "content": template.system_prompt})

    formatted_messages.extend(messages)

    text = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def send_request(server_url, messages, model, max_tokens, temperature, timeout):
    """Send a single request via OpenAI-compatible /v1/chat/completions endpoint."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    t0 = time.time()
    resp = requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()
    elapsed = time.time() - t0

    usage = result.get("usage", {})
    return {
        "elapsed": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "output_len": usage.get("completion_tokens", 0),
    }


def wait_for_server(server_url, max_wait=300):
    """Wait for the sglang server to be healthy."""
    logger.info(f"Waiting for server at {server_url}/health ...")
    for attempt in range(max_wait // 2):
        try:
            resp = requests.get(f"{server_url}/health", timeout=5)
            if resp.status_code == 200:
                logger.info("Server is healthy")
                return True
        except requests.RequestException:
            pass
        if attempt % 15 == 0 and attempt > 0:
            logger.info(f"Still waiting for server... ({attempt * 2}s)")
        time.sleep(2)
    logger.error(f"Server not reachable after {max_wait}s")
    return False


def main():
    parser = argparse.ArgumentParser(description="Send user requests to sglang server")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--server-url", default="http://localhost:30000")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-Coder-Next", help="Model name for tokenizer/chat template"
    )
    parser.add_argument(
        "--prompt-key", default="conversations", help="Key in JSONL for conversation messages"
    )
    parser.add_argument(
        "--chat-template",
        default="qwen",
        help="(Unused, kept for backward compat) Chat template name",
    )
    parser.add_argument(
        "--num-samples", type=int, default=0, help="Number of samples to send (0 = all)"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--loop", action="store_true", help="Loop over dataset continuously")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--delay", type=float, default=0.0, help="Delay between submitting requests (seconds)"
    )
    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    samples = load_dataset(args.dataset, prompt_key=args.prompt_key)
    logger.info(f"Loaded {len(samples)} samples")

    if not samples:
        logger.error("No samples found in dataset")
        return 1

    # Prepare messages (normalize + strip trailing assistant)
    logger.info("Preparing messages...")
    prompts = []
    for s in samples:
        try:
            messages = _normalize_conversation(s["messages"])
            messages = _strip_trailing_assistant(messages)
            prompts.append({"id": s["id"], "messages": messages})
        except Exception as e:
            logger.warning(f"Failed to prepare sample {s['id']}: {e}")
    logger.info(f"Prepared {len(prompts)} prompts (thinking enabled, OpenAI format)")

    if args.shuffle:
        random.shuffle(prompts)

    if args.num_samples > 0:
        prompts = prompts[: args.num_samples]

    # Wait for server
    if not wait_for_server(args.server_url):
        return 1

    # Send requests
    round_num = 0
    while True:
        round_num += 1
        batch = list(prompts)
        if args.shuffle and round_num > 1:
            random.shuffle(batch)

        logger.info(
            f"{'=' * 50}\n"
            f"Round {round_num}: sending {len(batch)} requests "
            f"with {args.num_workers} workers\n"
            f"  max_tokens={args.max_tokens}, temperature={args.temperature}\n"
            f"{'=' * 50}"
        )

        success = 0
        fail = 0
        total_elapsed = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        t_start = time.time()

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for i, prompt in enumerate(batch):
                if args.delay > 0 and i > 0:
                    time.sleep(args.delay)
                f = executor.submit(
                    send_request,
                    args.server_url,
                    prompt["messages"],
                    args.model,
                    args.max_tokens,
                    args.temperature,
                    args.timeout,
                )
                futures[f] = i

            for future in as_completed(futures):
                try:
                    result = future.result()
                    success += 1
                    total_elapsed += result["elapsed"]
                    total_prompt_tokens += result["prompt_tokens"]
                    total_completion_tokens += result["completion_tokens"]

                    if success % args.log_interval == 0:
                        wall = time.time() - t_start
                        logger.info(
                            f"Progress: {success}/{len(batch)} "
                            f"({fail} failed) | "
                            f"rate={success / wall:.1f} req/s | "
                            f"avg_latency={total_elapsed / success:.2f}s | "
                            f"tokens={total_prompt_tokens}+{total_completion_tokens}"
                        )
                except Exception as e:
                    fail += 1
                    logger.warning(f"Request failed: {e}")

        wall = time.time() - t_start
        logger.info(
            f"\nRound {round_num} complete: "
            f"{success} succeeded, {fail} failed, "
            f"{wall:.1f}s wall time"
        )
        if success > 0:
            logger.info(
                f"  Throughput: {success / wall:.1f} req/s, "
                f"Avg latency: {total_elapsed / success:.2f}s, "
                f"Tokens: {total_prompt_tokens} prompt + {total_completion_tokens} completion"
            )

        if not args.loop:
            break

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
