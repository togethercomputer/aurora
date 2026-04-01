# Copyright (c) 2026 Together Compute
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import hashlib
import logging as _logging
import multiprocessing as mp
import os

import torch
from tqdm import tqdm

from aurora.data.preprocessing import _normalize_conversation
from aurora.data.template import TEMPLATE_REGISTRY
from aurora.data.utils import (
    estimate_row_count,
    extract_media_urls,
    flatten_multimodal_content,
    load_hf_dataset,
    pack_loss_mask,
    serialize_packed_loss_mask,
)
from aurora.utils.logging import logger

_logging.getLogger("transformers_modules").setLevel(_logging.ERROR)

_worker_state = {}


def _init_tokenize_worker(
    tokenizer_path, trust_remote_code, chat_template_name, last_turn_loss_only=False
):
    """Initializer for each worker process — loads tokenizer once."""
    from aurora.data.preprocessing import preprocess_conversations
    from aurora.utils.processing import load_tokenizer

    _logging.getLogger("transformers_modules").setLevel(_logging.ERROR)
    _worker_state["tokenizer"] = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    _worker_state["template"] = TEMPLATE_REGISTRY.get(chat_template_name)
    _worker_state["preprocess"] = preprocess_conversations
    _worker_state["last_turn_loss_only"] = last_turn_loss_only


def _tokenize_single(args):
    """Worker function — tokenize one sample."""
    messages, max_length, train_with_decode = args
    processed = _worker_state["preprocess"](
        _worker_state["tokenizer"],
        [messages],
        _worker_state["template"],
        max_length=max_length,
        is_preformatted=False,
        include_attention_mask=False,
        use_packed_loss_mask=True,
        add_generation_prompt=train_with_decode,
        return_formatted_text=True,
        last_turn_loss_only=_worker_state.get("last_turn_loss_only", False),
    )
    if not processed["input_ids"]:
        return None
    # Return plain lists instead of tensors to avoid shared memory mmap
    # exhaustion when transferring results across process boundaries.
    input_ids = processed["input_ids"][0]
    return {
        "input_ids": input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids,
        "packed_loss_mask": processed["packed_loss_mask"][0],
        "formatted_prompt": processed["formatted_text"][0],
    }


def _init_format_worker(tokenizer_path, trust_remote_code, chat_template_name):
    from aurora.data.parse import create_parser
    from aurora.utils.processing import load_tokenizer

    _logging.getLogger("transformers_modules").setLevel(_logging.ERROR)
    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    _worker_state["template"] = TEMPLATE_REGISTRY.get(chat_template_name)
    _worker_state["parser"] = create_parser(tokenizer, _worker_state["template"])


def _format_single(args):
    """
    Worker function — format only, skip tokenization.
    """
    messages, _ = args
    messages = _normalize_conversation(messages)
    parser = _worker_state["parser"]
    formatted = parser.format(messages)
    if not formatted:
        return None
    return {"formatted_prompt": formatted}


def _load_pretokenized_dataset(args):
    """Load a pre-tokenized dataset (input_ids/labels/attention_mask already computed).

    Expects HF dataset with columns: input_ids (list[int]), labels (list[int]),
    and optionally attention_mask (list[int]). Labels use -100 for ignored tokens.

    Returns list of dicts with: data_id, input_ids, packed_loss_mask, formatted_prompt,
    multimodal_inputs, metadata.
    """
    max_length = args.max_seq_length
    hf_dataset = load_hf_dataset(args.train_data_path)

    dataset_name = os.path.basename(args.train_data_path)
    file_stat = ""
    if os.path.isfile(args.train_data_path):
        st = os.stat(args.train_data_path)
        file_stat = f"-{st.st_size}-{st.st_mtime}"
    cache_params = f"{dataset_name}-{args.train_data_path}{file_stat}-pretokenized-{max_length}"
    cache_key = hashlib.md5(cache_params.encode()).hexdigest()
    cache_dir = os.path.join(getattr(args, "cache_dir", "./cache"), "tokenized_dataset")
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

    if os.path.exists(cache_path):
        logger.info(f"Loading pretokenized dataset from cache: {cache_path}")
        prompts = torch.load(cache_path, weights_only=False)
        logger.info(f"Loaded {len(prompts)} cached samples")
        return prompts

    logger.info(f"Loading pretokenized dataset (cache will be saved to {cache_path})")

    total_estimate = estimate_row_count(args.train_data_path)
    prompts = []
    skipped = 0

    for idx, sample in enumerate(
        tqdm(hf_dataset, desc="Loading pretokenized samples", total=total_estimate)
    ):
        input_ids = sample.get("input_ids")
        labels = sample.get("labels")
        if input_ids is None or labels is None:
            raise ValueError(
                f"Pretokenized dataset sample {idx} missing 'input_ids' or 'labels'. "
                f"Available keys: {list(sample.keys())}"
            )

        # Truncate to max_length
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

        if len(input_ids) == 0:
            skipped += 1
            continue

        # Convert labels (-100 = ignore) to binary loss mask, then pack
        loss_mask = torch.tensor([0 if lab == -100 else 1 for lab in labels], dtype=torch.long)
        if loss_mask.sum() == 0:
            skipped += 1
            continue

        packed = pack_loss_mask(loss_mask)
        packed_str = serialize_packed_loss_mask(packed)

        prompts.append(
            {
                "data_id": sample.get("id", f"sample_{idx}"),
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "packed_loss_mask": packed_str,
                "formatted_prompt": None,
                "multimodal_inputs": None,
                "metadata": {},
            }
        )

    if skipped:
        logger.warning(f"Skipped {skipped} pretokenized samples (empty or zero loss mask)")

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(prompts, cache_path)
    logger.info(f"Saved {len(prompts)} pretokenized samples to cache: {cache_path}")
    return prompts


def load_conversation_dataset(args):
    """Load conversation dataset and optionally tokenize for training.

    When pretokenized=True, loads pre-tokenized data (input_ids/labels columns)
    directly, skipping conversation parsing and tokenization entirely.

    When defer_tokenization=True, only applies the chat template to produce
    formatted text — no tokenizer is loaded and no input_ids/loss_mask are
    generated. The inference engine handles tokenization and media token
    expansion; loss mask is computed at training time from the engine's
    actual input_ids.

    When defer_tokenization=False (default), fully tokenizes and produces
    input_ids + packed_loss_mask for the input_ids engine path.

    Returns list of dicts. Fields depend on mode:
        pretokenized=True:        data_id, input_ids, packed_loss_mask, formatted_prompt, multimodal_inputs, metadata
        defer_tokenization=True:  data_id, formatted_prompt, multimodal_inputs, metadata
        defer_tokenization=False: data_id, input_ids, packed_loss_mask, formatted_prompt, multimodal_inputs, metadata
    """
    pretokenized = getattr(args, "pretokenized", False)
    if pretokenized:
        return _load_pretokenized_dataset(args)

    prompt_key = getattr(args, "prompt_key", "text")
    chat_template_name = getattr(args, "chat_template", None)
    max_length = args.max_seq_length
    defer_tokenization = getattr(args, "defer_tokenization", False)

    logger.info(f"Max sequence length allowed for training: {max_length}")

    if not chat_template_name:
        raise ValueError("chat_template must be set for load_conversation_dataset")

    custom_template = TEMPLATE_REGISTRY.get(chat_template_name)
    hf_dataset = load_hf_dataset(args.train_data_path)

    dataset_name = os.path.basename(args.train_data_path)
    file_stat = ""
    if os.path.isfile(args.train_data_path):
        st = os.stat(args.train_data_path)
        file_stat = f"-{st.st_size}-{st.st_mtime}"
    train_with_decode = getattr(args, "train_with_decode", False)
    last_turn_loss_only_flag = getattr(args, "last_turn_loss_only", False)
    cache_params = (
        f"{dataset_name}-{args.train_data_path}{file_stat}-{args.target_model_path}"
        f"-{max_length}-{chat_template_name}-ltlo={last_turn_loss_only_flag}"
        f"-defer={defer_tokenization}-decode={train_with_decode}"
    )
    cache_key = hashlib.md5(cache_params.encode()).hexdigest()
    cache_dir = os.path.join(getattr(args, "cache_dir", "./cache"), "tokenized_dataset")
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

    if os.path.exists(cache_path):
        logger.info(f"Loading dataset from cache: {cache_path}")
        prompts = torch.load(cache_path, weights_only=False)
        logger.info(f"Loaded {len(prompts)} cached samples")
        return prompts

    mode_label = "Formatting" if defer_tokenization else "Tokenizing"
    logger.info(f"{mode_label} dataset (cache will be saved to {cache_path})")

    total_estimate = estimate_row_count(args.train_data_path)
    num_proc = getattr(args, "num_proc", 64)

    # Pass 1: collect and normalize raw samples (fast I/O, no tokenization)
    raw_samples = []
    for idx, sample in enumerate(tqdm(hf_dataset, desc="Loading samples", total=total_estimate)):
        raw_prompt = sample.get(prompt_key, "")

        if not isinstance(raw_prompt, list):
            raise ValueError(
                f"Expected conversation format (list of messages) for sample {idx}, got {type(raw_prompt)}"
            )

        messages = _normalize_conversation(raw_prompt)
        multimodal_inputs = extract_media_urls(messages)
        # In decode mode, extract only the first user prompt for generation
        if train_with_decode:
            first_user = next((m for m in messages if m["role"] == "user"), None)
            if first_user:
                messages = [first_user]
        flatten_multimodal_content(messages, custom_template.image_placeholder)
        data_id = sample.get("id", f"sample_{idx}")
        raw_samples.append((data_id, messages, multimodal_inputs))

    logger.info(
        f"Loaded {len(raw_samples)} samples, {mode_label.lower()} with {num_proc} workers..."
    )

    # Pass 2: process in parallel
    if defer_tokenization:
        work_items = [(messages, max_length) for _, messages, _ in raw_samples]
        worker_init = _init_format_worker
        worker_initargs = (args.target_model_path, True, chat_template_name)
        worker_fn = _format_single
        desc = "Formatting dataset"
    else:
        work_items = [(messages, max_length, train_with_decode) for _, messages, _ in raw_samples]
        last_turn_loss_only = getattr(args, "last_turn_loss_only", False)
        if last_turn_loss_only:
            logger.info(
                "last_turn_loss_only=True: loss mask will only cover the last assistant turn"
            )
        worker_init = _init_tokenize_worker
        worker_initargs = (args.target_model_path, True, chat_template_name, last_turn_loss_only)
        worker_fn = _tokenize_single
        desc = "Tokenizing dataset"

    if num_proc <= 1:
        worker_init(*worker_initargs)
        results = [worker_fn(item) for item in tqdm(work_items, desc=desc)]
    else:
        with mp.Pool(num_proc, initializer=worker_init, initargs=worker_initargs) as pool:
            results = list(
                tqdm(
                    pool.imap(worker_fn, work_items, chunksize=64),
                    total=len(work_items),
                    desc=desc,
                )
            )

    # Collect results
    prompts = []
    skipped = 0
    for (data_id, _, multimodal_inputs), result in zip(raw_samples, results):
        if result is None:
            skipped += 1
            continue
        entry = {
            "data_id": data_id,
            "metadata": {},
            "multimodal_inputs": multimodal_inputs,
            "formatted_prompt": result["formatted_prompt"],
        }

        if not defer_tokenization:
            input_ids = result["input_ids"]
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
            entry["input_ids"] = input_ids
            entry["packed_loss_mask"] = result["packed_loss_mask"]

        prompts.append(entry)

    if skipped:
        logger.warning(f"Skipped {skipped} samples (empty source or zero loss mask)")

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(prompts, cache_path)
    logger.info(f"Saved {len(prompts)} samples to cache: {cache_path}")

    return prompts
