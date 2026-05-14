from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset

from .config import TrainConfig


def normalize_dataset_name(dataset_name: str) -> str:
    aliases = {
        "smollm2": "HuggingFaceTB/smollm-corpus",
        "smollm": "HuggingFaceTB/smollm-corpus",
        "smollm-corpus": "HuggingFaceTB/smollm-corpus",
        "tinystories": "roneneldan/TinyStories",
        "tiny_stories": "roneneldan/TinyStories",
        "wikitext-2": "wikitext",
    }
    return aliases.get(dataset_name.lower(), dataset_name)


def load_raw_splits(config: TrainConfig) -> tuple[Dataset, Dataset]:
    dataset_name = normalize_dataset_name(config.dataset_name)
    dataset_config = config.dataset_config
    if dataset_name == "roneneldan/TinyStories" and dataset_config in {"wikitext-2-raw-v1", "cosmopedia-v2"}:
        dataset_config = None

    raw = load_dataset(dataset_name, dataset_config)
    if not isinstance(raw, DatasetDict):
        raise ValueError("Expected load_dataset to return a DatasetDict.")

    train_split = config.train_split
    eval_split = config.eval_split
    if eval_split is None:
        if "validation" in raw:
            eval_split = "validation"
        elif "test" in raw:
            eval_split = "test"
        else:
            eval_split = train_split

    return raw[train_split], raw[eval_split]


def infer_text_column(dataset: Dataset, explicit: Optional[str] = None) -> str:
    if explicit is not None:
        return explicit
    for candidate in ["text", "content", "story"]:
        if candidate in dataset.column_names:
            return candidate
    for name in dataset.column_names:
        if isinstance(dataset[0][name], str):
            return name
    raise ValueError(f"Could not infer text column from columns: {dataset.column_names}")


def infer_text_column_from_row(row: dict[str, Any], explicit: Optional[str] = None) -> str:
    if explicit is not None:
        return explicit
    for candidate in ["text", "content", "story"]:
        if candidate in row and isinstance(row[candidate], str):
            return candidate
    for name, value in row.items():
        if isinstance(value, str):
            return name
    raise ValueError(f"Could not infer text column from row keys: {list(row)}")


def tokenize_and_group(
    dataset: Dataset,
    tokenizer: Any,
    block_size: int,
    max_samples: Optional[int],
    text_column: Optional[str] = None,
) -> Dataset:
    text_column = infer_text_column(dataset, text_column)
    eos_id = tokenizer.eos_token_id

    def tokenize_batch(examples: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        texts = [x if isinstance(x, str) else "" for x in examples[text_column]]
        encoded = tokenizer(texts, add_special_tokens=False)
        input_ids = []
        for ids in encoded["input_ids"]:
            if eos_id is not None:
                ids = ids + [eos_id]
            input_ids.append(ids)
        return {"input_ids": input_ids}

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    def group_batch(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated: list[int] = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_length = (len(concatenated) // block_size) * block_size
        blocks = [
            concatenated[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        return {"input_ids": blocks}

    grouped = tokenized.map(group_batch, batched=True, desc="Grouping")
    if max_samples is not None:
        grouped = grouped.select(range(min(max_samples, len(grouped))))
    return grouped


def stream_tokenize_and_group(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    tokenizer: Any,
    block_size: int,
    max_blocks: Optional[int],
    skip_blocks: int,
    text_column: Optional[str] = None,
) -> Dataset:
    target_blocks = max_blocks if max_blocks is not None else 10_000
    stream = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    buffer: list[int] = []
    blocks: list[list[int]] = []
    skipped = 0
    inferred_text_column = text_column
    eos_id = tokenizer.eos_token_id

    for row in stream:
        if inferred_text_column is None:
            inferred_text_column = infer_text_column_from_row(row, text_column)
        text = row.get(inferred_text_column, "")
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if eos_id is not None:
            ids = ids + [eos_id]
        buffer.extend(ids)

        while len(buffer) >= block_size:
            block = buffer[:block_size]
            del buffer[:block_size]
            if skipped < skip_blocks:
                skipped += 1
                continue
            blocks.append(block)
            if len(blocks) >= target_blocks:
                return Dataset.from_dict({"input_ids": blocks})

    if not blocks:
        raise ValueError("Streaming dataset ended before producing any token blocks.")
    return Dataset.from_dict({"input_ids": blocks})


def build_block_datasets(config: TrainConfig, tokenizer: Any) -> tuple["MTPBlockDataset", "MTPBlockDataset"]:
    dataset_name = normalize_dataset_name(config.dataset_name)
    dataset_config = config.dataset_config
    if dataset_name == "HuggingFaceTB/smollm-corpus" and config.eval_split is None:
        train_blocks = stream_tokenize_and_group(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=config.train_split,
            tokenizer=tokenizer,
            block_size=config.block_size,
            max_blocks=config.max_train_samples,
            skip_blocks=0,
            text_column=config.text_column,
        )
        eval_blocks = stream_tokenize_and_group(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=config.train_split,
            tokenizer=tokenizer,
            block_size=config.block_size,
            max_blocks=config.max_eval_samples,
            skip_blocks=len(train_blocks),
            text_column=config.text_column,
        )
        return (
            MTPBlockDataset(train_blocks, tokenizer, config, training=True),
            MTPBlockDataset(eval_blocks, tokenizer, config, training=False),
        )

    train_raw, eval_raw = load_raw_splits(config)
    train_blocks = tokenize_and_group(
        train_raw,
        tokenizer=tokenizer,
        block_size=config.block_size,
        max_samples=config.max_train_samples,
        text_column=config.text_column,
    )
    eval_blocks = tokenize_and_group(
        eval_raw,
        tokenizer=tokenizer,
        block_size=config.block_size,
        max_samples=config.max_eval_samples,
        text_column=config.text_column,
    )
    return (
        MTPBlockDataset(train_blocks, tokenizer, config, training=True),
        MTPBlockDataset(eval_blocks, tokenizer, config, training=False),
    )


def config_with_eval_overrides(train_config: TrainConfig, overrides: dict[str, Any]) -> TrainConfig:
    updates = {k: v for k, v in overrides.items() if v is not None and hasattr(train_config, k)}
    return replace(train_config, **updates)


class MTPBlockDataset(TorchDataset):
    def __init__(self, blocks: Dataset, tokenizer: Any, config: TrainConfig, training: bool) -> None:
        self.blocks = blocks
        self.tokenizer = tokenizer
        self.max_phrase_len = config.max_phrase_len
        self.min_prefix_len = config.min_prefix_len
        self.max_prefix_len = config.max_prefix_len
        self.training = training
        self.seed = config.seed
        self.mtp_token_ids = [
            tokenizer.convert_tokens_to_ids(f"<mtp_{i}>")
            for i in range(1, self.max_phrase_len + 1)
        ]
        self.valid_indices = [
            i
            for i, row in enumerate(blocks)
            if len(row["input_ids"]) >= self.min_prefix_len + self.max_phrase_len
        ]
        if not self.valid_indices:
            raise ValueError("No token blocks are long enough for the requested max_phrase_len.")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        block = self.blocks[self.valid_indices[idx]]["input_ids"]
        max_prefix = len(block) - self.max_phrase_len
        if self.max_prefix_len is not None:
            max_prefix = min(max_prefix, self.max_prefix_len)
        if max_prefix < self.min_prefix_len:
            raise ValueError("max_prefix_len must be >= min_prefix_len.")
        if self.training:
            prefix_len = random.randint(self.min_prefix_len, max_prefix)
        else:
            rng = random.Random(self.seed + idx)
            prefix_len = rng.randint(self.min_prefix_len, max_prefix)

        input_ids = block[:prefix_len] + self.mtp_token_ids
        labels_mtp = block[prefix_len : prefix_len + self.max_phrase_len]
        mtp_positions = list(range(prefix_len, prefix_len + self.max_phrase_len))
        return {
            "input_ids": input_ids,
            "labels_mtp": labels_mtp,
            "mtp_positions": mtp_positions,
            "prefix_len": prefix_len,
        }


class MTPCollator:
    def __init__(self, tokenizer: Any, pad_to_length: Optional[int] = None) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_length = pad_to_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(ex["input_ids"]) for ex in examples)
        if self.pad_to_length is not None:
            max_len = max(max_len, self.pad_to_length)
        input_ids = []
        attention_mask = []
        for ex in examples:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels_mtp": torch.tensor([ex["labels_mtp"] for ex in examples], dtype=torch.long),
            "mtp_positions": torch.tensor([ex["mtp_positions"] for ex in examples], dtype=torch.long),
            "prefix_len": torch.tensor([ex["prefix_len"] for ex in examples], dtype=torch.long),
        }
