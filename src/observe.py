from __future__ import annotations

import json
import os
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .beam import beam_search_from_logits
from .config import ObserveConfig, TrainConfig, parse_dataclass
from .data import MTPCollator, build_block_datasets, config_with_eval_overrides
from .metrics import AnyLengthPrefixRecallMeter, PhraseRankMeter, TokenAccMeter, gather_mtp_logits
from .model import SmolLM2MTPWrapper
from .utils import get_device, move_batch_to_device, set_seed, write_json


def find_candidate_rank(candidates: list[dict[str, Any]], gold_tokens: list[int]) -> int | None:
    gold = tuple(gold_tokens)
    for cand in candidates:
        if tuple(cand["token_ids"]) == gold:
            return int(cand["rank"])
    return None


def prefix_hits_by_len(
    candidates_by_len: dict[int, list[dict[str, Any]]],
    gold_tokens: list[int],
    max_phrase_len: int,
) -> dict[str, bool]:
    all_candidates = [
        cand
        for candidates in candidates_by_len.values()
        for cand in candidates
    ]
    hits = {}
    for prefix_len in range(1, max_phrase_len + 1):
        gold_prefix = tuple(gold_tokens[:prefix_len])
        hits[f"len_{prefix_len}"] = any(
            len(cand["token_ids"]) >= prefix_len
            and tuple(cand["token_ids"][:prefix_len]) == gold_prefix
            for cand in all_candidates
        )
    return hits


@torch.no_grad()
def evaluate_observation_model(
    model: SmolLM2MTPWrapper,
    loader: DataLoader,
    device: torch.device,
    max_phrase_len: int,
    top_k: int,
    beam_size: int,
    num_examples: int,
    label: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    token_meter = TokenAccMeter(max_phrase_len=max_phrase_len)
    phrase_meter = PhraseRankMeter(max_phrase_len=max_phrase_len)
    any_len_prefix_meter = AnyLengthPrefixRecallMeter(max_phrase_len=max_phrase_len)
    examples: list[dict[str, Any]] = []
    sample_idx = 0

    model.eval()
    for batch in tqdm(loader, desc=f"observing {label}"):
        batch = move_batch_to_device(batch, device)
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits_by_step = gather_mtp_logits(outputs.logits, batch["mtp_positions"])[:, :max_phrase_len]
        labels = batch["labels_mtp"][:, :max_phrase_len]
        token_meter.update(logits_by_step, labels)

        for i in range(logits_by_step.size(0)):
            candidates = beam_search_from_logits(
                logits_by_step=logits_by_step[i],
                tokenizer=model.tokenizer,
                top_k=top_k,
                beam_size=beam_size,
                max_phrase_len=max_phrase_len,
            )
            gold = labels[i].detach().cpu().tolist()
            phrase_meter.update(candidates, gold)
            any_len_prefix_meter.update(candidates, gold)

            phrase4_candidates = candidates[max_phrase_len]
            phrase4_rank = find_candidate_rank(phrase4_candidates, gold[:max_phrase_len])
            if len(examples) < num_examples:
                prefix_len = int(batch["prefix_len"][i].item())
                prefix_ids = batch["input_ids"][i, :prefix_len].detach().cpu().tolist()
                examples.append(
                    {
                        "sample_idx": sample_idx,
                        "prefix_text": model.tokenizer.decode(prefix_ids, skip_special_tokens=True),
                        "gold_token_ids": gold,
                        "gold_text": model.tokenizer.decode(gold, skip_special_tokens=True),
                        "phrase4_rank": phrase4_rank,
                        "phrase4_hit_in_beam": phrase4_rank is not None,
                        "any_len_prefix_hits": prefix_hits_by_len(candidates, gold, max_phrase_len),
                        "top_phrase4": phrase4_candidates[: min(10, len(phrase4_candidates))],
                    }
                )
            sample_idx += 1

    metrics = {
        **token_meter.compute(),
        **phrase_meter.compute(),
        **any_len_prefix_meter.compute(),
    }
    return metrics, examples


def diff_metrics(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(before) | set(after))
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in keys}


def merge_examples(
    before_examples: list[dict[str, Any]],
    after_examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = []
    after_by_idx = {ex["sample_idx"]: ex for ex in after_examples}
    for before in before_examples:
        after = after_by_idx.get(before["sample_idx"], {})
        merged.append(
            {
                "sample_idx": before["sample_idx"],
                "prefix_text": before["prefix_text"],
                "gold_token_ids": before["gold_token_ids"],
                "gold_text": before["gold_text"],
                "before_phrase4_rank": before["phrase4_rank"],
                "before_phrase4_hit_in_beam": before["phrase4_hit_in_beam"],
                "before_any_len_prefix_hits": before["any_len_prefix_hits"],
                "before_top_phrase4": before["top_phrase4"],
                "after_phrase4_rank": after.get("phrase4_rank"),
                "after_phrase4_hit_in_beam": after.get("phrase4_hit_in_beam"),
                "after_any_len_prefix_hits": after.get("any_len_prefix_hits", {}),
                "after_top_phrase4": after.get("top_phrase4", []),
            }
        )
    return merged


@torch.no_grad()
def main() -> None:
    observe_config = parse_dataclass(ObserveConfig)
    set_seed(observe_config.seed)
    device = get_device(observe_config.device)

    config_path = os.path.join(observe_config.checkpoint_dir, "config.json")
    trainable_path = os.path.join(observe_config.checkpoint_dir, "trainable.pt")
    if not os.path.exists(config_path) or not os.path.exists(trainable_path):
        raise SystemExit(
            "Missing checkpoint files for before/after observation.\n"
            f"Expected:\n  {config_path}\n  {trainable_path}\n\n"
            "Run training first, for example:\n"
            "  python -m src.train --output_dir outputs/smoke\n\n"
            "Then rerun:\n"
            "  python -m src.observe --checkpoint_dir outputs/smoke"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg_dict["dtype"] = observe_config.dtype
    train_config = TrainConfig(**cfg_dict)
    max_phrase_len = min(observe_config.max_phrase_len, train_config.max_phrase_len)
    if max_phrase_len != 4:
        print(f"Observing phrase length {max_phrase_len}; use --max_phrase_len 4 for 4-token phrase checks.")

    overrides = {
        "dataset_name": observe_config.dataset_name,
        "dataset_config": observe_config.dataset_config,
        "eval_split": observe_config.eval_split,
        "text_column": observe_config.text_column,
        "block_size": observe_config.block_size,
        "max_eval_samples": observe_config.max_eval_samples,
        "max_phrase_len": max_phrase_len,
        "min_prefix_len": observe_config.min_prefix_len,
        "per_device_eval_batch_size": observe_config.per_device_eval_batch_size,
        "num_workers": observe_config.num_workers,
        "seed": observe_config.seed,
    }
    data_config = config_with_eval_overrides(train_config, overrides)

    baseline_model = SmolLM2MTPWrapper.from_pretrained(
        train_config,
        tokenizer_name_or_path=observe_config.checkpoint_dir,
    ).to(device)
    _, eval_dataset = build_block_datasets(data_config, baseline_model.tokenizer)
    loader = DataLoader(
        eval_dataset,
        batch_size=observe_config.per_device_eval_batch_size,
        shuffle=False,
        num_workers=observe_config.num_workers,
        collate_fn=MTPCollator(baseline_model.tokenizer, pad_to_length=data_config.block_size),
    )
    before_metrics, before_examples = evaluate_observation_model(
        model=baseline_model,
        loader=loader,
        device=device,
        max_phrase_len=max_phrase_len,
        top_k=observe_config.top_k,
        beam_size=observe_config.beam_size,
        num_examples=observe_config.num_examples,
        label="before training",
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    trained_model = SmolLM2MTPWrapper.load_checkpoint(
        observe_config.checkpoint_dir,
        device=device,
        dtype=observe_config.dtype,
    )
    after_metrics, after_examples = evaluate_observation_model(
        model=trained_model,
        loader=loader,
        device=device,
        max_phrase_len=max_phrase_len,
        top_k=observe_config.top_k,
        beam_size=observe_config.beam_size,
        num_examples=observe_config.num_examples,
        label="after training",
    )

    payload = {
        "checkpoint_dir": observe_config.checkpoint_dir,
        "num_eval_samples": len(eval_dataset),
        "top_k": observe_config.top_k,
        "beam_size": observe_config.beam_size,
        "max_phrase_len": max_phrase_len,
        "before": before_metrics,
        "after": after_metrics,
        "delta_after_minus_before": diff_metrics(before_metrics, after_metrics),
        "examples": merge_examples(before_examples, after_examples),
    }

    output_path = observe_config.output_file
    if not os.path.isabs(output_path):
        output_path = os.path.join(observe_config.checkpoint_dir, output_path)
    write_json(output_path, payload)

    print("Before:", before_metrics)
    print("After:", after_metrics)
    print("Delta:", payload["delta_after_minus_before"])
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
