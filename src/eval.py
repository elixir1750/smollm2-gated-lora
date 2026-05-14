from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .beam import beam_search_from_logits
from .config import EvalConfig, parse_dataclass
from .data import MTPCollator, build_block_datasets, config_with_eval_overrides
from .metrics import AnyLengthPrefixRecallMeter, PhraseRankMeter, TokenAccMeter, gather_mtp_logits
from .model import SmolLM2MTPWrapper
from .utils import get_device, move_batch_to_device, set_seed, write_json


@torch.no_grad()
def main() -> None:
    eval_config = parse_dataclass(EvalConfig)
    set_seed(eval_config.seed)
    device = get_device(eval_config.device)

    model = SmolLM2MTPWrapper.load_checkpoint(
        eval_config.checkpoint_dir,
        device=device,
        dtype=eval_config.dtype,
    )
    train_config = model.config
    max_phrase_len = eval_config.max_phrase_len or train_config.max_phrase_len
    max_phrase_len = min(max_phrase_len, model.max_phrase_len)
    overrides = {
        "dataset_name": eval_config.dataset_name,
        "dataset_config": eval_config.dataset_config,
        "eval_split": eval_config.eval_split,
        "text_column": eval_config.text_column,
        "block_size": eval_config.block_size,
        "max_eval_samples": eval_config.max_eval_samples,
        "max_phrase_len": max_phrase_len,
        "min_prefix_len": eval_config.min_prefix_len,
        "per_device_eval_batch_size": eval_config.per_device_eval_batch_size,
        "num_workers": eval_config.num_workers,
        "seed": eval_config.seed,
    }
    data_config = config_with_eval_overrides(train_config, overrides)
    _, eval_dataset = build_block_datasets(data_config, model.tokenizer)
    loader = DataLoader(
        eval_dataset,
        batch_size=eval_config.per_device_eval_batch_size,
        shuffle=False,
        num_workers=eval_config.num_workers,
        collate_fn=MTPCollator(model.tokenizer),
    )

    token_meter = TokenAccMeter(max_phrase_len=max_phrase_len)
    phrase_meter = PhraseRankMeter(max_phrase_len=max_phrase_len)
    any_len_prefix_meter = AnyLengthPrefixRecallMeter(max_phrase_len=max_phrase_len)

    model.eval()
    for batch in tqdm(loader, desc="evaluating"):
        batch = move_batch_to_device(batch, device)
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits_by_step = gather_mtp_logits(outputs.logits, batch["mtp_positions"])
        token_meter.update(logits_by_step[:, :max_phrase_len], batch["labels_mtp"][:, :max_phrase_len])

        for i in range(logits_by_step.size(0)):
            candidates = beam_search_from_logits(
                logits_by_step=logits_by_step[i, :max_phrase_len],
                tokenizer=model.tokenizer,
                top_k=eval_config.top_k,
                beam_size=eval_config.beam_size,
                max_phrase_len=max_phrase_len,
            )
            gold = batch["labels_mtp"][i, :max_phrase_len].detach().cpu().tolist()
            phrase_meter.update(candidates, gold)
            any_len_prefix_meter.update(candidates, gold)

    metrics = {
        **token_meter.compute(),
        **phrase_meter.compute(),
        **any_len_prefix_meter.compute(),
        "num_eval_samples": len(eval_dataset),
        "top_k": eval_config.top_k,
        "beam_size": eval_config.beam_size,
        "max_phrase_len": max_phrase_len,
    }
    output_path = os.path.join(eval_config.checkpoint_dir, "metrics.json")
    write_json(output_path, metrics)
    print(metrics)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
