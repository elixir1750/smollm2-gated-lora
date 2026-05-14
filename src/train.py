from __future__ import annotations

import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import TrainConfig, parse_dataclass
from .data import MTPCollator, build_block_datasets
from .model import SmolLM2MTPWrapper
from .utils import (
    append_jsonl,
    count_trainable_parameters,
    get_device,
    linear_warmup_decay_lr,
    move_batch_to_device,
    set_seed,
)


def main() -> None:
    config = parse_dataclass(TrainConfig)
    set_seed(config.seed)
    device = get_device(config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    model = SmolLM2MTPWrapper.from_pretrained(config)
    model.to(device)
    tokenizer = model.tokenizer

    train_dataset, _ = build_block_datasets(config, tokenizer)
    collator = MTPCollator(tokenizer)
    loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
    )
    data_iter = iter(loader)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    log_path = os.path.join(config.output_dir, "train_log.jsonl")
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Train examples: {len(train_dataset):,}")

    model.train()
    pbar = tqdm(range(config.num_train_steps), desc="training")
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = move_batch_to_device(batch, device)
        lr_scale = linear_warmup_decay_lr(step, config.num_train_steps, config.warmup_steps)
        lr = config.learning_rate * lr_scale
        for group in optimizer.param_groups:
            group["lr"] = lr

        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss, logs = model.compute_mtp_loss(
            logits=outputs.logits,
            mtp_positions=batch["mtp_positions"],
            labels_mtp=batch["labels_mtp"],
        )
        loss.backward()
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), config.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        row = {"step": step + 1, "lr": lr, **logs}
        append_jsonl(log_path, row)
        if (step + 1) % config.log_every == 0 or step == 0:
            pbar.set_postfix(loss=f"{logs['loss_mtp']:.4f}", lr=f"{lr:.2e}")

        if config.save_every and (step + 1) % config.save_every == 0:
            model.save_checkpoint(os.path.join(config.output_dir, f"step_{step + 1}"))

    model.save_checkpoint(config.output_dir)
    print(f"Saved checkpoint to {config.output_dir}")


if __name__ == "__main__":
    main()
