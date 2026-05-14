from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import TrainConfig, parse_alpha, target_module_names
from .gated_lora import lora_gate, replace_linear_with_gated_lora


class SmolLM2MTPWrapper(nn.Module):
    """Mask-token MTP proposer with token-level gated LoRA inside the frozen backbone."""

    def __init__(self, base_model: nn.Module, tokenizer: Any, config: TrainConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config
        self.max_phrase_len = config.max_phrase_len
        self.mtp_tokens = [f"<mtp_{i}>" for i in range(1, self.max_phrase_len + 1)]
        self.mtp_token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in self.mtp_tokens]

        hidden_size = self.base_model.get_input_embeddings().embedding_dim
        self.mtp_embedding_delta = nn.Parameter(torch.zeros(self.max_phrase_len, hidden_size))

    @classmethod
    def from_pretrained(
        cls,
        config: TrainConfig,
        tokenizer_name_or_path: Optional[str] = None,
    ) -> "SmolLM2MTPWrapper":
        torch_dtype = _dtype_from_string(config.dtype)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path or config.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        mtp_tokens = [f"<mtp_{i}>" for i in range(1, config.max_phrase_len + 1)]
        tokenizer.add_special_tokens({"additional_special_tokens": mtp_tokens})

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch_dtype,
        )
        base_model.resize_token_embeddings(len(tokenizer))
        if hasattr(base_model, "tie_weights"):
            base_model.tie_weights()
        _zero_mtp_token_rows(base_model, [tokenizer.convert_tokens_to_ids(tok) for tok in mtp_tokens])

        for param in base_model.parameters():
            param.requires_grad = False

        replaced = replace_linear_with_gated_lora(
            base_model,
            target_module_names=target_module_names(config.lora_target_modules),
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        if replaced == 0:
            raise RuntimeError(
                "No Linear layers were replaced. Check lora_target_modules for this architecture."
            )
        return cls(base_model=base_model, tokenizer=tokenizer, config=config)

    def build_lora_gate(self, input_ids: torch.Tensor) -> torch.Tensor:
        gate = torch.zeros((*input_ids.shape, 1), device=input_ids.device, dtype=torch.float32)
        for token_id in self.mtp_token_ids:
            gate = gate + (input_ids == token_id).unsqueeze(-1).float()
        return gate.clamp_(0.0, 1.0)

    def input_embeddings_with_mtp_delta(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.base_model.get_input_embeddings()(input_ids)
        delta = torch.zeros_like(embeds)
        for idx, token_id in enumerate(self.mtp_token_ids):
            mask = (input_ids == token_id).unsqueeze(-1).to(dtype=embeds.dtype)
            delta = delta + mask * self.mtp_embedding_delta[idx].to(dtype=embeds.dtype)
        return embeds + delta

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        gate = self.build_lora_gate(input_ids)
        inputs_embeds = self.input_embeddings_with_mtp_delta(input_ids)
        with lora_gate(gate):
            return self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )

    def compute_mtp_loss(
        self,
        logits: torch.Tensor,
        mtp_positions: torch.Tensor,
        labels_mtp: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        alphas = parse_alpha(self.config.alpha, self.max_phrase_len)
        total = logits.new_tensor(0.0)
        logs: dict[str, float] = {}
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        for k in range(self.max_phrase_len):
            step_logits = logits[batch_idx, mtp_positions[:, k]]
            step_labels = labels_mtp[:, k]
            loss = nn.functional.cross_entropy(step_logits, step_labels)
            total = total + float(alphas[k]) * loss
            logs[f"loss_mtp_step_{k + 1}"] = float(loss.detach().cpu())
        logs["loss_mtp"] = float(total.detach().cpu())
        return total, logs

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {name for name, p in self.named_parameters() if p.requires_grad}
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }

    def save_checkpoint(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.trainable_state_dict(), os.path.join(output_dir, "trainable.pt"))
        self.tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, sort_keys=True)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: str,
        device: str | torch.device | None = None,
        dtype: str | None = None,
    ) -> "SmolLM2MTPWrapper":
        with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        if dtype is not None:
            cfg_dict["dtype"] = dtype
        config = TrainConfig(**cfg_dict)
        model = cls.from_pretrained(config, tokenizer_name_or_path=checkpoint_dir)
        state = torch.load(os.path.join(checkpoint_dir, "trainable.pt"), map_location="cpu")
        if state and any(key.startswith("mtp_heads.") for key in state):
            raise RuntimeError(
                "This checkpoint was created by the current-hidden residual-head prototype. "
                "Please rerun training for the gated-LoRA mask-token MTP model."
            )
        missing, unexpected = model.load_state_dict(state, strict=False)
        unexpected = [key for key in unexpected if key in state]
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
        if device is not None:
            model.to(device)
        model.eval()
        return model


def _zero_mtp_token_rows(base_model: nn.Module, token_ids: list[int]) -> None:
    with torch.no_grad():
        input_embeddings = base_model.get_input_embeddings()
        for token_id in token_ids:
            input_embeddings.weight[token_id].zero_()

        output_embeddings = base_model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings.weight is not input_embeddings.weight:
            for token_id in token_ids:
                output_embeddings.weight[token_id].zero_()


def _dtype_from_string(value: str) -> torch.dtype | None:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "auto": None,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {value}")
    return mapping[value]
