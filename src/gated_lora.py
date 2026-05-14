from __future__ import annotations

import contextlib
import contextvars
import math
from typing import Iterator

import torch
from torch import nn


_ACTIVE_LORA_GATE: contextvars.ContextVar[torch.Tensor | None] = contextvars.ContextVar(
    "active_lora_gate", default=None
)


@contextlib.contextmanager
def lora_gate(gate: torch.Tensor | None) -> Iterator[None]:
    token = _ACTIVE_LORA_GATE.set(gate)
    try:
        yield
    finally:
        _ACTIVE_LORA_GATE.reset(token)


def current_lora_gate() -> torch.Tensor | None:
    return _ACTIVE_LORA_GATE.get()


class GatedLoRALinear(nn.Module):
    """Frozen Linear plus token-level gated LoRA branch."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        gate = current_lora_gate()
        if gate is None:
            return base

        lora_input = self.dropout(x).to(dtype=self.lora_A.weight.dtype)
        delta = self.lora_B(self.lora_A(lora_input)) * self.scaling
        delta = delta.to(dtype=base.dtype)
        if x.dim() == 3 and gate.dim() == 3 and gate.shape[:2] == x.shape[:2]:
            return base + gate.to(device=x.device, dtype=base.dtype) * delta
        return base


def replace_linear_with_gated_lora(
    module: nn.Module,
    target_module_names: list[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, GatedLoRALinear):
            continue
        if isinstance(child, nn.Linear) and child_name in target_module_names:
            setattr(module, child_name, GatedLoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced += 1
        else:
            replaced += replace_linear_with_gated_lora(
                child,
                target_module_names=target_module_names,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
    return replaced
