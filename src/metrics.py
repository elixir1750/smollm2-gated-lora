from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Iterable

import torch


class TokenAccMeter:
    def __init__(self, max_phrase_len: int, ks: Iterable[int] = (1, 4, 5, 50)) -> None:
        self.max_phrase_len = max_phrase_len
        self.ks = list(ks)
        self.correct = defaultdict(int)
        self.total = defaultdict(int)

    def update(self, logits_by_step: torch.Tensor, labels_mtp: torch.Tensor) -> None:
        # logits_by_step: [batch, max_phrase_len, vocab]
        for step in range(self.max_phrase_len):
            labels = labels_mtp[:, step]
            vocab = logits_by_step.size(-1)
            for k in self.ks:
                kk = min(k, vocab)
                top = logits_by_step[:, step].topk(kk, dim=-1).indices
                hit = (top == labels.unsqueeze(-1)).any(dim=-1)
                key = (step + 1, k)
                self.correct[key] += int(hit.sum().item())
                self.total[key] += int(labels.numel())

    def compute(self) -> dict[str, float]:
        out = {}
        for step in range(1, self.max_phrase_len + 1):
            for k in self.ks:
                key = (step, k)
                denom = max(self.total[key], 1)
                out[f"mtp_step_{step}_acc@{k}"] = self.correct[key] / denom
        return out


class PhraseRankMeter:
    def __init__(
        self,
        max_phrase_len: int,
        recall_ks: Iterable[int] = (10, 16, 50, 100),
    ) -> None:
        self.max_phrase_len = max_phrase_len
        self.recall_ks = list(recall_ks)
        self.hits = defaultdict(int)
        self.total = defaultdict(int)
        self.ranks: list[int] = []
        self.rrs: list[float] = []

    def update(self, candidates_by_len: dict[int, list[dict]], gold_tokens: list[int]) -> None:
        for phrase_len in range(1, self.max_phrase_len + 1):
            gold = tuple(gold_tokens[:phrase_len])
            candidates = candidates_by_len.get(phrase_len, [])
            rank = None
            for i, cand in enumerate(candidates, start=1):
                if tuple(cand["token_ids"]) == gold:
                    rank = i
                    break
            if rank is None:
                rank = len(candidates) + 1

            self.ranks.append(rank)
            self.rrs.append(1.0 / rank if rank <= len(candidates) else 0.0)
            for k in self.recall_ks:
                key = (phrase_len, k)
                self.hits[key] += int(rank <= k and rank <= len(candidates))
                self.total[key] += 1

    def compute(self) -> dict[str, float]:
        out = {}
        for phrase_len in range(1, self.max_phrase_len + 1):
            for k in self.recall_ks:
                key = (phrase_len, k)
                out[f"phrase_len_{phrase_len}_recall@{k}"] = self.hits[key] / max(self.total[key], 1)
        out["mean_gold_rank"] = float(mean(self.ranks)) if self.ranks else 0.0
        out["median_gold_rank"] = float(median(self.ranks)) if self.ranks else 0.0
        out["MRR"] = float(mean(self.rrs)) if self.rrs else 0.0
        return out


class AnyLengthPrefixRecallMeter:
    """Recall gold prefixes against every beam candidate length.

    For k=1..max_phrase_len, a hit means any candidate of length >= k has the
    same first k token ids as the gold continuation. This captures the practical
    question: did the beam contain a candidate whose prefix is a correct
    1/2/3/4-token phrase, regardless of the candidate's final length?
    """

    def __init__(self, max_phrase_len: int) -> None:
        self.max_phrase_len = max_phrase_len
        self.hits = defaultdict(int)
        self.total = defaultdict(int)

    def update(self, candidates_by_len: dict[int, list[dict]], gold_tokens: list[int]) -> None:
        all_candidates = [
            cand
            for phrase_len, candidates in candidates_by_len.items()
            for cand in candidates
            if phrase_len <= self.max_phrase_len
        ]
        for prefix_len in range(1, self.max_phrase_len + 1):
            gold_prefix = tuple(gold_tokens[:prefix_len])
            hit = any(
                len(cand["token_ids"]) >= prefix_len
                and tuple(cand["token_ids"][:prefix_len]) == gold_prefix
                for cand in all_candidates
            )
            self.hits[prefix_len] += int(hit)
            self.total[prefix_len] += 1

    def compute(self) -> dict[str, float]:
        return {
            f"any_len_prefix_{prefix_len}_recall@beam": self.hits[prefix_len] / max(self.total[prefix_len], 1)
            for prefix_len in range(1, self.max_phrase_len + 1)
        }


def gather_mtp_logits(logits: torch.Tensor, mtp_positions: torch.Tensor) -> torch.Tensor:
    batch_idx = torch.arange(logits.size(0), device=logits.device).unsqueeze(-1)
    return logits[batch_idx, mtp_positions]
