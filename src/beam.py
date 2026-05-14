from __future__ import annotations

import os
from typing import Any

import torch

from .config import BeamConfig, parse_dataclass
from .model import SmolLM2MTPWrapper
from .utils import get_device, write_json


def beam_search_from_logits(
    logits_by_step: torch.Tensor,
    tokenizer: Any,
    top_k: int = 4,
    beam_size: int = 16,
    max_phrase_len: int = 4,
) -> dict[int, list[dict[str, Any]]]:
    """Build phrase beams from per-mask-position logits."""

    log_probs = torch.log_softmax(logits_by_step[:max_phrase_len], dim=-1)
    beams: list[tuple[list[int], float]] = [([], 0.0)]
    by_len: dict[int, list[dict[str, Any]]] = {}

    for depth in range(max_phrase_len):
        step_top = torch.topk(log_probs[depth], k=min(top_k, log_probs.size(-1)))
        expansions: list[tuple[list[int], float]] = []
        for prefix, score in beams:
            for token_id, token_score in zip(step_top.indices.tolist(), step_top.values.tolist()):
                expansions.append((prefix + [int(token_id)], float(score + token_score)))
        expansions.sort(key=lambda x: x[1], reverse=True)
        beams = expansions[:beam_size]
        by_len[depth + 1] = [
            {
                "rank": rank,
                "token_ids": token_ids,
                "score": score,
                "text": tokenizer.decode(token_ids, skip_special_tokens=True),
            }
            for rank, (token_ids, score) in enumerate(beams, start=1)
        ]
    return by_len


@torch.no_grad()
def propose_phrases(
    model: SmolLM2MTPWrapper,
    prompt: str,
    top_k: int,
    beam_size: int,
    max_phrase_len: int,
    device: torch.device,
) -> dict[int, list[dict[str, Any]]]:
    tokenizer = model.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prefix_ids = encoded["input_ids"].to(device)
    max_phrase_len = min(max_phrase_len, model.max_phrase_len)
    mtp_ids = torch.tensor(model.mtp_token_ids[:max_phrase_len], device=device).unsqueeze(0)
    input_ids = torch.cat([prefix_ids, mtp_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    positions = torch.arange(prefix_ids.size(1), prefix_ids.size(1) + max_phrase_len, device=device)
    logits_by_step = outputs.logits[0, positions]
    return beam_search_from_logits(
        logits_by_step=logits_by_step,
        tokenizer=tokenizer,
        top_k=top_k,
        beam_size=beam_size,
        max_phrase_len=max_phrase_len,
    )


def main() -> None:
    config = parse_dataclass(BeamConfig)
    device = get_device(config.device)
    model = SmolLM2MTPWrapper.load_checkpoint(
        config.checkpoint_dir,
        device=device,
        dtype=config.dtype,
    )
    candidates = propose_phrases(
        model=model,
        prompt=config.prompt,
        top_k=config.top_k,
        beam_size=config.beam_size,
        max_phrase_len=config.max_phrase_len,
        device=device,
    )
    output_path = config.output_file
    if not os.path.isabs(output_path):
        output_path = os.path.join(config.checkpoint_dir, output_path)
    payload = {
        "prompt": config.prompt,
        "top_k": config.top_k,
        "beam_size": config.beam_size,
        "max_phrase_len": min(config.max_phrase_len, model.max_phrase_len),
        "candidates_by_len": candidates,
    }
    write_json(output_path, payload)
    for cand in candidates[min(config.max_phrase_len, model.max_phrase_len)][: min(10, config.beam_size)]:
        print(f"{cand['rank']:>3} score={cand['score']:.3f} text={cand['text']!r}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
