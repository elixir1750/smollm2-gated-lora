from __future__ import annotations

import argparse
import types
from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any, Optional, Type, TypeVar, Union, get_args, get_origin, get_type_hints


@dataclass
class TrainConfig:
    model_name_or_path: str = "HuggingFaceTB/SmolLM2-135M"
    dataset_name: str = "smollm2"
    dataset_config: Optional[str] = "cosmopedia-v2"
    train_split: str = "train"
    eval_split: Optional[str] = None
    text_column: Optional[str] = None
    output_dir: str = "outputs/smoke"

    block_size: int = 256
    max_phrase_len: int = 4
    min_prefix_len: int = 8
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    num_train_steps: int = 1000
    warmup_steps: int = 50
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    grad_clip_norm: float = 1.0
    alpha: str = "1.0,0.7,0.5,0.3"
    seed: int = 42
    log_every: int = 10
    save_every: int = 0
    num_workers: int = 0
    device: Optional[str] = None
    dtype: str = "float32"


@dataclass
class EvalConfig:
    checkpoint_dir: str = "outputs/smoke"
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    eval_split: Optional[str] = None
    text_column: Optional[str] = None
    block_size: Optional[int] = None
    max_eval_samples: Optional[int] = 500
    max_phrase_len: Optional[int] = None
    min_prefix_len: Optional[int] = None
    per_device_eval_batch_size: int = 4
    top_k: int = 4
    beam_size: int = 16
    seed: int = 42
    num_workers: int = 0
    device: Optional[str] = None
    dtype: str = "float32"


@dataclass
class BeamConfig:
    checkpoint_dir: str = "outputs/smoke"
    prompt: str = "The capital of France is"
    top_k: int = 4
    beam_size: int = 16
    max_phrase_len: int = 4
    output_file: str = "candidates.json"
    device: Optional[str] = None
    dtype: str = "float32"


@dataclass
class ObserveConfig:
    checkpoint_dir: str = "outputs/smoke"
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    eval_split: Optional[str] = None
    text_column: Optional[str] = None
    block_size: Optional[int] = None
    max_eval_samples: Optional[int] = 200
    min_prefix_len: Optional[int] = None
    per_device_eval_batch_size: int = 4
    top_k: int = 4
    beam_size: int = 16
    max_phrase_len: int = 4
    num_examples: int = 20
    output_file: str = "before_after_observe.json"
    seed: int = 42
    num_workers: int = 0
    device: Optional[str] = None
    dtype: str = "float32"


T = TypeVar("T")


def dataclass_to_dict(config: Any) -> dict[str, Any]:
    return asdict(config)


def _unwrap_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is None:
        return tp
    args = get_args(tp)
    union_origins = {Union}
    if hasattr(types, "UnionType"):
        union_origins.add(types.UnionType)
    if origin in union_origins and type(None) in args:
        return next(a for a in args if a is not type(None))
    return tp


def _coerce_value(value: str, tp: Any) -> Any:
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        origin = get_origin(tp)
        if origin is not None and type(None) in get_args(tp):
            return None
    tp = _unwrap_optional(tp)
    if tp is bool:
        return value.lower() in {"1", "true", "yes", "y"}
    return tp(value)


def parse_dataclass(cls: Type[T], argv: Optional[list[str]] = None) -> T:
    parser = argparse.ArgumentParser()
    hints = get_type_hints(cls)
    for f in fields(cls):
        tp = hints[f.name]
        arg_name = f"--{f.name}"
        default = None if f.default is MISSING else f.default
        if _unwrap_optional(tp) is bool:
            parser.add_argument(arg_name, type=str, default=default)
        else:
            parser.add_argument(arg_name, type=str, default=default)
    ns = parser.parse_args(argv)
    kwargs = {}
    for f in fields(cls):
        tp = hints[f.name]
        value = getattr(ns, f.name)
        if value is None:
            kwargs[f.name] = None
        else:
            kwargs[f.name] = _coerce_value(value, tp)
    return cls(**kwargs)


def parse_alpha(alpha: str, max_phrase_len: int) -> list[float]:
    values = [float(x.strip()) for x in alpha.split(",") if x.strip()]
    if not values:
        values = [1.0]
    while len(values) < max_phrase_len:
        values.append(values[-1])
    return values[:max_phrase_len]


def target_module_names(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]
