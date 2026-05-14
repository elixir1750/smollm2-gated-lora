#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
  --model_name_or_path HuggingFaceTB/SmolLM2-135M \
  --dataset_name smollm2 \
  --dataset_config cosmopedia-v2 \
  --max_train_samples 2000 \
  --max_eval_samples 200 \
  --block_size 256 \
  --max_phrase_len 4 \
  --lora_rank 8 \
  --learning_rate 2e-4 \
  --num_train_steps 1000 \
  --per_device_train_batch_size 4 \
  --output_dir outputs/smoke
