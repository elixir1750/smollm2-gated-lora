#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/colab_10m}"

python -m src.train \
  --model_name_or_path HuggingFaceTB/SmolLM2-135M \
  --dataset_name smollm2 \
  --dataset_config cosmopedia-v2 \
  --max_train_samples 10000 \
  --max_eval_samples 500 \
  --block_size 256 \
  --pad_to_length 256 \
  --max_phrase_len 4 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --learning_rate 2e-4 \
  --warmup_steps 100 \
  --num_train_steps 5000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --dtype float16 \
  --device cuda \
  --log_every 10 \
  --save_every 1000 \
  --output_dir "${OUTPUT_DIR}"

python -m src.observe \
  --checkpoint_dir "${OUTPUT_DIR}" \
  --max_eval_samples 500 \
  --top_k 4 \
  --beam_size 16 \
  --max_phrase_len 4 \
  --num_examples 20 \
  --dtype float16 \
  --device cuda
