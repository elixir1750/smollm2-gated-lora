#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/colab_medium_41m}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

python -m src.train \
  --model_name_or_path HuggingFaceTB/SmolLM2-135M \
  --dataset_name smollm2 \
  --dataset_config cosmopedia-v2 \
  --max_train_samples 20000 \
  --max_eval_samples 1000 \
  --block_size 512 \
  --pad_to_length 512 \
  --max_phrase_len 4 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --learning_rate 2e-4 \
  --warmup_steps 200 \
  --num_train_steps 10000 \
  --per_device_train_batch_size "${MICRO_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --dtype float16 \
  --device cuda \
  --log_every 10 \
  --save_every 2000 \
  --output_dir "${OUTPUT_DIR}"

python -m src.observe \
  --checkpoint_dir "${OUTPUT_DIR}" \
  --max_eval_samples 1000 \
  --top_k 4 \
  --beam_size 16 \
  --max_phrase_len 4 \
  --num_examples 30 \
  --dtype float16 \
  --device cuda
