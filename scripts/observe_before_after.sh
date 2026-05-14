#!/usr/bin/env bash
set -euo pipefail

python -m src.observe \
  --checkpoint_dir outputs/smoke \
  --max_eval_samples 200 \
  --top_k 4 \
  --beam_size 16 \
  --max_phrase_len 4 \
  --num_examples 20
