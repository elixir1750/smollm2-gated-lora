#!/usr/bin/env bash
set -euo pipefail

python -m src.beam \
  --checkpoint_dir outputs/smoke \
  --prompt "The capital of France is" \
  --top_k 4 \
  --beam_size 16 \
  --max_phrase_len 4
