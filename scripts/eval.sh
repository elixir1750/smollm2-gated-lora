#!/usr/bin/env bash
set -euo pipefail

python -m src.eval \
  --checkpoint_dir outputs/smoke \
  --max_eval_samples 500 \
  --top_k 4 \
  --beam_size 16
