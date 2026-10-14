#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; pass the masked-log directory.

MODEL=xiaomi/mimo-v2.6-flash
python -m gsl_vlm_benchmark.run --masked-log "$1" --backend openrouter \
  --model "$MODEL" --disable-reasoning
