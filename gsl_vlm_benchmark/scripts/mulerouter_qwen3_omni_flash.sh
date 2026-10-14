#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; pass the masked-log directory.

MODEL=qwen3-omni-flash
python -m gsl_vlm_benchmark.run --masked-log "$1" --backend mulerouter \
  --model "$MODEL"
