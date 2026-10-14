#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; pass the masked-log directory.

python -m gsl_vlm_benchmark.run --masked-log "$1" --backend deepseek \
  --model deepseek-flash --disable-reasoning
