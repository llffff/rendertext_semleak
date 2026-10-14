#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; pass the masked-log directory.

# Step 1: Select the OpenRouter VLM.
MODEL=qwen/qwen3-vl-32b-instruct

# Step 2: Score the masked images.
python -m gsl_vlm_benchmark.run --masked-log "$1" --backend openrouter \
  --model "$MODEL"
