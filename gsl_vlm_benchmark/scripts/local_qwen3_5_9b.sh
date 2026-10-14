#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root: ./gsl_vlm_benchmark/scripts/local_qwen3_5_9b.sh start|eval|stop
PID_FILE=outputs/gsl_vlm_benchmark/servers/local_qwen3_5_9b.pid

start() {
  mkdir -p outputs/gsl_vlm_benchmark/servers
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then echo "Already running" >&2; exit 1; fi
  CUDA_VISIBLE_DEVICES=4,5 vllm serve Qwen/Qwen3.5-9B --host 127.0.0.1 \
    --tensor-parallel-size 1 --dtype bfloat16 \
    --served-model-name qwen3.5-9b --gpu-memory-utilization 0.9 \
    --max-model-len 8192 --max-num-seqs 1 --port 8706 --default-chat-template-kwargs '{"enable_thinking":false}' &
  pid=$!
  echo "$pid" > "$PID_FILE"
  trap 'kill "$pid" 2>/dev/null || true; rm -f "$PID_FILE"' EXIT
  wait "$pid"
}

eval_model() {
  python -m gsl_vlm_benchmark.run --masked-log "$1" --backend vllm \
    --model qwen3.5-9b --base-url http://127.0.0.1:8706/v1
}

stop() {
  kill "$(cat "$PID_FILE")"
}

case ${1:-} in
  start) start ;;
  eval) eval_model "$2" ;;
  stop) stop ;;
  *) echo "Usage: $0 start|eval MASKS|stop" >&2; exit 2 ;;
esac
