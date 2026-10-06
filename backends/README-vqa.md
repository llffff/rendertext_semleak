# Local Qwen VQA environment

Serve Qwen3-VL and Qwen3.5 models in a separate Python 3.11 environment. The `rtsl` environment runs the image backends and scoring client.

For NVIDIA CUDA 12.9 wheels, create the VQA environment from the repository root:

```bash
conda create -n qwen35-vllm python=3.11 pip -y
conda activate qwen35-vllm
python -m pip install -U uv
uv pip install -U --pre vllm "torchcodec==0.16.0+cu129" \
  --torch-backend=cu129 \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
python -m pip check
```

Activate `qwen35-vllm` before running a `vllm serve` command from the [main guide](../README.md#evaluate-an-image-generator) or a [local benchmark script](../gsl_vlm_benchmark/README.md). Run the notebook or evaluation client in `rtsl`.
