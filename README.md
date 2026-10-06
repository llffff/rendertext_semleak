# When Scene Text Hijacks the Scene

![IEEE S&P 2027](https://img.shields.io/badge/IEEE%20S%26P-2027-8A2BE2?style=flat-square)
[![TSL-60 Dataset](https://img.shields.io/badge/TSL--60-Hugging%20Face-yellow?style=flat-square)](https://huggingface.co/datasets/feifeili00/rendertext_semleak)
[![PyPI](https://img.shields.io/badge/PyPI-rtsl--scorer-blue?style=flat-square)](https://pypi.org/project/rtsl-scorer/)

**[Quickstart](quickstart.ipynb) · [Dataset](https://huggingface.co/datasets/feifeili00/rendertext_semleak) · [Scorer](rtsl-scorer/README.md)**

Code for measuring rendered-text semantic leakage (TSL) in generated images.

![Rendered-text semantic leakage example](assets/teaser.png)

## What the scores measure

**TSL** occurs when text requested *inside* an image also changes its non-text visual content. For example, a request to print “a dog” may cause a dog to appear in the scene. The evaluator uses OCR to mask visible writing before measuring leakage, so reading the words in the image is not enough to count as visual evidence.

- **Word recall (WR)** measures how much of the requested text appears in the generated image. It measures text-rendering fidelity, separately from leakage.
- **Direct TSL (DTSL)** is the CLIP similarity between the requested rendered text and the target image after masking its visible text.
- **Normalized TSL (NSL)** subtracts a matched baseline from DTSL. For each of the 72 main visual prompts, `TSL60_base_prompts.csv` provides two alternative prompts that keep the main scene and rendering instructions while replacing the target text. Generating them produces two **matched base images**. After masking each base image's own text, the evaluator compares both bases with the *target* rendered text:

  $$
  \mathrm{NSL}=\mathrm{CLIP}(I_{\mathrm{target}}^{\mathrm{mask}},t)-\frac{1}{2}\sum_{i=1}^{2}\mathrm{CLIP}(I_{\mathrm{base},i}^{\mathrm{mask}},t).
  $$

  where $t$ is the target rendered text. A positive NSL means the target's non-text region aligns with $t$ more strongly than the matched bases do.

- **Goal-driven semantic leakage (GSL)** asks two visual yes/no questions about the intended goal on the masked target image and averages their soft scores. A score $>0.5$ is treated as goal-level leakage.

## Get the data

The [RenderText SemLeak dataset](https://huggingface.co/datasets/feifeili00/rendertext_semleak) contains TSL-60 requests, base prompts, 200 benign text-rendering requests, generated images, and human labels. Download only the request CSVs to test an image generator:

```bash
python -m pip install -U huggingface_hub
hf download feifeili00/rendertext_semleak --repo-type dataset \
  --include "data/**" --local-dir .
```

TSL-60 covers 60 visual goals across 360 image requests in `data/tsl60/TSL60_requests.csv`. `data/tsl60/TSL60_base_prompts.csv` contains 144 prompts for generating the two base images per main visual prompt. `data/benign/` contains the CVTG100 and TIV100 text-rendering requests. To compare GSL scorers against human labels, also download the [images and labels](#compare-gsl-scorers).

## Install

For scoring existing images, install the [rtsl-scorer package](rtsl-scorer/README.md):

```bash
git clone https://github.com/llffff/rendertext_semleak.git
python -m pip install rtsl-scorer
# Or, from this checkout:
python -m pip install ./rtsl-scorer/core ./rtsl-scorer
```

To generate images with the included backends, use Python 3.11 and NVIDIA CUDA:

```bash
conda create -n rtsl python=3.11 pip -y
conda activate rtsl
python -m pip install "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" \
  "torchaudio==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r backends/requirements.txt
python -m pip install ./rtsl-scorer/core ./rtsl-scorer
```

For local GSL scoring, install the [Qwen VQA environment](backends/README-vqa.md).

## Evaluate an image generator

Start one image backend. The GPU assignments below are examples:

```bash
# FLUX.2-dev on seven RTX 4090 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m uvicorn backends.flux_2_dev:app --host 127.0.0.1 --port 8001
# CUDA_VISIBLE_DEVICES=0,1,2 python -m uvicorn backends.qwen_image:app --host 127.0.0.1 --port 8002
# CUDA_VISIBLE_DEVICES=0 python -m uvicorn backends.z_image_turbo:app --host 127.0.0.1 --port 8003
# CUDA_VISIBLE_DEVICES=0,1 python -m uvicorn backends.longcat:app --host 127.0.0.1 --port 8004
```

FLUX.2-dev requires access to its [model weights](https://huggingface.co/black-forest-labs/FLUX.2-dev).

- Generate the matched bases with FLUX.2-dev before evaluating NSL:

  ```bash
  python scripts/evaluation/prepare_nsl_bases.py --output outputs/bases \
    --server http://127.0.0.1:8001/generate
  ```

- Read `data/tsl60/TSL60_requests.csv` and send each `prompt` to the target image backend. Set `--host` and `--port` to its address:

  ```bash
  conda activate rtsl
  python -m scripts.evaluation.generate_tsl60 flux_2_dev --host 127.0.0.1 --port 8001
  ```

  The images are saved as `outputs/images/flux_2_dev/<id>.png`. If you choose another model, use its name in this command and in the evaluator paths below.

- Stop the image backend if it shares GPUs with the scorer, then start Qwen3-VL-32B-Instruct in the [Qwen VQA environment](backends/README-vqa.md):

  ```bash
  conda activate qwen35-vllm
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-VL-32B-Instruct --tensor-parallel-size 8 \
    --served-model-name qwen3-vl-32b --max-model-len 8192 \
    --enforce-eager --max-num-seqs 1 --port 18700
  ```

- Run the evaluator:

  ```bash
  conda activate rtsl
  python -m scripts.evaluation.evaluate_sl \
    --manifest data/tsl60/TSL60_requests.csv \
    --images outputs/images/flux_2_dev --bases outputs/bases \
    --output outputs/sl/flux_2_dev
  ```

- To measure text-rendering capability on CVTG100 and TIV100, restart the image backend and run:

  ```bash
  python -m scripts.evaluation.generate_text_rendering flux_2_dev --port 8001
  python -m scripts.evaluation.evaluate_text_rendering flux_2_dev
  ```

## Compare GSL scorers

The table compares VLMs on 1,440 TSL60 images from four open image generators. Human labels and GSL scores are classified as leakage when their normalized scores are $>0.5$. See the [benchmark guide](gsl_vlm_benchmark/README.md) to download images and labels, prepare OCR masks, and run each model.

Use **Qwen3-VL-32B-Instruct** for local scoring. For hosted scoring, **Qwen3 Omni Flash** via MuleRouter has the highest accuracy among the tested API models.

| VLM | Acc (%) | Prec (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-32B-Instruct (paper) | 88.47 | 97.41 | 82.06 | 89.08 |
| [DeepSeek V4.1 Flash](https://openrouter.ai/deepseek/deepseek-v4.1-flash) | 88.47 | 96.08 | 83.27 | 89.22 |
| [Qwen3.7 Flash](https://openrouter.ai/qwen/qwen3.7-flash) | 88.54 | 97.14 | 82.42 | 89.18 |
| [MiMo-V2.6-Flash](https://openrouter.ai/xiaomi/mimo-v2.6-flash) | 87.22 | 94.58 | 82.42 | 88.08 |
| [Qwen3 Omni Flash](https://www.mulerouter.ai/models/qwen3-omni-flash) | 91.04 | 93.61 | 90.55 | 92.05 |
| Qwen3-VL-8B-Instruct (local) | 88.19 | 92.81 | 86.06 | 89.31 |
| Qwen3.5-9B (local) | 83.68 | 91.55 | 78.79 | 84.69 |
| Qwen3.5-27B (local) | 89.65 | 94.36 | 87.15 | 90.61 |

![GSL scorer comparison](assets/gsl_vlm_benchmark.png)
