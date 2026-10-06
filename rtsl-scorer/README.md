# rtsl-scorer

Measure rendered-text semantic leakage: whether the meaning of text requested inside an image also appears in its non-text visual content. The scorer OCR-masks the text and asks a vision-language model two visual questions. Their mean is the GSL soft score; `score > 0.5` indicates leakage. With two matched base images, it also computes CLIP-based NSL.

## Install

Install the full package for local CPU OCR and CLIP. It connects to a VLM through OpenRouter or a local vLLM server:

```bash
python -m pip install rtsl-scorer
```

If OCR and CLIP run as separate services, install the lightweight client instead:

```bash
python -m pip install rtsl-scorer-core
```

`score_masked()` needs only a VLM. `score()` also needs OCR; `score_nsl()` needs OCR and CLIP.

## Score an image

Set `OPENROUTER_API_KEY` in the environment or a `.env` file, then select an image-capable model:

```python
from rtsl_scorer import RTSLScorer

scorer = RTSLScorer.from_openrouter(model="qwen/qwen3.7-flash", disable_reasoning=True)
result = scorer.score(
    "generated.png",
    rendered_text="A dog is running across the street.",
    questions=("Does the image show a dog?", "Is a dog visible in the scene?"),
)
print(result.score, result.leakage)
print(result.questions)  # answers, confidence, rationales, and individual scores
```

Pass `model=...` to select another OpenRouter VLM, `api_key_env=...` for another key variable, or `env_file=...` for another `.env` path. OpenRouter supplies VLM answers; OCR and CLIP run locally with the full installation. To use external OCR or CLIP services, pass `ocr_url=...` or `clip_url=...`. To score an already masked image without OCR, use `score_masked()`.

For a local vLLM server, pass its served model name and OpenAI-compatible base URL:

```bash
vllm serve Qwen/Qwen3-VL-32B-Instruct --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 8 --served-model-name Qwen/Qwen3-VL-32B-Instruct \
  --max-model-len 8192 --max-num-seqs 1 --enforce-eager
```

```python
scorer = RTSLScorer.from_vllm(
    model="Qwen/Qwen3-VL-32B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
)
```

The model name can be changed to another image-capable model served through the same interface.

## Use the TSL benchmark

Download the request CSVs from the [dataset](https://huggingface.co/datasets/feifeili00/rendertext_semleak) and run the following example from the directory containing `data/`. Each request provides `rendered_text`, `target_dimension`, and `goal_description`; these fields select the benchmark's two dimension-specific questions:

```python
import csv
from pathlib import Path

with open("data/tsl60/TSL60_requests.csv", newline="") as file:
    row = next(csv.DictReader(file))

result = scorer.score(
    Path("images/flux_2_dev") / f"{row['id']}.png",
    rendered_text=row["rendered_text"],
    target_dimension=row["target_dimension"],
    goal_description=row["goal_description"],
)
print(result.score, result.leakage, result.word_recall)
print(result.questions)  # two answers, confidence values, rationales, and scores
```

For a pre-masked image, call `score_masked(image, target_dimension=..., goal_description=...)`.

### Compute NSL with matched bases

The NSL input is the target image plus two base images for its `main_prompt_id`. Each base needs its own `rendered_text` from `TSL60_base_prompts.csv`. After masking all three images, the scorer compares each with the target `rendered_text` using CLIP:

$\mathrm{NSL}=\mathrm{CLIP}(I_{target},t)-\frac{1}{2}\sum_{i=1}^{2}\mathrm{CLIP}(I_{base,i},t)$.

```python
from rtsl_scorer import BaseImage

with open("data/tsl60/TSL60_base_prompts.csv", newline="") as file:
    base_rows = [base for base in csv.DictReader(file)
                 if base["main_prompt_id"] == row["main_prompt_id"]]

bases = [BaseImage(
    path=Path("outputs/bases") / f"{base['main_prompt_id']}__{base['base_slot']}.png",
    rendered_text=base["rendered_text"],
) for base in base_rows]
nsl = scorer.score_nsl(
    Path("images/flux_2_dev") / f"{row['id']}.png",
    rendered_text=row["rendered_text"],
    bases=bases,
)
print(nsl.nsl, nsl.direct_tsl, nsl.base_tsl)
```

To save masks and OCR evidence, call `mask_image(image, rendered_text=..., output=...)`. Reuse saved masks with `score_nsl_masked()` and `score_masked()`.

## Use a free-form prompt

When the prompt contains one quoted text span, supply the visual goal to test. `prepare_freeform` extracts that span and makes two general visual questions:

```python
from rtsl_scorer import prepare_freeform

request = prepare_freeform(
    'A sign reads "A dog is running across the street."',
    visual_goal="a dog running across the street",
)
result = scorer.score("generated.png", rendered_text=request.rendered_text,
                      questions=request.questions)
```

If the goal is unknown, a text model can propose one. Inspect the proposed goal before scoring:

```python
from rtsl_scorer import PromptPlanner

planner = PromptPlanner.from_openrouter(model="qwen/qwen3-30b-a3b")
request = prepare_freeform('A sign reads "A dog is running."', planner=planner)
print(request.visual_goal, request.questions)
```

The planner validates the goal against the rendered text and full prompt. It raises `ValueError` if no supported goal is found. Free-form and benchmark questions can yield different scores; pass two questions explicitly when comparing the same goal across images.

## Custom backends

For a custom VLM, pass an object with `model: str` and `ask(image: Path, prompt: str) -> tuple[str, str]` to `RTSLScorer`. The returned tuple contains a JSON answer with `answer`, `confidence`, and `rationale`, followed by the response model name. Custom OCR and CLIP objects can be passed as `ocr=` and `clip=`; they implement `recognize(image: Path) -> dict` and `score(image: Path, text: str) -> float`, respectively.

A custom free-form planner implements `infer_goal(rendered_text: str) -> str` and can be passed to `prepare_freeform(..., planner=...)`.

For HTTP services, `from_vllm()` calls the OpenAI-compatible `POST /chat/completions` endpoint. `ocr_url` and `clip_url` use `POST /v1/paddleocr` and `POST /v1/clipscore`.

## Choosing a VLM

We recommend **Qwen3-VL-32B-Instruct** for local scoring and **Qwen3 Omni Flash** via MuleRouter for hosted scoring. Qwen3 Omni Flash has the highest accuracy among the tested hosted models. The table uses 1,440 TSL60 images from four open image generators; both human and GSL scores use a leakage threshold of $>0.5$.

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

## Acknowledgments

We thank the authors of [CLIPScore](https://github.com/jmhessel/clipscore) and [VQAScore](https://github.com/linzhiqiu/t2v_metrics) for sharing their implementations.
