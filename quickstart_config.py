"""Quickstart settings and image/cache support. Edit settings before running the notebook."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL = "z_image_turbo"  # flux_2_dev, qwen_image, z_image_turbo, longcat, or custom
PORT = 8003
SERVER_URL = f"http://127.0.0.1:{PORT}"
IMAGE_SIZE = 1024
GENERATION_TIMEOUT = 900
VERBOSE = True

DATASET = "tsl21"  # tsl21 or tsl60
BEGIN_IDX = 0
OFFSET = 3
BASE_SOURCE = "model"  # model or flux_2_dev

VQA_SOURCE = "openrouter"  # openrouter or local
OPENROUTER_MODEL = "qwen/qwen3-vl-32b-instruct"
LOCAL_VQA_MODEL = "qwen3-vl-32b"
LOCAL_VQA_URL = "http://127.0.0.1:18700/v1"
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_DEVICE = "cpu"

OUTPUT = ROOT / "outputs/quickstart" / MODEL / "random"
BASE_DIR = OUTPUT / ("flux_2_dev_bases" if BASE_SOURCE == "flux_2_dev" else "bases")
MASKED_BASE_DIR = OUTPUT / ("masked_flux_2_dev_bases" if BASE_SOURCE == "flux_2_dev" else "masked_bases")
SHARED_BASE_DIR = ROOT / "outputs/bases"
PREVIEW_SIZE = (512, 512)


# Image preparation and cache support.
import base64
import io
import json
from dataclasses import asdict
from shutil import copy2
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image

from scripts.evaluation.text_rendering_data import MODELS

assert MODEL in MODELS
assert BASE_SOURCE in ("model", "flux_2_dev")


def image_path(row, *, masked=False):
    return OUTPUT / DATASET / ("masked" if masked else "images") / f"{row.id}.png"


def matched_bases(row, base_prompts):
    bases = base_prompts[base_prompts.main_prompt_id == row.main_prompt_id].sort_values("base_slot")
    assert len(bases) == 2, row.main_prompt_id
    return list(bases.itertuples(index=False))


def generate_png(prompt, path):
    request = Request(
        f"{SERVER_URL}/generate",
        data=json.dumps({"prompt": prompt, "width": IMAGE_SIZE, "height": IMAGE_SIZE}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=GENERATION_TIMEOUT) as response:
        item = json.load(response)["data"][0]
    with Image.open(io.BytesIO(base64.b64decode(item["b64_json"]))) as image:
        temporary = path.with_suffix(".tmp")
        image.convert("RGB").save(temporary, format="PNG")
    temporary.replace(path)


def prepare_image(path, prompt, source=None):
    if path.exists():
        with Image.open(path) as image:
            image.verify()
        action = "reused"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if source is None:
            generate_png(prompt, path)
        else:
            assert source.exists(), source
            copy2(source, path)
        action = "generated" if source is None else "copied"
    if VERBOSE:
        print(f"{action}: {path.relative_to(ROOT)}")


def signature(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def cached(method, image, **kwargs):
    """Call an RTSL method, reusing results for unchanged inputs and settings."""
    name = method.__name__
    output = None
    if name == "mask_image":
        stage = "mask"
        settings = ["rtsl-scorer", "PP-OCRv5", kwargs["rendered_text"], kwargs.get("require_clear", True)]
        output = kwargs["output"]
    elif name == "score_nsl_masked":
        stage = "nsl"
        settings = [kwargs["rendered_text"], CLIP_MODEL,
                    [[str(path.relative_to(OUTPUT)), signature(path)] for path in kwargs["bases"]]]
    elif name == "score_masked":
        stage = "gsl"
        settings = [VQA_SOURCE, method.__self__.backend.model,
                    kwargs["target_dimension"], kwargs["goal_description"]]
    else:
        raise ValueError(f"Unsupported quickstart cache method: {name}")

    relative = image.relative_to(OUTPUT)
    path = OUTPUT / "cache" / stage / relative.with_suffix(".json")
    saved = json.loads(path.read_text()) if path.exists() else {}
    entries = saved.get("entries", []) if saved.get("image_signature") == signature(image) else []
    for entry in entries:
        if entry["settings"] == settings and (
            output is None or output.exists() and entry.get("output_signature") == signature(output)
        ):
            if VERBOSE:
                print(f"cached {stage}: {relative}")
            return entry["result"]

    result = asdict(method(image, **kwargs))
    entry = {"settings": settings, "result": result}
    if output is not None:
        entry["output_signature"] = signature(output)
    entries = [old for old in entries if old["settings"] != settings] + [entry]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"image_signature": signature(image), "entries": entries},
                                    ensure_ascii=False, indent=2))
    temporary.replace(path)
    if VERBOSE:
        print(f"computed {stage}: {relative}")
    return result


def collect_scores(selected, masks, nsl_results, gsl_results=None):
    rows = []
    for row in selected:
        mask, nsl = masks[row.id], nsl_results[row.id]
        scores = {
            "dataset": DATASET, "id": row.id, "goal_id": row.goal_id,
            "base_source": BASE_SOURCE, "dimension_id": row.dimension_id,
            "word_recall": mask["word_recall"], "masked_word_recall": mask["masked_word_recall"],
            "mask_valid": mask["masked_word_recall"] == 0,
            "mask_box_count": len(mask["boxes"]), "mask_strategy": mask["strategy"],
            "direct_tsl": nsl["direct_tsl"], "base_tsl": nsl["base_tsl"], "nsl": nsl["nsl"],
        }
        if gsl_results is not None:
            gsl = gsl_results[row.id]
            scores.update(gsl=gsl["score"], gsl_success=gsl["score"] > 0.5,
                          gsl_details=json.dumps(gsl["questions"], ensure_ascii=False))
        rows.append(scores)
    return pd.DataFrame(rows)


def base_path(base, *, masked=False):
    folder = MASKED_BASE_DIR if masked else BASE_DIR
    return folder / f"{base.main_prompt_id}__{base.base_slot}.png"
