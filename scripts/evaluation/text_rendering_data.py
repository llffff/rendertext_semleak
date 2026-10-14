"""Released text-rendering subsets and output paths."""

import csv
import json
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "text_rendering"
MODELS = ("flux_2_dev", "qwen_image", "z_image_turbo", "longcat", "custom")
GENERATION_HOST = "http://127.0.0.1"


class Sample(NamedTuple):
    id: str
    prompt: str
    expected_texts: list[str]


def load_dataset(name: str) -> list[Sample]:
    with (ROOT / "data" / "benign" / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        samples = [Sample(row["id"], row["prompt"], json.loads(row["expected_texts"])) for row in rows]
    if len(samples) != 100 or len({sample.id for sample in samples}) != 100:
        raise ValueError(f"{name} must contain 100 unique samples")
    return samples


def generation_url(port: int) -> str:
    return f"{GENERATION_HOST}:{port}/generate"
