#!/usr/bin/env python3
"""Generate the two selected FLUX.2-dev base images for each main prompt."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


def post_generate(url: str, prompt: str) -> tuple[bytes, int]:
    payload = json.dumps({"prompt": prompt, "width": 1024, "height": 1024}).encode()
    request = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=900) as response:
        result = json.load(response)
    item = result["data"][0]
    if "seed" not in item:
        raise ValueError("Image backend response is missing the generation seed")
    return base64.b64decode(item["b64_json"], validate=True), item["seed"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-prompts", type=Path, default=Path("data/tsl60/TSL60_base_prompts.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, help="Directory for generation.jsonl (default: --output)")
    parser.add_argument("--server", default="http://127.0.0.1:8001/generate")
    args = parser.parse_args()

    with args.base_prompts.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = (args.log_dir or args.output) / "generation.jsonl"
    reused = generated = 0
    for index, row in enumerate(rows, 1):
        image_id = f"{row['main_prompt_id']}__{row['base_slot']}"
        if not all(char.isalnum() or char in "_.-" for char in image_id):
            raise ValueError(f"Unsafe base ID: {image_id}")
        destination = args.output / f"{image_id}.png"
        if destination.exists():
            with Image.open(destination) as image:
                image.verify()
            reused += 1
            print(f"[{index}/{len(rows)}] exists: {destination.name}", flush=True)
            continue
        image, used_seed = post_generate(args.server, row["prompt"])
        with destination.open("xb") as output:
            output.write(image)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"image": destination.name, "prompt": row["prompt"],
                                  "seed": used_seed}, ensure_ascii=False) + "\n")
        generated += 1
        print(f"[{index}/{len(rows)}] {destination.name}", flush=True)
    print(f"{reused} existing, {generated} generated")


if __name__ == "__main__":
    main()
