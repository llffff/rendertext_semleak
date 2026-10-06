"""Generate one image for each released TSL60 request with a local backend."""

import argparse
import base64
import csv
import io
import json

from PIL import Image

from scripts.evaluation.sl_utils import _post_json
from scripts.evaluation.text_rendering_data import MODELS, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--host", default="127.0.0.1", help="IP address or hostname of the image backend")
    parser.add_argument("--port", type=int, required=True, help="Port of the running image backend")
    args = parser.parse_args()
    model = args.model

    with (ROOT / "data/tsl60/TSL60_requests.csv").open(newline="", encoding="utf-8-sig") as handle:
        requests = list(csv.DictReader(handle))
    if len(requests) != 360 or len({row["id"] for row in requests}) != 360:
        raise ValueError("Expected 360 unique TSL60 requests")

    output = ROOT / "outputs/images" / model
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "generation.jsonl"
    logged = set()
    if log_path.exists():
        with log_path.open(encoding="utf-8") as handle:
            logged = {json.loads(line)["image"] for line in handle if line.strip()}

    for index, row in enumerate(requests, 1):
        image_path = output / f"{row['id']}.png"
        if image_path.exists():
            if image_path.name not in logged:
                raise ValueError(f"Existing image has no generation log: {image_path}")
            with Image.open(image_path) as image:
                image.verify()
            print(f"[{index}/360] exists: {image_path.name}", flush=True)
            continue

        result = _post_json(f"http://{args.host}:{args.port}/generate", {"prompt": row["prompt"]}, 900)
        item = result["data"][0]
        image_bytes = base64.b64decode(item["b64_json"], validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.convert("RGB").save(image_path.with_suffix(".tmp"), format="PNG")
        image_path.with_suffix(".tmp").replace(image_path)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": row["id"], "prompt": row["prompt"],
                                     "image": image_path.name, "seed": item["seed"]},
                                    ensure_ascii=False) + "\n")
        logged.add(image_path.name)
        print(f"[{index}/360] {image_path.name}", flush=True)


if __name__ == "__main__":
    main()
