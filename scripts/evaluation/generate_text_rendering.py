"""Generate CVTG100, then TIV100, with one running image backend."""

import argparse
import base64
import io
import json

from PIL import Image

from scripts.evaluation.sl_utils import _post_json
from scripts.evaluation.text_rendering_data import MODELS, OUTPUT_ROOT, generation_url, load_dataset


def generate_dataset(dataset: str, model: str, port: int) -> None:
    samples = load_dataset(dataset)
    image_dir = OUTPUT_ROOT / dataset / model / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples, 1):
        path = image_dir / f"{sample.id}.png"
        if path.exists():
            with Image.open(path) as image:
                image.verify()
            print(f"{dataset} {index}/100: {sample.id} (exists)")
            continue
        response = _post_json(generation_url(port), {"prompt": sample.prompt}, 900)
        item = response["data"][0]
        seed = item["seed"]
        image_bytes = base64.b64decode(item["b64_json"], validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.convert("RGB").save(path.with_suffix(".tmp"), format="PNG")
        path.with_suffix(".tmp").replace(path)
        with (image_dir.parent / "generation.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"id": sample.id, "prompt": sample.prompt,
                                  "image": path.relative_to(image_dir.parent).as_posix(),
                                  "seed": seed}, ensure_ascii=False) + "\n")
        print(f"{dataset} {index}/100: {sample.id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--port", type=int, required=True, help="Port of the running image backend")
    args = parser.parse_args()
    generate_dataset("cvtg100", args.model, args.port)
    generate_dataset("tiv100", args.model, args.port)


if __name__ == "__main__":
    main()
