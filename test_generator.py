"""Check a local, custom, or OpenRouter image generator with one prompt."""

import argparse
import base64
import io
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


ROOT = Path(__file__).resolve().parent
PROMPT = 'A red bicycle parked beside a white wall. A small sign on the wall reads "HELLO".'
OUTPUT = ROOT / "outputs/generator_test.png"
OPENROUTER_URL = "https://openrouter.ai/api/v1/images"


def read_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip().removeprefix("export ").strip() == "OPENROUTER_API_KEY":
                key = value.strip().strip("\"'")
                break
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY in the environment or repository .env")
    return key.strip()


def request_json(url, *, headers=None, payload=None, timeout=30):
    request = Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers or {},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    backends = parser.add_subparsers(dest="backend", required=True)
    for name in ("local", "custom"):
        backend = backends.add_parser(name)
        backend.add_argument("--url", required=True,
                             help="Backend address, for example http://127.0.0.1:8001")
    hosted = backends.add_parser("openrouter")
    hosted.add_argument("--model", required=True, help="OpenRouter image model ID")
    args = parser.parse_args()
    print(f"Prompt: {PROMPT}")

    if args.backend in ("local", "custom"):
        url = args.url.rstrip("/")
        health = request_json(f"{url}/health")
        if health.get("status") != "ok":
            raise RuntimeError(f"Backend is not ready: {health}")
        print(f"Connected to {health.get('model', url)}")
        result = request_json(
            f"{url}/generate",
            headers={"Content-Type": "application/json"},
            payload={"prompt": PROMPT, "width": 1024, "height": 1024},
            timeout=900,
        )
    else:
        headers = {"Authorization": f"Bearer {read_key()}"}
        catalog = request_json(f"{OPENROUTER_URL}/models", headers=headers)
        if args.model not in {item["id"] for item in catalog["data"]}:
            raise ValueError(f"{args.model} is not listed by the OpenRouter Images API")
        print(f"Connected to {args.model}")
        result = request_json(
            OPENROUTER_URL,
            headers={**headers, "Content-Type": "application/json"},
            payload={"model": args.model, "prompt": PROMPT},
            timeout=900,
        )

    item = result["data"][0]
    if args.backend in ("local", "custom"):
        for field in ("seed", "width", "height"):
            if field not in item:
                raise ValueError(f"Image response is missing data[0].{field}")
    encoded = item["b64_json"]
    if encoded.startswith("data:"):
        encoded = encoded.partition(",")[2]
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
        if args.backend in ("local", "custom") and image.size != (item["width"], item["height"]):
            raise ValueError("Image dimensions do not match data[0].width and data[0].height")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(OUTPUT)
        print(f"Generated {image.width}×{image.height} image: {OUTPUT}")


if __name__ == "__main__":
    main()
