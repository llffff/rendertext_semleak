"""Score the existing TSL60 OCR-masked images with one VLM."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from time import sleep

from rtsl_scorer import APIBackend, RTSLScorer
from rtsl_scorer.backends import read_api_key
from tqdm import tqdm


IMAGE_MODELS = {"flux_2_dev", "qwen_image", "z_image_turbo", "longcat"}
MAX_RETRIES = 3
DEEPSEEK_RUN = "openrouter__deepseek_deepseek-v4.1-flash"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def samples(masked_log: Path) -> list[tuple[str, Path, dict]]:
    requests = {row["id"]: row for row in read_csv(Path("data/tsl60/TSL60_requests.csv"))}
    result = []
    for row in read_csv(masked_log / "manifest.csv"):
        if row["model"] in IMAGE_MODELS:
            image_id = f"{row['model']}/{row['id']}"
            image = masked_log / "images" / row["model"] / f"{row['id']}.png"
            result.append((image_id, image, requests[row["id"]]))
    return result


def scorer_for(args: argparse.Namespace) -> RTSLScorer:
    if args.backend == "openrouter":
        return RTSLScorer.from_openrouter(
            model=args.model, disable_reasoning=args.disable_reasoning)
    if args.backend == "deepseek":
        return RTSLScorer.from_deepseek(
            model=args.model, disable_thinking=args.disable_reasoning)
    if args.backend == "mulerouter":
        return RTSLScorer(APIBackend(
            args.model, "https://api.mulerouter.ai/vendors/openai/v1",
            read_api_key("MULEROUTER_API_KEY"), user_agent="curl/8.7.1"))
    return RTSLScorer.from_vllm(model=args.model, base_url=args.base_url)


def export_csv(path: Path, image_ids: list[str], records: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_id", "gsl_score"))
        writer.writerows((image_id, records[image_id]["gsl_score"])
                         for image_id in image_ids if image_id in records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masked-log", type=Path, required=True)
    parser.add_argument("--backend", choices=("openrouter", "deepseek", "mulerouter", "vllm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", help="Local vLLM /v1 URL")
    parser.add_argument("--disable-reasoning", action="store_true")
    args = parser.parse_args()

    if args.backend == "vllm" and not args.base_url:
        parser.error("--base-url is required for vLLM")
    if args.backend in {"vllm", "mulerouter"} and args.disable_reasoning:
        parser.error("--disable-reasoning applies only to OpenRouter or DeepSeek")

    masked_log = args.masked_log.expanduser().resolve()
    entries = samples(masked_log)
    shared_deepseek = args.disable_reasoning and (
        (args.backend, args.model) in {
            ("openrouter", "deepseek/deepseek-v4.1-flash"),
            ("deepseek", "deepseek-flash"),
        })
    run_name = (DEEPSEEK_RUN if shared_deepseek else
                f"{args.backend}__{re.sub(r'[^A-Za-z0-9._-]+', '_', args.model)}")
    run_dir = Path("outputs/gsl_vlm_benchmark/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = ({"model": "DeepSeek-V4.1-Flash", "masked_log": str(masked_log),
               "thinking": "disabled"} if shared_deepseek else {
                   "backend": args.backend, "model": args.model,
                   "masked_log": str(masked_log),
                   "base_url": args.base_url or "", "disable_reasoning": args.disable_reasoning,
               })
    config_path = run_dir / "config.json"
    if config_path.exists():
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        legacy_openrouter = {
            "backend": "openrouter", "model": "deepseek/deepseek-v4.1-flash",
            "masked_log": str(masked_log), "base_url": "", "disable_reasoning": True,
        }
        if old_config != config and not (shared_deepseek and old_config == legacy_openrouter):
            raise ValueError(f"Run configuration changed: {config_path}")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    log = run_dir / "gsl.jsonl"
    records = {row["image_id"]: row for row in read_jsonl(log)}
    scorer = scorer_for(args)
    try:
        progress = tqdm(entries, desc="Scoring", unit="image")
        for image_id, image, request in progress:
            progress.set_description_str(image_id)
            if image_id in records:
                continue
            result = None
            for attempt in range(1, MAX_RETRIES + 2):
                scorer.backend.last_response = None
                scorer.backend.last_raw_response = None
                try:
                    result = scorer.score_masked(
                        image, target_dimension=request["target_dimension"],
                        goal_description=request["goal_description"])
                    break
                except Exception as exc:
                    error = {"image_id": image_id, "attempt": attempt,
                             "error": f"{type(exc).__name__}: {exc}"}
                    tqdm.write(f"{image_id}: attempt {attempt}/{MAX_RETRIES + 1}: {error['error']}")
                    if scorer.backend.last_raw_response is not None:
                        error["prompt"] = scorer.backend.last_prompt
                        error["raw_response"] = scorer.backend.last_raw_response
                        tqdm.write(f"{image_id}: raw response:\n{scorer.backend.last_raw_response}")
                    append_jsonl(run_dir / "errors.jsonl", error)
                    if attempt <= MAX_RETRIES:
                        sleep(2)
            if result is None:
                tqdm.write(f"{image_id}: failed after {MAX_RETRIES + 1} attempts; continuing")
                continue

            row = {
                "image_id": image_id, "gsl_score": result.score,
                "questions": [
                    {"question": q.question, "answer": q.answer,
                     "confidence": q.confidence, "score": q.score,
                     "rationale": q.rationale}
                    for q in result.questions
                ],
            }
            append_jsonl(log, row)
            records[image_id] = row
    finally:
        progress.close()
        export_csv(run_dir / "gsl.csv", [item[0] for item in entries], records)
    print(f"Scored {len(records)}/{len(entries)} images: {run_dir / 'gsl.csv'}")


if __name__ == "__main__":
    main()
