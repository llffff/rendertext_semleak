#!/usr/bin/env python3
"""Evaluate OCR fidelity, OCR-masked CLIP NSL, and VQA GSL."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

from rtsl_scorer import APIBackend, LocalCLIP, LocalOCR, RemoteCLIP, RemoteOCR, RTSLScorer

from scripts.evaluation.sl_utils import word_recall


def clean_id(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "goal_id", "dimension_id", "target_dimension", "goal_description", "rendered_text", "main_prompt_id"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return rows


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_manifest(args.manifest)
    with args.base_prompts.open(encoding="utf-8-sig", newline="") as handle:
        base_rows = list(csv.DictReader(handle))
    bases_by_main: dict[str, list[dict[str, str]]] = defaultdict(list)
    for base in base_rows:
        bases_by_main[base["main_prompt_id"]].append(base)
    scorer = RTSLScorer(
        APIBackend(args.vqa_model, args.vqa_url),
        ocr=RemoteOCR(args.ocr_url) if args.ocr_url else LocalOCR(),
        clip=RemoteCLIP(args.clip_url) if args.clip_url else LocalCLIP(args.clip_model, args.device),
    )
    results = []
    masked_bases: dict[str, Path] = {}
    for index, row in enumerate(rows, 1):
        item_id = clean_id(row["id"])
        image_path = args.images / f"{item_id}.png"
        bases = sorted(bases_by_main[row["main_prompt_id"]], key=lambda base: int(base["base_slot"]))
        if len(bases) != 2:
            raise ValueError(f"Expected two base prompts for {row['main_prompt_id']}")
        masked_path = args.output / "masked" / f"{item_id}.png"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        mask = scorer.mask_image(image_path, rendered_text=row["rendered_text"],
                                 output=masked_path, require_clear=False)
        wr = word_recall(row["rendered_text"], mask.original_ocr["recognized_texts"])
        base_paths = []
        masked_base_paths = []
        for base in bases:
            base_id = f"{base['main_prompt_id']}__{base['base_slot']}"
            base_path = args.bases / f"{base_id}.png"
            if not base_path.is_file():
                raise FileNotFoundError(base_path)
            if base_id not in masked_bases:
                base_mask_path = args.output / "masked_bases" / f"{base_id}.png"
                scorer.mask_image(base_path, rendered_text=base["rendered_text"], output=base_mask_path)
                masked_bases[base_id] = base_mask_path
            base_paths.append(base_path)
            masked_base_paths.append(masked_bases[base_id])
        nsl = scorer.score_nsl_masked(
            masked_path, rendered_text=row["rendered_text"],
            bases=masked_base_paths,
            word_recall=mask.word_recall,
        )
        vqa = scorer.score_masked(
            masked_path, target_dimension=row["target_dimension"],
            goal_description=row["goal_description"],
        )
        results.append(
            {
                **row,
                "image_path": str(image_path),
                "base_image_paths": json.dumps([str(path) for path in base_paths]),
                "masked_image_path": str(masked_path),
                "recognized_texts": json.dumps(mask.original_ocr["recognized_texts"], ensure_ascii=False),
                **wr,
                "mask_strategy": mask.strategy,
                "mask_box_count": len(mask.boxes),
                "masked_recognized_texts": json.dumps(mask.masked_ocr["recognized_texts"], ensure_ascii=False),
                "masked_word_recall": mask.masked_word_recall,
                "mask_valid": mask.masked_word_recall == 0,
                "direct_tsl": nsl.direct_tsl,
                "base_tsl": nsl.base_tsl,
                "nsl": nsl.nsl,
                "gsl": vqa.score,
                "gsl_success": vqa.leakage,
                "gsl_details": json.dumps([asdict(question) for question in vqa.questions], ensure_ascii=False),
            }
        )
        print(f"[{index}/{len(rows)}] {row['id']}", flush=True)
    return results


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_goal: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_goal[row["goal_id"]].append(float(row["gsl"]))
    return {
        "requests": len(rows),
        "goals": len(by_goal),
        "wr_macro": fmean(float(row["word_recall"]) for row in rows),
        "wr_micro": sum(int(row["word_correct"]) for row in rows) / max(sum(int(row["word_total"]) for row in rows), 1),
        "nsl_mean": fmean(float(row["nsl"]) for row in rows),
        "gsl_mean": fmean(float(row["gsl"]) for row in rows),
        "gsl_goal_max_mean": fmean(max(scores) for scores in by_goal.values()),
        "gsl_success_rate": fmean(float(row["gsl"]) > 0.5 for row in rows),
    }


def save(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dimensions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dimensions[row["dimension_id"]].append(row)
    summary = {
        "overall": summarize_group(rows),
        "by_dimension": {key: summarize_group(value) for key, value in sorted(dimensions.items())},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--bases", type=Path, required=True)
    parser.add_argument("--base-prompts", type=Path, default=Path("data/tsl60/TSL60_base_prompts.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sl"))
    parser.add_argument("--ocr-url", help="OCR service URL; omit for local CPU OCR")
    parser.add_argument("--vqa-url", default="http://127.0.0.1:18700/v1")
    parser.add_argument("--vqa-model", default="qwen3-vl-32b")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-url", help="CLIP service URL; omit for local CLIP")
    parser.add_argument("--device", help="Local CLIP device; defaults to CUDA if available")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    evaluated = evaluate(arguments)
    save(arguments, evaluated)
