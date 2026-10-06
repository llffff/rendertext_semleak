"""Evaluate CVTG100 and TIV100 images and save per-image scores and means."""

import argparse
import csv
import json

from scripts.evaluation.text_rendering_data import MODELS, OUTPUT_ROOT, load_dataset
from scripts.evaluation.text_rendering_metrics import TextRenderingEvaluator, WordScores

BATCH_SIZE = 8


def evaluate_dataset(dataset: str, model: str, evaluator: TextRenderingEvaluator) -> None:
    samples = load_dataset(dataset)
    output_dir = OUTPUT_ROOT / dataset / model
    paths = [output_dir / "images" / f"{sample.id}.png" for sample in samples]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{dataset}: {len(missing)}/100 images missing; first: {missing[0]}")
    summary_path = output_dir / "summary.csv"
    summary_path.unlink(missing_ok=True)
    results = []
    with (output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prompt", "expected_texts", "recognized_texts", *WordScores._fields,
                         "clipscore", "scene_prompt", "scene_clipscore", "masked_scene_clipscore",
                         "masked_recognized_texts", "masked_word_recall", "mask_fill"])
        for start in range(0, len(samples), BATCH_SIZE):
            batch = samples[start:start + BATCH_SIZE]
            scores = evaluator.evaluate_batch(paths[start:start + BATCH_SIZE],
                                              [sample.prompt for sample in batch],
                                              [sample.expected_texts for sample in batch],
                                              output_dir / "masked")
            for sample, score in zip(batch, scores):
                writer.writerow([sample.id, sample.prompt, json.dumps(sample.expected_texts),
                                 json.dumps(score.recognized_texts), *score.words, score.clipscore,
                                 score.scene_prompt, score.scene_clipscore, score.masked_scene_clipscore,
                                 json.dumps(score.masked_recognized_texts), score.masked_word_recall,
                                 score.mask_fill])
            handle.flush()
            results.extend(scores)
            print(f"{dataset}: {len(results)}/100 evaluated")
    count = len(results)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_count", "word_recall", "word_precision", "word_f1", "clipscore",
                         "scene_clipscore", "masked_scene_clipscore", "masked_word_recall"])
        writer.writerow([count,
                         sum(row.words.word_recall for row in results) / count,
                         sum(row.words.word_precision for row in results) / count,
                         sum(row.words.word_f1 for row in results) / count,
                         sum(row.clipscore for row in results) / count,
                         sum(row.scene_clipscore for row in results) / count,
                         sum(row.masked_scene_clipscore for row in results) / count,
                         sum(row.masked_word_recall for row in results) / count])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--ocr-url", help="OCR service URL; omit for local CPU OCR")
    parser.add_argument("--clip-url", help="CLIP service URL; omit for local CLIP")
    parser.add_argument("--clip-device", help="Local CLIP device; defaults to CUDA if available")
    args = parser.parse_args()
    evaluator = TextRenderingEvaluator(args.ocr_url, args.clip_device, args.clip_url)
    evaluate_dataset("cvtg100", args.model, evaluator)
    evaluate_dataset("tiv100", args.model, evaluator)


if __name__ == "__main__":
    main()
