"""OCR-mask the four open-model image sets for GSL comparison."""

import argparse
import csv
import json
from pathlib import Path

from rtsl_scorer import LocalOCR, mask_rendered_text

from scripts.evaluation.woi import woi_status


MODELS = ("flux_2_dev", "qwen_image", "z_image_turbo", "longcat")
FIELDS = ("model", "id", "mask_strategy", "original_word_recall",
          "masked_word_recall", "woi_status")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Downloaded Hugging Face dataset directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/gsl_vlm_benchmark/masked_log"))
    args = parser.parse_args()

    requests = read_csv(args.dataset / "data/tsl60/TSL60_requests.csv")
    words = {row["id"]: row["word_of_interest"] for row in
             read_csv(args.dataset / "data/tsl60/TSL60_word_of_interest.csv")}
    ocr = None
    summary = []
    for model in MODELS:
        for row in requests:
            image_id = row["id"]
            image = args.dataset / "images" / model / f"{image_id}.png"
            masked = args.output / "images" / model / f"{image_id}.png"
            log = args.output / "evaluations" / model / f"{image_id}.json"
            if masked.exists() and log.exists():
                result = json.loads(log.read_text(encoding="utf-8"))
            else:
                if ocr is None:
                    ocr = LocalOCR()
                mask = mask_rendered_text(image, row["rendered_text"], masked, ocr,
                                          require_clear=False)
                result = {
                    "mask": {"strategy": mask.strategy, "boxes": mask.boxes},
                    "ocr": {"original": mask.original_ocr, "masked": mask.masked_ocr},
                    "original_word_recall": mask.word_recall,
                    "masked_word_recall": mask.masked_word_recall,
                }
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
            summary.append({
                "model": model, "id": image_id,
                "mask_strategy": result["mask"]["strategy"],
                "original_word_recall": result["original_word_recall"],
                "masked_word_recall": result["masked_word_recall"],
                "woi_status": woi_status(words[image_id],
                                          result["ocr"]["original"]["recognized_texts"]),
            })
        print(f"{model}: {len(requests)} images")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved {len(summary)} masked images to {args.output}")


if __name__ == "__main__":
    main()
