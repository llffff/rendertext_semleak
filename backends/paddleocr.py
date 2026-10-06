"""PP-OCRv5 service used by the semantic-leakage evaluator."""

from __future__ import annotations

import base64
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("FLAGS_use_mkldnn", "false")
os.environ.setdefault("FLAGS_use_onednn", "false")

from fastapi import FastAPI
from pydantic import BaseModel


class OCRRequest(BaseModel):
    image_base64: str
    image_id: str | None = None
    filename: str = "image.png"


def _box(points: Any) -> list[list[float]]:
    if hasattr(points, "tolist"):
        points = points.tolist()
    return [[float(point[0]), float(point[1])] for point in (points or [])]


def _result_dict(result: Any) -> dict[str, Any]:
    value = result if isinstance(result, dict) else getattr(result, "json", None)
    if callable(value):
        value = value()
    if not isinstance(value, dict):
        value = result.to_dict()
    return value.get("res", value)


def _extract(results: Any) -> tuple[list[str], list[dict[str, Any]]]:
    texts: list[str] = []
    detections: list[dict[str, Any]] = []
    for result in results if isinstance(results, (list, tuple)) else [results]:
        data = _result_dict(result)
        recognized = data.get("rec_texts")
        scores = data.get("rec_scores")
        polygons = data.get("rec_polys")
        if recognized is None:
            recognized = []
        if scores is None:
            scores = []
        if polygons is None:
            polygons = data.get("dt_polys")
        if polygons is None:
            polygons = []
        if hasattr(recognized, "tolist"):
            recognized = recognized.tolist()
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        for index, text in enumerate(recognized):
            clean = str(text).strip()
            if clean:
                texts.append(clean)
            detections.append(
                {
                    "text": clean,
                    "score": float(scores[index]) if index < len(scores) else None,
                    "box": _box(polygons[index]),
                }
            )
    return texts, detections


@lru_cache(maxsize=1)
def _model() -> Any:
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="en",
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=os.environ.get("PADDLEOCR_DEVICE", "cpu"),
        enable_hpi=False,
        enable_mkldnn=False,
    )


app = FastAPI(title="PP-OCRv5", version="1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/paddleocr")
def recognize(request: OCRRequest) -> dict[str, Any]:
    suffix = Path(request.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(base64.b64decode(request.image_base64))
        handle.flush()
        results = _model().predict(
            input=handle.name,
            text_det_thresh=0.3,
            text_det_box_thresh=0.6,
            text_det_unclip_ratio=1.5,
        )
    texts, detections = _extract(results)
    return {
        "image_id": request.image_id,
        "recognized_texts": texts,
        "detections": detections,
    }
