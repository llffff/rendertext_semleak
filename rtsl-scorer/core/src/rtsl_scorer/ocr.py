"""Local PP-OCRv5 and compatible HTTP OCR adapter."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from .backends import post_json


def _box(points: Any) -> list[list[float]]:
    if hasattr(points, "tolist"):
        points = points.tolist()
    return [[float(point[0]), float(point[1])] for point in (points or [])]


def _extract(results: Any) -> dict:
    texts: list[str] = []
    detections: list[dict] = []
    for result in results if isinstance(results, (list, tuple)) else [results]:
        value = result if isinstance(result, dict) else getattr(result, "json", None)
        if callable(value):
            value = value()
        if not isinstance(value, dict):
            value = result.to_dict()
        data = value.get("res", value)
        recognized = data.get("rec_texts")
        scores = data.get("rec_scores")
        polygons = data.get("rec_polys")
        if polygons is None:
            polygons = data.get("dt_polys")
        recognized = [] if recognized is None else recognized
        scores = [] if scores is None else scores
        polygons = [] if polygons is None else polygons
        if hasattr(recognized, "tolist"):
            recognized = recognized.tolist()
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        for index, text in enumerate(recognized):
            clean = str(text).strip()
            if clean:
                texts.append(clean)
            detections.append({
                "text": clean,
                "score": float(scores[index]) if index < len(scores) else None,
                "box": _box(polygons[index]),
            })
    return {"recognized_texts": texts, "detections": detections}


class LocalOCR:
    """Run PP-OCRv5 on CPU in this Python process."""

    def __init__(self):
        self._model = None

    def recognize(self, image: Path) -> dict:
        if self._model is None:
            os.environ.setdefault("FLAGS_use_mkldnn", "false")
            os.environ.setdefault("FLAGS_use_onednn", "false")
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ImportError("Install rtsl-scorer for local CPU OCR, or provide an OCR backend") from exc
            self._model = PaddleOCR(
                lang="en", ocr_version="PP-OCRv5",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu", enable_hpi=False, enable_mkldnn=False,
            )
        results = self._model.predict(
            input=str(image), text_det_thresh=0.3,
            text_det_box_thresh=0.6, text_det_unclip_ratio=1.5,
        )
        return _extract(results)


class RemoteOCR:
    """Call an OCR service compatible with POST /v1/paddleocr."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def recognize(self, image: Path) -> dict:
        result = post_json(f"{self.base_url}/v1/paddleocr", {
            "image_id": image.stem,
            "filename": image.name,
            "image_base64": base64.b64encode(image.read_bytes()).decode("ascii"),
        }, timeout=180)
        if not isinstance(result.get("recognized_texts"), list) or not isinstance(result.get("detections"), list):
            raise ValueError("Invalid PP-OCRv5 response")
        return result
