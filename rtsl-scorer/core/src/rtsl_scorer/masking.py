"""OCR masking used before visual-goal scoring."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw, ImageFilter, ImageStat


class OCRBackend(Protocol):
    def recognize(self, image: Path) -> dict: ...


@dataclass(frozen=True)
class MaskResult:
    strategy: str
    word_recall: float
    masked_word_recall: float
    original_ocr: dict
    masked_ocr: dict
    boxes: list[dict]


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).split())


def _word_recall(expected: str, recognized: list[str]) -> float:
    expected_words = _normalize(expected).split()
    recognized_words = _normalize(" ".join(recognized)).split()
    return sum((Counter(expected_words) & Counter(recognized)).values()) / max(len(expected_words), 1)


def _target_detection(text: str, target: str) -> bool:
    detected, expected = _normalize(text), _normalize(target)
    if not detected or not expected:
        return False
    return detected in expected or expected in detected or bool(set(detected.split()) & set(expected.split()))


def _expand_polygon(points: list[list[float]], padding: float, width: int, height: int) -> list[tuple[float, float]]:
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    expanded = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        expanded.append((
            min(max(x + padding * dx / length, 0), width - 1),
            min(max(y + padding * dy / length, 0), height - 1),
        ))
    return expanded


def _mask(image_path: Path, output_path: Path, detections: list[dict], target: str, mode: str) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    padding = 4.0 if mode == "blur" else 18.0
    for row in detections:
        if not _target_detection(str(row.get("text", "")), target):
            continue
        points = row.get("box") or []
        if len(points) >= 3:
            draw.polygon(_expand_polygon(points, padding, *image.size), fill=255)
    if mode == "blur":
        fill = image.filter(ImageFilter.GaussianBlur(radius=18.0))
    else:
        mean = ImageStat.Stat(image, mask).mean or [128, 128, 128]
        fill = Image.new("RGB", image.size, tuple(round(value) for value in mean[:3]))
    Image.composite(fill, image, mask).save(output_path)


def mask_detections(image: Path, output: Path, detections: list[dict],
                    rendered_text: str, *, mode: str) -> list[dict]:
    """Mask the matching OCR boxes in an image."""
    if mode not in {"blur", "mean"}:
        raise ValueError(f"Unknown mask mode: {mode}")
    selected = [row for row in detections if _target_detection(str(row.get("text", "")), rendered_text)]
    output.parent.mkdir(parents=True, exist_ok=True)
    _mask(image, output, selected, rendered_text, mode)
    return selected


def mask_rendered_text(image: Path, rendered_text: str, output: Path, ocr: OCRBackend,
                       *, require_clear: bool = True) -> MaskResult:
    """Mask OCR detections and return the strategy and original word recall."""
    original = ocr.recognize(image)
    recall = _word_recall(rendered_text, original["recognized_texts"])
    for mode in ("blur", "mean"):
        selected = mask_detections(image, output, original["detections"], rendered_text, mode=mode)
        masked = ocr.recognize(output)
        residual = _word_recall(rendered_text, masked["recognized_texts"])
        if residual == 0:
            return MaskResult(mode, recall, residual, original, masked, selected)
    if not require_clear:
        return MaskResult(mode, recall, residual, original, masked, selected)
    raise ValueError(f"Rendered text remains readable after masking: {image}")
