"""Word metrics, CLIP similarity, and OCR-masked scene evaluation."""

import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from rtsl_scorer import LocalCLIP, LocalOCR, RemoteCLIP, RemoteOCR, mask_rendered_text

from scripts.evaluation.sl_utils import normalize_text


class WordScores(NamedTuple):
    word_recall: float
    word_precision: float
    word_f1: float
    word_correct: int
    word_expected: int
    word_recognized: int


def word_scores(expected: list[str], recognized: list[str]) -> WordScores:
    expected_words = normalize_text(" ".join(expected)).split()
    recognized_words = normalize_text(" ".join(recognized)).split()
    correct = sum((Counter(expected_words) & Counter(recognized_words)).values())
    recall = correct / max(len(expected_words), 1)
    precision = correct / max(len(recognized_words), 1)
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return WordScores(recall, precision, f1, correct, len(expected_words), len(recognized_words))


class CLIPScorer(LocalCLIP):
    """CLIP scorer with a required device argument."""

    def __init__(self, device: str):
        super().__init__(device=device)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def strip_rendered_text(prompt: str, texts: list[str]) -> str:
    stripped = prompt
    for text in sorted((t for t in texts if t.strip()), key=len, reverse=True):
        escaped = re.escape(text)
        for pattern in (rf"['\"]{escaped}['\"]", escaped):
            stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
    cleanup_patterns = [
        r"\bwith the text\b",
        r"\binclude the text\b",
        r"\bcontaining the text\b",
        r"\bprinted with the text\b",
        r"\bwritten on it\b",
        r"\bwritten on\b",
        r"\breads\b",
        r"\bread\b",
        r"\bsays\b",
        r"\bsaying\b",
        r"\bshows\b",
        r"\bshowing\b",
        r"\bdisplaying\b",
        r"\bdisplayed\b",
        r"\blabeled\b",
        r"\bannouncing\b",
    ]
    for pattern in cleanup_patterns:
        stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+([,.;:])", r"\1", stripped)
    stripped = re.sub(
        r"\b(on it|on them|at the bottom|at the top|in bold font|in elegant font|in modern font)\b",
        "",
        stripped,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(r"([,.;:]){2,}", r"\1", stripped)
    stripped = re.sub(r"\(\s*\)", "", stripped)
    stripped = re.sub(r'""', "", stripped)
    stripped = re.sub(r"\s+\.", ".", stripped)
    return normalize_spaces(stripped)


class ImageScores(NamedTuple):
    recognized_texts: list[str]
    words: WordScores
    clipscore: float
    scene_prompt: str
    scene_clipscore: float
    masked_scene_clipscore: float
    masked_recognized_texts: list[str]
    masked_word_recall: float
    mask_fill: str


class TextRenderingEvaluator:
    def __init__(self, ocr_url: str | None = None, clip_device: str | None = None,
                 clip_url: str | None = None):
        self.ocr = RemoteOCR(ocr_url) if ocr_url else LocalOCR()
        self.clip = RemoteCLIP(clip_url) if clip_url else LocalCLIP(device=clip_device)

    def evaluate_batch(self, image_paths: list[Path], prompts: list[str],
                       expected_texts: list[list[str]], masked_dir: Path) -> list[ImageScores]:
        """Return scores in input order; save target-text masks under masked_dir."""
        if not len(image_paths) == len(prompts) == len(expected_texts):
            raise ValueError("Images, prompts, and expected texts must have equal lengths")
        original_texts, masked_texts, word_results, mask_fills, masked_paths = [], [], [], [], []
        for path, expected in zip(image_paths, expected_texts):
            masked_path = masked_dir / path.name
            mask = mask_rendered_text(path, " ".join(expected), masked_path, self.ocr,
                                      require_clear=False)
            original_texts.append(mask.original_ocr["recognized_texts"])
            masked_texts.append(mask.masked_ocr["recognized_texts"])
            word_results.append(word_scores(expected, mask.original_ocr["recognized_texts"]))
            mask_fills.append(mask.strategy)
            masked_paths.append(masked_path)

        scenes = [strip_rendered_text(prompt, expected) for prompt, expected in zip(prompts, expected_texts)]
        full_scores = self.clip.score_batch(image_paths, prompts)
        scene_scores = self.clip.score_batch(image_paths, scenes)
        masked_scores = self.clip.score_batch(masked_paths, scenes)
        results = []
        for i, expected in enumerate(expected_texts):
            results.append(ImageScores(original_texts[i], word_results[i], full_scores[i], scenes[i],
                                       scene_scores[i], masked_scores[i], masked_texts[i],
                                       word_scores(expected, masked_texts[i]).word_recall, mask_fills[i]))
        return results
