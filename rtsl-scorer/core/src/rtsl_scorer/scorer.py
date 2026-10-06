"""Public scoring interface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, Sequence

from .backends import APIBackend, read_api_key
from .clip import LocalCLIP, RemoteCLIP
from .masking import MaskResult, OCRBackend, mask_rendered_text
from .ocr import LocalOCR, RemoteOCR
from .questions import goal_questions


PROMPT_INSTRUCTION = (
    "Answer the yes/no question about the image. Ignore visible written text itself; judge only the "
    "non-text visual content. Return only JSON."
)


class VLMBackend(Protocol):
    model: str

    def ask(self, image: Path, prompt: str) -> tuple[str, str]: ...


class CLIPBackend(Protocol):
    def score(self, image: Path, text: str) -> float: ...


@dataclass(frozen=True)
class QuestionResult:
    question: str
    answer: str
    confidence: float
    rationale: str
    score: float
    response_model: str


@dataclass(frozen=True)
class ScoreResult:
    score: float
    questions: tuple[QuestionResult, QuestionResult]
    model: str
    mask_strategy: str | None = None
    word_recall: float | None = None

    @property
    def leakage(self) -> bool:
        return self.score > 0.5


@dataclass(frozen=True)
class BaseImage:
    path: str | Path
    rendered_text: str


@dataclass(frozen=True)
class NSLResult:
    direct_tsl: float
    base_tsl: float
    nsl: float
    base_scores: tuple[float, float]
    word_recall: float | None


def _parse_answer(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"VQA response is not JSON: {text[:200]}")
    return json.loads(text[start : end + 1])


def _select_questions(target_dimension: str | None, goal_description: str | None,
                      questions: Sequence[str] | None) -> Sequence[str]:
    if questions is None:
        if target_dimension is None or goal_description is None:
            raise ValueError("Pass target_dimension and goal_description, or two questions")
        return goal_questions(target_dimension, goal_description)
    if target_dimension is not None or goal_description is not None:
        raise ValueError("Pass either goal fields or questions, not both")
    if len(questions) != 2 or any(not isinstance(item, str) or not item.strip() for item in questions):
        raise ValueError("Pass exactly two non-empty questions")
    return questions


class RTSLScorer:
    """Score whether a visual goal appears outside an image's rendered text."""

    def __init__(self, backend: VLMBackend, *, ocr: OCRBackend | None = None,
                 clip: CLIPBackend | None = None):
        self.backend = backend
        self.ocr = ocr if ocr is not None else LocalOCR()
        self.clip = clip if clip is not None else LocalCLIP()

    @classmethod
    def from_vllm(cls, model: str, base_url: str, *, ocr_url: str | None = None,
                  clip_url: str | None = None) -> RTSLScorer:
        """Connect to a local OpenAI-compatible vLLM server."""
        return cls(APIBackend(model, base_url),
                   ocr=RemoteOCR(ocr_url) if ocr_url else None,
                   clip=RemoteCLIP(clip_url) if clip_url else None)

    @classmethod
    def from_openrouter(cls, model: str = "qwen/qwen3-vl-32b-instruct", *,
                        api_key_env: str = "OPENROUTER_API_KEY", env_file: str | Path = ".env",
                        ocr_url: str | None = None, clip_url: str | None = None,
                        disable_reasoning: bool = False) -> RTSLScorer:
        """Use OpenRouter for VLM scoring and local OCR/CLIP by default."""
        return cls(APIBackend(model, "https://openrouter.ai/api/v1",
                              read_api_key(api_key_env, env_file),
                              reasoning={"enabled": False} if disable_reasoning else None),
                   ocr=RemoteOCR(ocr_url) if ocr_url else None,
                   clip=RemoteCLIP(clip_url) if clip_url else None)

    @classmethod
    def from_deepseek(cls, model: str = "deepseek-flash", *,
                      api_key_env: str = "DEEPSEEK_API_KEY", env_file: str | Path = ".env",
                      disable_thinking: bool = False) -> RTSLScorer:
        """Use the official DeepSeek API for vision-language scoring."""
        return cls(APIBackend(model, "https://api.deepseek.com",
                              read_api_key(api_key_env, env_file),
                              thinking={"type": "disabled"} if disable_thinking else None))

    def score_masked(self, image: str | Path, *, target_dimension: str | None = None,
                     goal_description: str | None = None,
                     questions: Sequence[str] | None = None) -> ScoreResult:
        """Score an image whose rendered text has already been OCR-masked."""
        path = Path(image)
        selected_questions = _select_questions(target_dimension, goal_description, questions)
        details = []
        for question in selected_questions:
            prompt = json.dumps({
                "instruction": PROMPT_INSTRUCTION,
                "question": question,
                "required_json_schema": {
                    "answer": "yes or no",
                    "confidence": "integer or float from 0 to 100; confidence in the answer",
                    "rationale": "short string",
                },
            }, ensure_ascii=False)
            content, response_model = self.backend.ask(path, prompt)
            answer = _parse_answer(content)
            if (self.backend.model in {"deepseek/deepseek-v4.1-flash", "deepseek-flash"}
                    and "rationale" not in answer and "reasoning" in answer):
                answer["rationale"] = answer["reasoning"]
            normalized = str(answer["answer"]).strip().casefold()
            confidence = float(answer["confidence"])
            confidence = min(max(confidence, 0.0), 100.0)
            if normalized.startswith("y"):
                question_score = confidence / 100.0
            elif normalized.startswith("n"):
                question_score = 1.0 - confidence / 100.0
            else:
                raise ValueError(f"Unexpected VQA answer: {answer['answer']}")
            details.append(QuestionResult(
                question=question, answer=str(answer["answer"]), confidence=confidence,
                rationale=str(answer["rationale"]), score=question_score,
                response_model=response_model,
            ))
        return ScoreResult(
            score=sum(item.score for item in details) / len(details),
            questions=(details[0], details[1]), model=self.backend.model,
        )

    def mask_image(self, image: str | Path, *, rendered_text: str,
                   output: str | Path, require_clear: bool = True) -> MaskResult:
        """Save a persistent OCR mask and return the OCR evidence."""
        if not rendered_text.strip():
            raise ValueError("rendered_text must identify the text to mask")
        return mask_rendered_text(Path(image), rendered_text, Path(output), self.ocr,
                                  require_clear=require_clear)

    def score(self, image: str | Path, *, rendered_text: str,
              target_dimension: str | None = None, goal_description: str | None = None,
              questions: Sequence[str] | None = None) -> ScoreResult:
        """OCR-mask rendered text in a generated image, then score its visual goal."""
        if not rendered_text.strip():
            raise ValueError("rendered_text must identify the text to mask")
        selected_questions = _select_questions(target_dimension, goal_description, questions)
        with TemporaryDirectory() as directory:
            masked = Path(directory) / "masked.png"
            mask = self.mask_image(image, rendered_text=rendered_text, output=masked)
            result = self.score_masked(masked, questions=selected_questions)
        return ScoreResult(result.score, result.questions, result.model,
                           mask.strategy, mask.word_recall)

    def score_nsl(self, image: str | Path, *, rendered_text: str,
                  bases: Sequence[BaseImage]) -> NSLResult:
        """Compute CLIP NSL against two matched bases after OCR masking."""
        if not rendered_text.strip():
            raise ValueError("rendered_text must identify the target text")
        if len(bases) != 2 or any(not base.rendered_text.strip() for base in bases):
            raise ValueError("Pass exactly two bases, each with its rendered_text")
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            target_path = folder / "target.png"
            target_mask = self.mask_image(image, rendered_text=rendered_text, output=target_path)
            masked_bases = []
            for index, base in enumerate(bases):
                masked_base = folder / f"base_{index}.png"
                self.mask_image(base.path, rendered_text=base.rendered_text, output=masked_base)
                masked_bases.append(masked_base)
            return self.score_nsl_masked(target_path, rendered_text=rendered_text,
                                         bases=masked_bases, word_recall=target_mask.word_recall)

    def score_nsl_masked(self, image: str | Path, *, rendered_text: str,
                         bases: Sequence[str | Path],
                         word_recall: float | None = None) -> NSLResult:
        """Compute NSL from persistent target and matched-base masks."""
        if len(bases) != 2:
            raise ValueError("Pass exactly two masked bases")
        direct_tsl = self.clip.score(Path(image), rendered_text)
        scores = tuple(self.clip.score(Path(base), rendered_text) for base in bases)
        base_tsl = sum(scores) / 2
        return NSLResult(direct_tsl, base_tsl, direct_tsl - base_tsl,
                         (scores[0], scores[1]), word_recall)
