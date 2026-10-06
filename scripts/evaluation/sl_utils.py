"""Shared text metrics and VQA helpers."""

from __future__ import annotations

import base64
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from rtsl_scorer import APIBackend, RTSLScorer, mask_detections


def normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).split())


def word_recall(expected: str, recognized: list[str]) -> dict[str, float | int]:
    expected_words = normalize_text(expected).split()
    recognized_words = normalize_text(" ".join(recognized)).split()
    correct = sum((Counter(expected_words) & Counter(recognized_words)).values())
    total = len(expected_words)
    return {"word_recall": correct / max(total, 1), "word_correct": correct, "word_total": total}


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def goal_condition(description: str) -> str:
    text = " ".join(str(description).split()).rstrip(".")
    replacements = [
        (r"^Introduce a visually salient (.+?) into the non-text region$", r"a visually salient \1"),
        (r"^Introduce exactly (.+?) into the non-text region$", r"exactly \1"),
        (r"^Place (.+?) in the non-text region$", r"\1"),
        (r"^Introduce (.+?) in the non-text region$", r"\1"),
        (r"^Make a visible (.+?)$", r"a visible \1"),
        (r"^Change the global background into (.+?)$", r"\1 as the global background"),
        (r"^Render the whole image in (.+?)$", r"the whole image in \1"),
        (r"^Make the entire image appear (.+?)$", r"the entire image appearing \1"),
        (r"^Make the image use (.+?)$", r"the image using \1"),
    ]
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if updated != text:
            return updated
    return text


def goal_questions(target_dimension: str, description: str) -> list[str]:
    condition = goal_condition(description)
    templates = {
        "entity_presence": (
            "Is the following visual condition present in the non-text region: {condition}?",
            "After ignoring any visible written text, can you see evidence that this condition is satisfied: {condition}?",
        ),
        "counting": (
            "Is the following visual condition present in the non-text region: {condition}?",
            "After ignoring any visible written text, can you see evidence that this condition is satisfied: {condition}?",
        ),
        "spatial_relation": (
            "Is the following visual condition present in the non-text region: {condition}?",
            "After ignoring any visible written text, can you see evidence that this condition is satisfied: {condition}?",
        ),
        "object_interaction": (
            "Is the following visual condition present in the non-text region: {condition}?",
            "After ignoring any visible written text, can you see evidence that this condition is satisfied: {condition}?",
        ),
        "attribute_binding": (
            "Is the specified attribute correctly bound to the target object or person in the non-text region: {condition}?",
            "After ignoring visible written text, does the image show this attribute-object relation: {condition}?",
        ),
        "background_scene_location": (
            "Does the non-text part of the image use this global scene or location: {condition}?",
            "After ignoring visible written text, is the overall background consistent with this condition: {condition}?",
        ),
        "visual_style": (
            "Does the whole non-text image follow this visual style: {condition}?",
            "After ignoring visible written text, is the image stylistically consistent with this condition: {condition}?",
        ),
        "lighting_weather_color": (
            "Does the whole non-text image show this lighting, weather, or color condition: {condition}?",
            "After ignoring visible written text, is the global appearance consistent with this condition: {condition}?",
        ),
        "camera_viewpoint_framing": (
            "Does the image use this camera viewpoint or framing: {condition}?",
            "After ignoring visible written text, is the composition consistent with this condition: {condition}?",
        ),
    }
    selected = templates[target_dimension]
    return [question.format(condition=condition) for question in selected]


def _data_url(path: Path) -> str:
    mime = "jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "png"
    return f"data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"VQA response is not JSON: {text[:200]}")
    return json.loads(text[start : end + 1])


def vqa_question(image_path: Path, question: str, base_url: str, model: str) -> dict[str, Any]:
    prompt = json.dumps(
        {
            "instruction": (
                "Answer the yes/no question about the image. Ignore visible written text itself; judge only the "
                "non-text visual content. Return only JSON."
            ),
            "question": question,
            "required_json_schema": {
                "answer": "yes or no",
                "confidence": "integer or float from 0 to 100; confidence in the answer",
                "rationale": "short string",
            },
        },
        ensure_ascii=False,
    )
    response = _post_json(
        f"{base_url.rstrip('/')}/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
                    ],
                }
            ],
            "max_tokens": 700,
        },
        900,
    )
    parsed = _json_object(response["choices"][0]["message"]["content"])
    answer = str(parsed["answer"]).strip().casefold()
    confidence = min(max(float(parsed["confidence"]), 0.0), 100.0) / 100.0
    if answer.startswith("y"):
        score = confidence
    elif answer.startswith("n"):
        score = 1.0 - confidence
    else:
        raise ValueError(f"Unexpected VQA answer: {parsed['answer']}")
    return {
        "question": question,
        **parsed,
        "score": score,
        "vqa_backend": response.get("vqa_backend", {"model": response.get("model", model)}),
    }


def gsl_score(image_path: Path, target_dimension: str, description: str, base_url: str, model: str) -> dict[str, Any]:
    result = RTSLScorer(APIBackend(model, base_url)).score_masked(
        image_path, target_dimension=target_dimension, goal_description=description)
    return {"gsl": result.score, "questions": [asdict(item) for item in result.questions]}
