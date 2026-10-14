"""CLIP cosine-similarity service for the RTSL scorer."""

from __future__ import annotations

import base64
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from rtsl_scorer.clip import LocalCLIP
from scripts.evaluation.clip_compatibility import compatibility_scores


class CLIPRequest(BaseModel):
    filename: str = "image.png"
    image_base64: str
    text: str


class CompatibilityRequest(BaseModel):
    texts: list[str]
    templates: list[str]


@lru_cache(maxsize=1)
def _model() -> LocalCLIP:
    return LocalCLIP()


app = FastAPI(title="CLIP similarity", version="1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/compatibility")
def compatibility(request: CompatibilityRequest) -> dict[str, list[float]]:
    return {"scores": compatibility_scores(request.texts, request.templates)}


@app.post("/v1/clipscore")
def score(request: CLIPRequest) -> dict[str, float]:
    suffix = Path(request.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(base64.b64decode(request.image_base64))
        handle.flush()
        value = _model().score(Path(handle.name), request.text)
    return {"score": value}
