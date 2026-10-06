"""Shared HTTP interface for the four text-to-image backends."""

from __future__ import annotations

import base64
import io
import secrets
from collections.abc import Callable
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    seed: int | None = Field(default=None, ge=0)
    width: int = Field(default=1024, ge=64)
    height: int = Field(default=1024, ge=64)
    num_inference_steps: int | None = Field(default=None, ge=1)
    guidance_scale: float | None = Field(default=None, ge=0)


class GenerateResponseItem(BaseModel):
    b64_json: str
    seed: int
    width: int
    height: int


class GenerateResponse(BaseModel):
    data: list[GenerateResponseItem]


def extract_prompt(request: GenerateRequest) -> str:
    if request.prompt and request.prompt.strip():
        return request.prompt.strip()

    text_parts: list[str] = []
    for message in request.messages or []:
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            text_parts.extend(
                str(item["text"])
                for item in content
                if isinstance(item, dict) and item.get("text")
            )

    prompt = "\n".join(part.strip() for part in text_parts if part.strip())
    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Provide prompt or messages[].content[].text.",
        )
    return prompt


def encode_response(image: Image.Image, seed: int) -> GenerateResponse:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    width, height = image.size
    return GenerateResponse(
        data=[
            GenerateResponseItem(
                b64_json=base64.b64encode(buffer.getvalue()).decode("ascii"),
                seed=seed,
                width=width,
                height=height,
            )
        ]
    )


def create_app(
    *,
    name: str,
    model_id: str,
    generate_image: Callable[[str, GenerateRequest], Image.Image],
) -> FastAPI:
    app = FastAPI(title=f"{name} text-to-image backend")
    model_lock = Lock()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": name, "model_id": model_id}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        prompt = extract_prompt(request)
        seed = request.seed if request.seed is not None else secrets.randbelow(2**32)
        request = request.model_copy(update={"seed": seed})
        try:
            with model_lock:
                image = generate_image(prompt, request)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
        return encode_response(image, seed)

    return app
