"""Expose OpenRouter vision completions at the evaluator's local VQA endpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from dotenv import dotenv_values


MODEL = "qwen/qwen3-vl-32b-instruct"
SERVED_MODEL = "qwen3-vl-32b"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
app = FastAPI(title="OpenRouter VQA adapter")


def configured_models() -> tuple[str, str]:
    settings = dotenv_values(ENV_FILE)
    model = os.environ.get("OPENROUTER_VQA_MODEL") or settings.get("OPENROUTER_VQA_MODEL") or MODEL
    served = (os.environ.get("OPENROUTER_VQA_SERVED_MODEL")
              or settings.get("OPENROUTER_VQA_SERVED_MODEL") or SERVED_MODEL)
    return model.strip(), served.strip()


def api_key() -> str:
    value = os.environ.get("OPENROUTER_API_KEY")
    if value is None:
        value = dotenv_values(ENV_FILE).get("OPENROUTER_API_KEY")
    return (value or "").strip()


@app.get("/health")
def health() -> dict[str, Any]:
    model, served = configured_models()
    return {"configured": bool(api_key()), "backend": "openrouter",
            "model": model, "served_model": served}


@app.post("/v1/chat/completions")
def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
    model, served = configured_models()
    key = api_key()
    if not key:
        raise HTTPException(503, "Set OPENROUTER_API_KEY in the project .env file or environment.")
    if payload.get("model") not in {served, model}:
        raise HTTPException(400, f"Use model {served!r} or {model!r}.")
    if payload.get("stream"):
        raise HTTPException(400, "This adapter supports non-streaming evaluation requests only.")
    if not isinstance(payload.get("messages"), list) or not payload["messages"]:
        raise HTTPException(400, "messages must be a non-empty list.")

    # Preserve the evaluator's prompt, image data URLs, and generation parameters.
    upstream = {**payload, "model": model}
    request = Request(
        ENDPOINT,
        data=json.dumps(upstream).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=880) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise HTTPException(exc.code, f"OpenRouter returned HTTP {exc.code}.") from None
    except TimeoutError:
        raise HTTPException(504, "OpenRouter request timed out.") from None
    except URLError:
        raise HTTPException(502, "Could not reach OpenRouter.") from None
    except (ValueError, UnicodeError):
        raise HTTPException(502, "OpenRouter returned invalid JSON.") from None

    # An upstream error may arrive in a successful HTTP response.
    if not isinstance(result, dict) or result.get("error"):
        raise HTTPException(502, "OpenRouter returned an inference error.")
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(502, "OpenRouter returned no completion.") from None
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(502, "OpenRouter returned an empty text completion.")
    result["vqa_backend"] = {
        "backend": "openrouter",
        "requested_model": model,
        "model": result.get("model", model),
        "provider": result.get("provider"),
        "request_id": result.get("id"),
    }
    return result
