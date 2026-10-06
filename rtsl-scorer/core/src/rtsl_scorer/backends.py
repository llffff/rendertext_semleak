"""Vision-language model adapters."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict, *, api_key: str | None = None, timeout: int = 900,
              on_response: Callable[[str], None] | None = None,
              user_agent: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if user_agent:
        headers["User-Agent"] = user_agent
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        if on_response is not None:
            on_response(exc.read().decode("utf-8"))
        raise
    if on_response is not None:
        on_response(body)
    return json.loads(body)


def read_api_key(variable: str, env_file: str | Path = ".env") -> str:
    """Read a named key from the process environment or a simple .env file."""
    value = os.environ.get(variable)
    if value:
        return value
    path = Path(env_file)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            item = line.strip().removeprefix("export ")
            key, sep, raw = item.partition("=")
            if sep and key.strip() == variable:
                value = raw.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                break
    if not value:
        raise ValueError(f"Set {variable} in the environment or {path}")
    return value


class APIBackend:
    def __init__(self, model: str, base_url: str, api_key: str | None = None,
                 reasoning: dict | None = None, thinking: dict | None = None,
                 user_agent: str | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.reasoning = reasoning
        self.thinking = thinking
        self.user_agent = user_agent
        self.last_response = None
        self.last_raw_response = None
        self.last_prompt = None

    def ask(self, image: Path, prompt: str) -> tuple[str, str]:
        self.last_response = None
        self.last_raw_response = None
        self.last_prompt = prompt
        mime = mimetypes.guess_type(image.name)[0] or "image/png"
        data_url = f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode('ascii')}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            "max_tokens": 700,
        }
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        response = post_json(
            f"{self.base_url}/chat/completions", payload, api_key=self.api_key,
            on_response=lambda body: setattr(self, "last_raw_response", body),
            user_agent=self.user_agent)
        self.last_response = response
        return response["choices"][0]["message"]["content"], response.get("model", self.model)
