"""Local CLIP cosine similarity and compatible HTTP adapter."""

from __future__ import annotations

import base64
from pathlib import Path

from .backends import post_json


class LocalCLIP:
    def __init__(self, model: str = "openai/clip-vit-base-patch32", device: str | None = None):
        self.model_name = model
        self.device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError("Install rtsl-scorer for local CLIP, or provide a CLIP backend") from exc
        if self._model is None:
            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_name)

    def score(self, image: Path, text: str) -> float:
        from PIL import Image
        import torch

        self._load()
        with Image.open(image) as source:
            inputs = self._processor(text=[text], images=[source.convert("RGB")],
                                     return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
            image_embedding = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embedding = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        return float((image_embedding * text_embedding).sum(dim=-1).item())

    def score_batch(self, images: list[Path], texts: list[str]) -> list[float]:
        if len(images) != len(texts):
            raise ValueError("Each image needs one text")
        if not images:
            return []
        from PIL import Image
        import torch

        self._load()
        opened = []
        for path in images:
            with Image.open(path) as source:
                opened.append(source.convert("RGB"))
        inputs = self._processor(text=texts, images=opened, return_tensors="pt",
                                 padding=True, truncation=True, max_length=77).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            scores = (image_embeds * text_embeds).sum(dim=-1).cpu().tolist()
        return [score if text.strip() else 0.0 for score, text in zip(scores, texts)]

    def compatibility(self, texts: list[str], templates: list[str]) -> list[float]:
        """Mean CLIP text similarity of each template to the supplied texts."""
        import torch

        if not texts or not templates:
            return []
        self._load()
        inputs = self._processor(text=texts + templates, return_tensors="pt", padding=True,
                                 truncation=True, max_length=77).to(self.device)
        with torch.no_grad():
            output = self._model.text_model(**inputs)
            features = self._model.text_projection(output.pooler_output)
            features = features / features.norm(dim=-1, keepdim=True)
        return (features[:len(texts)] @ features[len(texts):].T).mean(dim=0).cpu().tolist()


class RemoteCLIP:
    """Call POST /v1/clipscore with an image and text; receive a float score."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def score(self, image: Path, text: str) -> float:
        result = post_json(f"{self.base_url}/v1/clipscore", {
            "filename": image.name,
            "image_base64": base64.b64encode(image.read_bytes()).decode("ascii"),
            "text": text,
        })
        return float(result["score"])

    def score_batch(self, images: list[Path], texts: list[str]) -> list[float]:
        if len(images) != len(texts):
            raise ValueError("Each image needs one text")
        return [self.score(image, text) if text.strip() else 0.0
                for image, text in zip(images, texts)]
