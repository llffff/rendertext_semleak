"""CLIP text similarity for selecting scene templates."""

import json
from functools import lru_cache
from urllib.request import Request, urlopen


@lru_cache(maxsize=1)
def _clip():
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to("cpu").eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def compatibility_scores(texts: list[str], templates: list[str],
                         clip_url: str | None = None) -> list[float]:
    if not texts or not templates:
        return []
    if clip_url:
        request = Request(
            f"{clip_url.rstrip('/')}/v1/compatibility",
            data=json.dumps({"texts": texts, "templates": templates}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            return json.load(response)["scores"]

    import torch

    model, processor = _clip()

    def embeddings(items: list[str]):
        inputs = processor(text=items, return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to("cpu")
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            return features / features.norm(dim=-1, keepdim=True)

    return (embeddings(texts) @ embeddings(templates).T).mean(dim=0).cpu().tolist()
