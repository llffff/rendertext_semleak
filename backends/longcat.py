"""LongCat-Image backend on two GPUs."""

import torch
from diffusers import LongCatImagePipeline

from .common import GenerateRequest, create_app


MODEL_ID = "meituan-longcat/LongCat-Image"
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 4.0

pipe = LongCatImagePipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    max_memory={0: "20GiB", 1: "20GiB"},
)


@torch.inference_mode()
def generate_image(prompt: str, request: GenerateRequest):
    return pipe(
        prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps or DEFAULT_STEPS,
        guidance_scale=(
            request.guidance_scale
            if request.guidance_scale is not None
            else DEFAULT_GUIDANCE
        ),
        num_images_per_prompt=1,
        generator=torch.Generator("cuda").manual_seed(request.seed),
        enable_cfg_renorm=True,
        enable_prompt_rewrite=True,
    ).images[0]


app = create_app(
    name="LongCat-Image",
    model_id=MODEL_ID,
    generate_image=generate_image,
)
