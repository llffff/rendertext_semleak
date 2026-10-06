"""Z-Image-Turbo backend based on the official Diffusers example."""

import torch
from diffusers import ZImagePipeline

from .common import GenerateRequest, create_app


MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_STEPS = 9
DEFAULT_GUIDANCE = 0.0

pipe = ZImagePipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
).to("cuda")


@torch.inference_mode()
def generate_image(prompt: str, request: GenerateRequest):
    return pipe(
        prompt=prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps or DEFAULT_STEPS,
        guidance_scale=(
            request.guidance_scale
            if request.guidance_scale is not None
            else DEFAULT_GUIDANCE
        ),
        generator=torch.Generator("cuda").manual_seed(request.seed),
    ).images[0]


app = create_app(
    name="Z-Image-Turbo",
    model_id=MODEL_ID,
    generate_image=generate_image,
)
