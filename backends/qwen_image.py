"""Qwen-Image-2512 on three to five visible GPUs."""

import torch
from diffusers import QwenImagePipeline
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration

from .common import GenerateRequest, create_app


MODEL_ID = "Qwen/Qwen-Image-2512"
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 4.0
TEXT_DEVICE = torch.device("cuda:0")
GEN_DEVICE = torch.device("cuda:1")
GPU_COUNT = torch.cuda.device_count()
if GPU_COUNT not in (3, 4, 5):
    raise RuntimeError("Qwen-Image-2512 requires three, four, or five visible GPUs")
VAE_DEVICE = torch.device("cuda:0" if GPU_COUNT == 3 else f"cuda:{GPU_COUNT - 1}")
TRANSFORMER_MEMORY = "22GiB" if GPU_COUNT == 3 else "19GiB"

text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    device_map=0,
    low_cpu_mem_usage=True,
)
transformer = QwenImageTransformer2DModel.from_pretrained(
    MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
    device_map="balanced", max_memory={gpu: TRANSFORMER_MEMORY for gpu in range(1, GPU_COUNT)},
    low_cpu_mem_usage=True,
)
vae = AutoencoderKLQwenImage.from_pretrained(
    MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
).to(VAE_DEVICE)
pipe = QwenImagePipeline.from_pretrained(
    MODEL_ID, text_encoder=text_encoder, transformer=transformer, vae=vae,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
)

original_decode = vae.decode


def decode_on_vae_device(latents, *args, **kwargs):
    return original_decode(latents.to(VAE_DEVICE), *args, **kwargs)


vae.decode = decode_on_vae_device


def encode_text(prompt: str):
    text = pipe.prompt_template_encode.format(prompt)
    tokens = pipe.tokenizer(
        [text], max_length=pipe.tokenizer_max_length + pipe.prompt_template_encode_start_idx,
        padding=True, truncation=True, return_tensors="pt",
    ).to(TEXT_DEVICE)
    hidden = text_encoder(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        output_hidden_states=True,
    ).hidden_states[-1]
    visible = hidden[0][tokens.attention_mask[0].to(hidden.device).bool()]
    return visible[pipe.prompt_template_encode_start_idx:].unsqueeze(0).to(GEN_DEVICE)


@torch.inference_mode()
def generate_image(prompt: str, request: GenerateRequest):
    prompt_embeds = encode_text(prompt)
    negative_embeds = encode_text("")
    negative_mask = torch.ones(negative_embeds.shape[:2], dtype=torch.long, device=GEN_DEVICE)

    pipe.text_encoder = None
    try:
        return pipe(
            prompt_embeds=prompt_embeds.to(GEN_DEVICE),
            negative_prompt_embeds=negative_embeds.to(GEN_DEVICE),
            negative_prompt_embeds_mask=negative_mask,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps or DEFAULT_STEPS,
            true_cfg_scale=request.guidance_scale if request.guidance_scale is not None else DEFAULT_GUIDANCE,
            generator=torch.Generator(device=GEN_DEVICE).manual_seed(request.seed),
        ).images[0]
    finally:
        pipe.text_encoder = text_encoder


app = create_app(
    name="Qwen-Image-2512",
    model_id=MODEL_ID,
    generate_image=generate_image,
)
