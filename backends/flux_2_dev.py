"""FLUX.2-dev on one 96 GB GPU or seven 24 GB GPUs."""

import gc

import torch
from diffusers import Flux2Pipeline
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from transformers import Mistral3ForConditionalGeneration

from .common import GenerateRequest, create_app


MODEL_ID = "black-forest-labs/FLUX.2-dev"
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 4.0
TEXT_DEVICE = torch.device("cuda:0")
GPU_COUNT = torch.cuda.device_count()
SINGLE_GPU = GPU_COUNT >= 1 and torch.cuda.get_device_properties(0).total_memory >= 88 * 1024**3

if SINGLE_GPU:
    GEN_DEVICE = VAE_DEVICE = TEXT_DEVICE
    pipe = Flux2Pipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    text_encoder = pipe.text_encoder.to(TEXT_DEVICE)
    vae = pipe.vae
elif GPU_COUNT >= 7:
    GEN_DEVICE = torch.device("cuda:3")
    VAE_DEVICE = torch.device("cuda:6")
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16,
        device_map="balanced", max_memory={gpu: "20GiB" for gpu in range(3)},
        low_cpu_mem_usage=True,
    )
    transformer = Flux2Transformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
        device_map="balanced", max_memory={gpu: "19GiB" for gpu in range(3, 7)},
        low_cpu_mem_usage=True,
    )
    vae = AutoencoderKLFlux2.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(VAE_DEVICE)
    pipe = Flux2Pipeline.from_pretrained(
        MODEL_ID, text_encoder=text_encoder, transformer=transformer, vae=vae,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
else:
    raise RuntimeError("FLUX.2-dev requires one GPU with at least 88 GiB or seven visible GPUs")

original_decode = vae.decode


def decode_on_vae_device(latents, *args, **kwargs):
    return original_decode(latents.to(VAE_DEVICE), *args, **kwargs)


vae.decode = decode_on_vae_device


@torch.inference_mode()
def generate_image(prompt: str, request: GenerateRequest):
    messages = [[
        {"role": "system", "content": [{"type": "text", "text": pipe.system_message}]},
        {"role": "user", "content": [{"type": "text", "text": prompt.replace("[IMG]", "")}]},
    ]]
    inputs = pipe.tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True, return_dict=True,
        return_tensors="pt", padding="max_length", truncation=True, max_length=512,
    )
    output = text_encoder(
        input_ids=inputs["input_ids"].to(TEXT_DEVICE),
        attention_mask=inputs["attention_mask"].to(TEXT_DEVICE),
        output_hidden_states=True, use_cache=False,
    )
    states = torch.stack([output.hidden_states[layer].to(GEN_DEVICE) for layer in (10, 20, 30)], dim=1)
    batch, channels, length, width = states.shape
    prompt_embeds = states.permute(0, 2, 1, 3).reshape(batch, length, channels * width)
    del output, states, inputs

    pipe.text_encoder = None
    try:
        if SINGLE_GPU:
            text_encoder.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            pipe.transformer.to(GEN_DEVICE)
            vae.to(VAE_DEVICE)
        return pipe(
            prompt_embeds=prompt_embeds,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps or DEFAULT_STEPS,
            guidance_scale=request.guidance_scale if request.guidance_scale is not None else DEFAULT_GUIDANCE,
            generator=torch.Generator(device=GEN_DEVICE).manual_seed(request.seed),
        ).images[0]
    finally:
        if SINGLE_GPU:
            pipe.transformer.to("cpu")
            vae.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            text_encoder.to(TEXT_DEVICE)
        pipe.text_encoder = text_encoder


app = create_app(name="FLUX.2-dev", model_id=MODEL_ID, generate_image=generate_image)
