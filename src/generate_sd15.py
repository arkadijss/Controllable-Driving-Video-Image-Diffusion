import numpy as np
import torch
from diffusers import (
    AutoPipelineForInpainting,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image

generator = torch.Generator(device="cuda").manual_seed(0)


def init_generation_pipeline():
    depth_controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
    )

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=depth_controlnet,
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()
    return pipeline


def init_inpainting_pipeline():
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipeline.enable_model_cpu_offload()
    return pipeline


def preprocess_depth_image(depth_image):
    depth_image = depth_image[:, :, None]
    depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)
    depth_image = Image.fromarray(depth_image)
    return depth_image


def generate_image(pipeline, prompt, negative_prompt, depth_image):
    image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        image=depth_image,
        num_inference_steps=30,
        generator=generator,
    ).images[0]

    return image


def inpaint_image(pipeline, prompt, negative_prompt, init_image, mask_image):
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        generator=generator,
    ).images[0]
    return image
