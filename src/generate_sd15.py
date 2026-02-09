import numpy as np
import torch
from diffusers import (
    AutoPipelineForInpainting,
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image


def init_generation_pipeline(use_segmentation=True):
    depth_controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
    )
    controlnets = [depth_controlnet]

    if use_segmentation:
        seg_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16
        )
        controlnets.append(seg_controlnet)

    controlnet = controlnets if len(controlnets) > 1 else controlnets[0]

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")

    return pipeline


def init_base_inpainting_pipeline():
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipeline.to("cuda")
    return pipeline


def init_controllable_inpainting_pipeline(use_depth=True, use_segmentation=True):
    controlnets = []

    if use_depth:
        depth_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
        )
        controlnets.append(depth_controlnet)

    if use_segmentation:
        seg_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16
        )
        controlnets.append(seg_controlnet)

    controlnet = controlnets if len(controlnets) > 1 else controlnets[0]

    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")

    return pipeline


def init_inpainting_pipeline(use_depth=True, use_segmentation=True):
    if use_depth or use_segmentation:
        pipeline = init_controllable_inpainting_pipeline(
            use_depth=use_depth, use_segmentation=use_segmentation
        )
    else:
        pipeline = init_base_inpainting_pipeline()

    return pipeline


def preprocess_depth_image(depth_image):
    depth_image = depth_image[:, :, None]
    depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)
    depth_image = Image.fromarray(depth_image)
    return depth_image


def generate_image(pipeline, prompt, negative_prompt, control_image, seed=42, **kwargs):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        generator=generator,
        **kwargs,
    ).images[0]

    return image


def inpaint_image(
    pipeline, prompt, negative_prompt, init_image, mask_image, seed=42, **kwargs
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        generator=generator,
        **kwargs,
    ).images[0]
    return image
