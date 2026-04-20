# Copyright (c) 2026 Arkādijs Sergejevs
# Adapted from ControlNet 1.1
# Original Copyright (c) 2023 ControlNet Authors
# Licensed under the Apache License 2.0.

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml
from gradio_depth import process as process_depth
from gradio_seg import process as process_seg

PROCESS_MAP = {
    "segmentation": process_seg,
    "depth": process_depth,
}


@dataclass
class InferenceConfig:
    method: str
    condition: str
    prompt: str
    a_prompt: str
    n_prompt: str
    num_samples: int
    image_resolution: int
    detect_resolution: int
    ddim_steps: int
    guess_mode: bool
    strength: float
    scale: float
    seed: int
    eta: float
    input_ds_dir: str
    output_ds_dir: str
    suffix: str = ".png"
    det: str = "None"


def save_config_yaml(config: InferenceConfig, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(vars(config), f, sort_keys=False)
    print(f"Config saved to {config_path}")


def process_sequence(clip_dir: Path, config: InferenceConfig):
    condition = config.condition
    process_fn = PROCESS_MAP.get(condition)
    if process_fn is None:
        raise ValueError(f"Unsupported condition: {condition}")

    print(f"Processing clip {clip_dir.name} with condition {condition}")
    output_dir = config.output_ds_dir / config.method / clip_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_img_path in sorted(clip_dir.glob(f"*{config.suffix}")):
        input_img = cv2.imread(str(input_img_path))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        results = process_fn(
            det=config.det,
            input_image=input_img,
            prompt=config.prompt,
            a_prompt=config.a_prompt,
            n_prompt=config.n_prompt,
            num_samples=config.num_samples,
            image_resolution=config.image_resolution,
            detect_resolution=config.detect_resolution,
            ddim_steps=config.ddim_steps,
            guess_mode=config.guess_mode,
            strength=config.strength,
            scale=config.scale,
            seed=config.seed,
            eta=config.eta,
        )
        detected_cond = results[0]
        generated_images = results[1:]
        stem = input_img_path.stem

        cv2.imwrite(
            str(output_dir / f"{stem}_{condition}{config.suffix}"),
            cv2.cvtColor(detected_cond, cv2.COLOR_RGB2BGR),
        )
        for i, img in enumerate(generated_images):
            cv2.imwrite(
                str(output_dir / f"{stem}_{condition}_{i}{config.suffix}"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )


def process_dataset(config: InferenceConfig):
    input_ds_dir = Path(config.input_ds_dir)
    condition_dir = (
        input_ds_dir / config.condition
        if config.det == "None"
        else input_ds_dir / "raw_input"
    )
    for clip_dir in condition_dir.iterdir():
        if clip_dir.is_dir():
            process_sequence(clip_dir, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset with ControlNet")

    parser.add_argument("--method", type=str, required=True)
    parser.add_argument(
        "--condition", type=str, choices=["segmentation", "depth"], required=True
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a driving scene in a town, photorealistic, sunny weather",
    )
    parser.add_argument(
        "--a_prompt",
        type=str,
        default="best quality, photorealistic, extremely detailed",
    )
    parser.add_argument(
        "--n_prompt", type=str, default="bad quality, cartoon style, unrealistic"
    )
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--detect_resolution", type=int, default=512)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument(
        "--guess_mode", action="store_true", help="Enable guess mode (default: False)"
    )
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--input_ds_dir", type=str, required=True)
    parser.add_argument("--output_ds_dir", type=str, required=True)
    parser.add_argument("--suffix", type=str, default=".png")
    parser.add_argument("--det", type=str, default="None")

    args = parser.parse_args()

    config = InferenceConfig(**vars(args))

    # Save config
    output_method_dir = config.output_ds_dir / config.method
    save_config_yaml(config, output_method_dir)

    process_dataset(config)
