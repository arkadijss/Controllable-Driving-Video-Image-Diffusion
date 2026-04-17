from dataclasses import dataclass
from pathlib import Path

import cv2
from gradio_depth2image import process as process_depth
from gradio_seg2image import process as process_seg

PROCESS_MAP = {
    "segmentation": process_seg,
    "depth": process_depth,
}


@dataclass
class InferenceConfig:
    method: str
    condition: str
    use_pregenerated: bool
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
    input_ds_dir: Path
    output_ds_dir: Path
    suffix: str = ".png"


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
            use_pregenerated=config.use_pregenerated,
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
    condition_dir = (
        config.input_ds_dir / config.condition
        if config.use_pregenerated
        else config.input_ds_dir / "raw_input"
    )
    for clip_dir in condition_dir.iterdir():
        if clip_dir.is_dir():
            process_sequence(clip_dir, config)


if __name__ == "__main__":
    base_dir = Path.home() / "data"
    input_ds_dir = base_dir / "KITTI-360_proc/clips_16_flat/val"
    output_ds_dir = base_dir / "KITTI-360_output/val/ControlNet_v10"

    prompt = "a driving scene in a town, photo-realistic, sunny weather"
    a_prompt = "best quality, photo-realistic, extremely detailed"
    n_prompt = "bad quality, cartoon style, unrealistic"

    common_kwargs = dict(
        prompt=prompt,
        use_pregenerated=True,
        a_prompt=a_prompt,
        n_prompt=n_prompt,
        num_samples=1,
        image_resolution=512,
        detect_resolution=512,
        ddim_steps=40,
        guess_mode=False,
        strength=1.0,
        scale=9.0,
        seed=42,
        eta=0.0,
        input_ds_dir=input_ds_dir,
        output_ds_dir=output_ds_dir,
        suffix=".png",
    )

    seg_config = InferenceConfig(
        method="ControlNet_v10_InternImage-H", condition="segmentation", **common_kwargs
    )
    depth_config = InferenceConfig(
        method="ControlNet_v10_Depth_Anything_V2", condition="depth", **common_kwargs
    )

    process_dataset(seg_config)
    process_dataset(depth_config)
