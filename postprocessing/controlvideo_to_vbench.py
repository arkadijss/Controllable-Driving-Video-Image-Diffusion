import shutil
from pathlib import Path

import utils
import yaml


def controlvideo_to_vbench(input_dir, output_dir, prompt):
    output_dir.mkdir(exist_ok=True)

    for submethod_dir in sorted(input_dir.iterdir()):
        submethod_output_dir = output_dir / submethod_dir.name
        submethod_output_dir.mkdir(exist_ok=True)

        video_output_dir = submethod_output_dir / "output_videos"
        frame_output_dir = submethod_output_dir / "output_frames"
        video_output_dir.mkdir(exist_ok=True)
        frame_output_dir.mkdir(exist_ok=True)

        inference_args_path = submethod_dir / "inference_args.yaml"
        with open(inference_args_path, "r") as file:
            inference_args = yaml.safe_load(file)

        condition = inference_args["condition"]
        vbench_config = {"conditions": [condition]}
        config_output_path = submethod_output_dir / "vbench_config.yaml"
        with open(config_output_path, "w") as file:
            yaml.dump(vbench_config, file)

        for clip_dir in sorted(submethod_dir.iterdir()):
            if not clip_dir.is_dir():
                continue

            video_path = clip_dir / f"{prompt}.mp4"
            video_output_path = video_output_dir / f"{clip_dir.name}.mp4"
            shutil.copy(video_path, video_output_path)
            print(f"Saved video: {video_output_path}")

            clip_frame_output_dir = frame_output_dir / clip_dir.name
            utils.video_to_frames(video_path, clip_frame_output_dir)
            print(f"Saved frames: {clip_frame_output_dir}")


if __name__ == "__main__":
    prompt = "a driving scene in a town, photorealistic, sunny weather"
    input_dir = Path("~/data/KITTI-360_output/val/ControlVideo").expanduser()
    output_dir = Path("~/data/KITTI-360_output/VBench/ControlVideo").expanduser()
    controlvideo_to_vbench(input_dir, output_dir, prompt)
