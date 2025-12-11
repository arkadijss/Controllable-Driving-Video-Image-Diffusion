import shutil
from pathlib import Path

import utils
import yaml


def controlnet_to_vbench(input_dir, output_dir, fps):
    output_dir.mkdir(parents=True, exist_ok=True)

    for submethod_dir in sorted(input_dir.iterdir()):
        submethod_output_dir = output_dir / submethod_dir.name
        submethod_output_dir.mkdir(parents=True, exist_ok=True)

        video_output_dir = submethod_output_dir / "output_videos"
        frame_output_dir = submethod_output_dir / "output_frames"
        video_output_dir.mkdir(exist_ok=True)
        frame_output_dir.mkdir(exist_ok=True)

        config_path = submethod_dir / "config.yaml"
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        condition = config["condition"] if config["det"] == "None" else config["det"]
        vbench_config = {"conditions": [condition]}
        config_output_path = submethod_output_dir / "vbench_config.yaml"
        with open(config_output_path, "w") as file:
            yaml.dump(vbench_config, file)

        for clip_dir in sorted(submethod_dir.iterdir()):
            if not clip_dir.is_dir():
                continue

            frame_paths = sorted(
                [
                    frame_path
                    for frame_path in clip_dir.iterdir()
                    if frame_path.name.endswith("_0.png")
                ]
            )

            video_output_path = video_output_dir / f"{clip_dir.name}.mp4"
            utils.frames_to_video(frame_paths, video_output_path, fps)
            print(f"Saved video: {video_output_path}")

            clip_frame_output_dir = frame_output_dir / f"{clip_dir.name}"
            clip_frame_output_dir.mkdir(exist_ok=True)
            for frame_path in frame_paths:
                frame_id = frame_path.name.split("_")[0]
                dst_path = clip_frame_output_dir / f"{frame_id}{frame_path.suffix}"
                shutil.copy(frame_path, dst_path)
            print(f"Saved frames: {clip_frame_output_dir}")


if __name__ == "__main__":
    input_dir = Path("~/data/KITTI-360_output/val/ControlNet_v11").expanduser()
    output_dir = Path("~/data/KITTI-360_output/VBench/ControlNet_v11").expanduser()
    fps = 4
    controlnet_to_vbench(input_dir, output_dir, fps)
