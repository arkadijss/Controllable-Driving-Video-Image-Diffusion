import shutil
from pathlib import Path

import utils
import yaml


def ctrl_adapter_to_vbench(input_dir, output_dir, fps):
    output_dir.mkdir(exist_ok=True)

    for submethod_dir in sorted(input_dir.iterdir()):
        submethod_output_dir = output_dir / submethod_dir.name
        submethod_output_dir.mkdir(exist_ok=True)

        video_output_dir = submethod_output_dir / "output_videos"
        frame_output_dir = submethod_output_dir / "output_frames"
        video_output_dir.mkdir(exist_ok=True)
        frame_output_dir.mkdir(exist_ok=True)

        conditions = [
            condition_dir.name.split("_")[1]
            for condition_dir in submethod_dir.iterdir()
            if "condition" in condition_dir.name
        ]
        conditions = list(set(conditions))
        vbench_config = {"conditions": conditions}
        config_output_path = submethod_output_dir / "vbench_config.yaml"
        with open(config_output_path, "w") as file:
            yaml.dump(vbench_config, file)

        frames_root = submethod_dir / "output_frames"
        for clip_dir in sorted(frames_root.iterdir()):
            video_output_path = video_output_dir / f"{clip_dir.name}.mp4"
            frame_paths = sorted(clip_dir.glob("*.png"))
            utils.frames_to_video(frame_paths, video_output_path, fps=fps)
            print(f"Saved video: {video_output_path}")

            clip_frame_output_dir = frame_output_dir / f"{clip_dir.name}"
            shutil.copytree(clip_dir, clip_frame_output_dir, dirs_exist_ok=True)
            print(f"Saved frames: {clip_frame_output_dir}")


if __name__ == "__main__":
    input_dir = Path("~/data/KITTI-360_output/val/Ctrl-Adapter").expanduser()
    output_dir = Path("~/data/KITTI-360_output/VBench/Ctrl-Adapter").expanduser()
    fps = 4
    ctrl_adapter_to_vbench(input_dir, output_dir, fps=fps)
