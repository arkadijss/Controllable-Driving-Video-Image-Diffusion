import shutil
from pathlib import Path


def controlvideo_to_vbench(root_dir, output_root, prompt):
    output_root.mkdir(exist_ok=True)

    for method_dir in sorted(root_dir.iterdir()):
        method_output_dir = output_root / method_dir.name
        method_output_dir.mkdir(parents=True, exist_ok=True)

        for clip_dir in sorted(method_dir.iterdir()):

            for mp4_file in clip_dir.glob("*.mp4"):
                if mp4_file.stem == prompt:
                    output_path = method_output_dir / f"{clip_dir.name}.mp4"
                    shutil.copy(mp4_file, output_path)
                    print(f"Copied {mp4_file} to {output_path}")
                    break


if __name__ == "__main__":
    prompt = "a driving scene in a town, photorealistic, sunny weather"
    root_dir = Path("~/data/KITTI-360_output/val/ControlVideo").expanduser()
    output_root = Path("~/data/KITTI-360_output/VBench/ControlVideo").expanduser()
    controlvideo_to_vbench(root_dir, output_root, prompt)
