from pathlib import Path

import cv2


def controlnet_to_vbench(root_dir, output_dir, fps):
    for method_dir in sorted(root_dir.iterdir()):
        method_output_dir = output_dir / method_dir.name
        method_output_dir.mkdir(parents=True, exist_ok=True)

        for clip_dir in sorted(method_dir.iterdir()):
            frames = sorted(
                [
                    f
                    for f in clip_dir.iterdir()
                    if f.is_file() and f.name.endswith("_0.png")
                ]
            )

            first_frame = cv2.imread(str(frames[0]))
            height, width, _ = first_frame.shape

            output_path = method_output_dir / f"{clip_dir.name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            for frame in frames:
                img = cv2.imread(str(frame))
                writer.write(img)

            writer.release()
            print(f"Saved video: {output_path}")


if __name__ == "__main__":
    root_dir = Path("~/data/KITTI-360_output/val/ControlNet_v11").expanduser()
    output_dir = Path("~/data/KITTI-360_output/VBench/ControlNet_v11").expanduser()
    fps = 4
    output_dir.mkdir(parents=True, exist_ok=True)
    controlnet_to_vbench(root_dir, output_dir, fps)
