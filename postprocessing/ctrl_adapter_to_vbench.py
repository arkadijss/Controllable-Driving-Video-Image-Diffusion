from pathlib import Path

import cv2


def frames_to_video(frames_dir, output_video_path, fps):
    frames = sorted(frames_dir.glob("*.png"))

    first_frame = cv2.imread(str(frames[0]))
    height, width, _ = first_frame.shape

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        writer.write(frame)

    writer.release()


def ctrl_adapter_to_vbench(input_root, output_root, fps):
    for method_dir in sorted(input_root.iterdir()):
        frames_root = method_dir / "output_frames"

        output_method_dir = output_root / method_dir.name
        for clip_dir in sorted(frames_root.iterdir()):
            output_video_path = output_method_dir / f"{clip_dir.name}.mp4"
            frames_to_video(clip_dir, output_video_path, fps=fps)
            print(f"Saved {output_video_path}")


if __name__ == "__main__":
    input_root = Path("~/data/KITTI-360_output/val/Ctrl-Adapter").expanduser()
    output_root = Path("~/data/KITTI-360_output/VBench/Ctrl-Adapter").expanduser()
    ctrl_adapter_to_vbench(input_root, output_root, fps=4)
