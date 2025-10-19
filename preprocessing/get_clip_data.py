from pathlib import Path

import cv2
import pandas as pd


def get_clip_data_flat(
    split_dir: Path, output_dir: Path, data: pd.DataFrame, output_fps: int = 4
):
    cameras = ["image_00"]

    for src_dir in split_dir.iterdir():
        print(f"Processing modality: {src_dir.name}")
        output_modality_dir = output_dir / src_dir.name
        output_modality_dir.mkdir(parents=True, exist_ok=True)
        mp4_modality_dir = output_dir / f"{src_dir.name}_mp4"
        mp4_modality_dir.mkdir(parents=True, exist_ok=True)

        for sequence_dir in src_dir.iterdir():
            sequence_data = data[data["sequence"] == sequence_dir.name]

            for cam in cameras:
                cam_dir = sequence_dir / cam

                for _, row in sequence_data.iterrows():
                    clip_num = int(row["clip_num"])
                    frames = [int(float(x)) for x in row["frames"].split(";")]

                    valid = True
                    frame_paths = []
                    for frame in frames:
                        frame_filename = f"{frame:010d}.png"
                        input_frame_path = cam_dir / frame_filename
                        if not input_frame_path.exists():
                            print(f"Warning: {input_frame_path} does not exist.")
                            valid = False
                        frame_paths.append(input_frame_path)

                    if not valid:
                        print(
                            f"Skipping clip {clip_num} in sequence {sequence_dir.name} due to missing frames."
                        )
                        continue

                    clip_name = f"{sequence_dir.name}_{cam}_clip_{clip_num:04d}"
                    frames_dir = output_modality_dir / clip_name
                    frames_dir.mkdir(parents=True, exist_ok=True)

                    for input_frame_path in frame_paths:
                        output_frame_path = frames_dir / input_frame_path.name
                        output_frame_path.write_bytes(input_frame_path.read_bytes())

                    video_path = mp4_modality_dir / f"{clip_name}.mp4"
                    first_frame = cv2.imread(str(frame_paths[0]))
                    height, width, _ = first_frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(
                        str(video_path), fourcc, output_fps, (width, height)
                    )

                    for frame_path in frame_paths:
                        frame = cv2.imread(str(frame_path))
                        out.write(frame)
                    out.release()

                    print(f"Saved video: {video_path.relative_to(output_dir)}")


if __name__ == "__main__":
    data_dir = Path("~/data/KITTI-360_proc").expanduser()
    split_dir = data_dir / "val"
    num_frames_per_clip = 16
    clip_dir = data_dir / f"clips_{num_frames_per_clip}_flat"
    output_dir = clip_dir / "val"

    data = pd.read_csv(clip_dir / f"data_num_frames_{num_frames_per_clip}.csv")
    data = data[data["split"] == "val"]

    get_clip_data_flat(split_dir, output_dir, data)
