from pathlib import Path

import cv2
import pandas as pd


def center_crop_by_height(input_dir: Path, output_dir: Path, frame_filenames):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".png"}

    for input_path in input_dir.iterdir():
        if input_path.suffix.lower() not in image_extensions:
            continue

        # Skip files not in the frame filenames list
        if input_path.name not in frame_filenames:
            continue

        output_path = output_dir / input_path.name
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Skipping invalid image: {input_path.name}")
            continue

        h, w = img.shape[:2]

        if w > h:
            left = (w - h) // 2
            right = left + h
            cropped = img[:, left:right]
        elif h > w:
            top = (h - w) // 2
            bottom = top + w
            cropped = img[top:bottom, :]
        else:
            cropped = img

        cv2.imwrite(str(output_path), cropped)
        print(f"Cropped and saved: {output_path}")


def center_crop_raw(input_dir: Path, output_dir: Path, data):
    cameras = ["image_00"]
    for sequence_dir in input_dir.iterdir():
        sequence_data = data[data[1] == sequence_dir.name]
        frame_filenames = set(
            sequence_data[2].apply(lambda x: f"{int(x):010d}.png").tolist()
        )
        for cam in cameras:
            cam_dir = sequence_dir / cam / "data_rect"
        if sequence_dir.is_dir():
            center_crop_by_height(
                cam_dir, output_dir / sequence_dir.name / cam, frame_filenames
            )


if __name__ == "__main__":
    input_dir = Path("~/data/KITTI-360/data_2d_raw").expanduser()
    output_dir = Path(
        "~/data/KITTI-360_proc/val/data_2d_raw_center_cropped"
    ).expanduser()
    data_txt_path = Path("preprocessing/data.txt")
    data = pd.read_csv(data_txt_path, sep="\s+", header=None)
    val_data = data[data[3] == "val"]
    center_crop_raw(input_dir, output_dir, val_data)
