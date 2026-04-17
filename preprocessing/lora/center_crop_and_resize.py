from pathlib import Path

import cv2
import pandas as pd


def center_crop_and_resize_from_csv(
    input_dir: Path, output_dir: Path, ds_csv_path: str, output_size: int = 512
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ds_df = pd.read_csv(ds_csv_path)
    for _, row in ds_df.iterrows():
        sequence = row["sequence"]
        frame_filename = row["frame_filename"]
        input_path = input_dir / sequence / "image_00" / "data_rect" / frame_filename
        out_frame_filename = f"{sequence}_{frame_filename}"
        output_path = output_dir / out_frame_filename

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

        cropped = cv2.resize(
            cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR
        )
        cv2.imwrite(str(output_path), cropped)
        print(f"Processed image: {output_path}")


if __name__ == "__main__":
    input_dir = Path("~/data/KITTI-360/data_2d_raw").expanduser()
    output_dir = Path("~/data/KITTI-360_proc/lora/center_cropped").expanduser()
    ds_csv_path = "data/lora/lora_ds.csv"
    output_size = 512
    print(
        f"Center-cropping images from {input_dir} and saving to {output_dir} with output size {output_size}x{output_size}"
    )
    center_crop_and_resize_from_csv(input_dir, output_dir, ds_csv_path, output_size)
