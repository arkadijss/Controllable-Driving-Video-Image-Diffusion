from pathlib import Path
import pandas as pd
import numpy as np


def get_lora_ds_samples(input_dir: Path, num_frames_dict: dict, output_csv_path: str):
    records = []
    for sequence_dir in input_dir.iterdir():
        cam_dir = sequence_dir / "image_00" / "data_rect"
        frame_filenames = sorted(
            [f.name for f in cam_dir.iterdir() if f.suffix == ".png"]
        )
        num_frames = num_frames_dict[sequence_dir.name]
        # Uniformly sample num_frames from frame_filenames
        sampled_frames = np.linspace(0, len(frame_filenames) - 1, num_frames, dtype=int)
        sampled_frame_filenames = [frame_filenames[i] for i in sampled_frames]
        for frame_filename in sampled_frame_filenames:
            frame_id = int(frame_filename.split(".")[0])
            records.append(
                {
                    "sequence": sequence_dir.name,
                    "frame_filename": frame_filename,
                    "frame_id": frame_id,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    input_dir = Path("~/data/KITTI-360/data_2d_raw").expanduser()
    num_frames_dict = {
        "2013_05_28_drive_0000_sync": 556,
        "2013_05_28_drive_0002_sync": 556,
        "2013_05_28_drive_0003_sync": 556,
        "2013_05_28_drive_0004_sync": 556,
        "2013_05_28_drive_0005_sync": 556,
        "2013_05_28_drive_0006_sync": 555,
        "2013_05_28_drive_0007_sync": 555,
        "2013_05_28_drive_0009_sync": 555,
        "2013_05_28_drive_0010_sync": 555,
    }
    # Assert total num frames is 5000
    total_frames = sum([num_frames for _, num_frames in num_frames_dict.items()])
    assert (
        total_frames == 5000
    ), f"Total number of frames is {total_frames}, expected 5000"
    output_csv_path = "data/lora/lora_ds.csv"
    get_lora_ds_samples(input_dir, num_frames_dict, output_csv_path)
