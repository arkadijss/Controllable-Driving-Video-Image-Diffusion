import pandas as pd
from pathlib import Path


def make_clips_csv(data_txt_path, num_frames, output_dir=Path.cwd(), max_gap=20):
    df = pd.read_csv(data_txt_path, sep=r"\s+", header=None)
    df = df[[1, 2, 3]]
    df.columns = ["sequence", "frame", "split"]
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")

    clip_rows = []
    total_clips = {"train": 0, "val": 0}

    for split in ["train", "val"]:
        split_df = df[df["split"] == split]

        for seq in split_df["sequence"].unique():
            seq_df = split_df[split_df["sequence"] == seq].sort_values("frame")
            frames = seq_df["frame"].tolist()

            i = 0
            while i + num_frames <= len(frames):
                clip_frames = frames[i : i + num_frames]

                # Check if any gap exceeds max_gap
                gap_too_large = any(
                    clip_frames[j + 1] - clip_frames[j] > max_gap
                    for j in range(len(clip_frames) - 1)
                )

                if not gap_too_large:
                    clip_rows.append(
                        {
                            "split": split,
                            "sequence": seq,
                            "clip_num": total_clips[split],
                            "frames": ";".join(map(str, clip_frames)),
                        }
                    )
                    total_clips[split] += 1
                    i += num_frames
                else:
                    # Skip to the first frame after the large gap
                    for j in range(len(clip_frames) - 1):
                        if clip_frames[j + 1] - clip_frames[j] > max_gap:
                            i += j + 1
                            break

    clips_df = pd.DataFrame(clip_rows)

    base_name = data_txt_path.stem
    output_csv = output_dir / f"{base_name}_num_frames_{num_frames}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_df.to_csv(output_csv, index=False)

    print(f"Total train clips: {total_clips['train']}")
    print(f"Total val clips: {total_clips['val']}")
    print(f"Saved clip table to {output_csv}")


if __name__ == "__main__":
    data_txt_path = Path("data/data.txt")
    num_frames = 16
    output_dir = Path(f"~/data/KITTI-360_proc/clips_{num_frames}_flat").expanduser()

    make_clips_csv(data_txt_path, num_frames, output_dir)
