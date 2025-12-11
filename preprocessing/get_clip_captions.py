import json
from pathlib import Path


def get_clip_captions(split_dir: Path, prompt: str):
    captions = {}
    raw_dir = split_dir / "raw_input"
    for clip_dir in raw_dir.iterdir():
        if clip_dir.is_dir():
            clip_name = clip_dir.name
            captions[clip_name] = prompt

    # Save to a json file
    captions_path = split_dir / "captions.json"
    with open(captions_path, "w") as f:
        json.dump(captions, f, indent=4)
    captions_multi_path = split_dir / "captions_multi.json"
    with open(captions_multi_path, "w") as f:
        json.dump(captions, f, indent=4)


if __name__ == "__main__":
    data_dir = Path("~/data/KITTI-360_proc").expanduser()
    split_dir = data_dir / "val"
    num_frames_per_clip = 16
    clip_split_dir = data_dir / f"clips_{num_frames_per_clip}_flat" / "val"
    prompt = "a driving scene in a town, photorealistic, sunny weather"

    get_clip_captions(clip_split_dir, prompt)
