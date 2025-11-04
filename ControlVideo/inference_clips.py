import os
from dataclasses import dataclass, field
from typing import List, Optional

import inference
import yaml


@dataclass
class VideoInferenceConfig:
    prompt: str = "a driving scene in a town, photorealistic, sunny weather"
    condition: str = "depth"
    video_length: int = 16
    height: int = 512
    width: int = 512
    smoother_steps: List[int] = field(default_factory=lambda: [19, 20])
    is_long_video: bool = False
    version: str = "v11"
    frame_rate: Optional[int] = 1
    num_inference_steps: int = 50
    src_dir: Optional[str] = None
    output_dir: Optional[str] = "./outputs"
    use_processor: bool = False

    def __post_init__(self):
        if self.src_dir is None:
            self.src_dir = os.path.expanduser(
                f"~/data/KITTI-360_proc/clips_{self.video_length}_flat/val"
            )


def merge_cfg_into_parser(parser, cfg: VideoInferenceConfig):
    parser.set_defaults(**cfg.__dict__)


def process_clips(args):
    pipe, generator, processor = inference.load_model(args)

    raw_mp4_dir = os.path.join(args.src_dir, "raw_input_mp4")
    for clip_mp4_name in sorted(os.listdir(raw_mp4_dir)):
        clip_mp4 = os.path.join(raw_mp4_dir, clip_mp4_name)
        args.video_path = clip_mp4
        clip_mp4_stem = os.path.splitext(clip_mp4_name)[0]
        args.output_path = os.path.join(args.output_dir, clip_mp4_stem)
        os.makedirs(args.output_path, exist_ok=True)
        if not args.use_processor:
            args.condition_dir = os.path.join(
                args.src_dir, args.condition, clip_mp4_stem
            )

        inference.process_video(args, pipe, generator, processor)
        print(f"Processed clip: {clip_mp4_name}")


if __name__ == "__main__":
    parser = inference.get_parser()
    cfg = VideoInferenceConfig()
    merge_cfg_into_parser(parser, cfg)
    parser.add_argument(
        "--src_dir", type=str, default=cfg.src_dir, help="Directory of source clips"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=cfg.output_dir,
        help="Directory of output clips",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "inference_args.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    process_clips(args)
