import argparse
from pathlib import Path

import cv2
import pandas as pd
import yaml

import generate_sd15
import warp_and_inpaint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vkitti_2_path", type=str, default="~/data/VKITTI-2_processed/"
    )
    parser.add_argument(
        "--eval_dataset_csv_path", type=str, default="data/vkitti_2_eval_dataset.csv"
    )
    parser.add_argument("--diffusion_img_width", type=int, default=512)
    parser.add_argument("--diffusion_img_height", type=int, default=512)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A driving scene in a town, photorealistic, clear daylight, blue sky, highly detailed",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bad quality, worst quality, cartoon style, unrealistic, blurry",
    )
    parser.add_argument("--gen_num_inference_steps", type=int, default=50)
    parser.add_argument("--gen_depth_cond_scale", type=float, default=1.0)
    parser.add_argument("--use_segmentation_for_generation", action="store_true")
    parser.add_argument("--gen_segmentation_cond_scale", type=float, default=1.0)
    parser.add_argument(
        "--output_root_dir", type=str, default="~/data/VKITTI-2_output/ControlNet_v11"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="ControlNet_v11_depth_and_segmentation"
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main(args):
    vkitti_2_path = Path(args.vkitti_2_path).expanduser()
    eval_dataset_csv_path = Path(args.eval_dataset_csv_path)
    output_root_dir = Path(args.output_root_dir).expanduser()

    output_dir = output_root_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_frames_dir = output_dir / "output_frames"
    output_frames_dir.mkdir(exist_ok=True)

    output_videos_dir = output_dir / "output_videos"
    output_videos_dir.mkdir(exist_ok=True)

    if args.debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

    args_out_path = output_dir / "args.yaml"
    with open(args_out_path, "w") as f:
        yaml.dump(vars(args), f)

    # Save conditions in VBench config for later processing
    conditions = ["depth"]
    if args.use_segmentation_for_generation:
        conditions.append("segmentation")
    vbench_config = {"conditions": conditions}
    vbench_config["prompt"] = args.prompt
    vbench_config["negative_prompt"] = args.negative_prompt
    config_output_path = output_dir / "vbench_config.yaml"
    with open(config_output_path, "w") as file:
        yaml.dump(vbench_config, file)

    gen_pipeline = generate_sd15.init_generation_pipeline(
        args.use_segmentation_for_generation
    )

    depth_dir = vkitti_2_path / "depth"
    seg_dir = vkitti_2_path / "segmentation"

    eval_dataset_df = pd.read_csv(eval_dataset_csv_path)
    for idx, row in eval_dataset_df.iterrows():
        clip_id = row["clip_id"]
        scene_id = row["scene_id"]
        start_frame_id = row["start_frame_id"]
        end_frame_id = row["end_frame_id"]
        print(
            f"Clip: {clip_id}, Scene: {scene_id}, Frames: {start_frame_id} - {end_frame_id}"
        )

        frame_ids = list(range(start_frame_id, end_frame_id + 1))
        print(f"Frame IDs: {frame_ids}")

        clip_dir_name = f"Scene{scene_id:02d}_clone_Camera_0_clip_{clip_id:04d}"
        depth_clip_dir = depth_dir / clip_dir_name
        seg_clip_dir = seg_dir / clip_dir_name

        output_frames_clip_dir = output_frames_dir / clip_dir_name
        output_frames_clip_dir.mkdir(exist_ok=True)

        # Save output video
        output_video_path = output_videos_dir / f"{clip_dir_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            args.fps,
            (args.diffusion_img_width, args.diffusion_img_height),
        )

        if args.debug:
            debug_clip_dir = debug_dir / clip_dir_name
            debug_clip_dir.mkdir(exist_ok=True)
            debug_clip_depth_dir = debug_clip_dir / "depth"
            debug_clip_depth_dir.mkdir(exist_ok=True)
            if args.use_segmentation_for_generation:
                debug_clip_seg_dir = debug_clip_dir / "segmentation"
                debug_clip_seg_dir.mkdir(exist_ok=True)

        for frame_id in frame_ids:
            print(f"Processing frame {frame_id}...")
            depth_path = depth_clip_dir / f"{frame_id:05d}.png"
            seg_path = seg_clip_dir / f"{frame_id:05d}.png"

            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
            if args.debug:
                debug_depth_out_path = debug_clip_depth_dir / f"{frame_id:05d}.png"
                cv2.imwrite(str(debug_depth_out_path), depth_img)
            seg_img = None
            if args.use_segmentation_for_generation:
                seg_img = cv2.imread(str(seg_path))
                seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
                if args.debug:
                    debug_seg_out_path = debug_clip_seg_dir / f"{frame_id:05d}.png"
                    cv2.imwrite(
                        str(debug_seg_out_path),
                        cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR),
                    )

            generated_img = warp_and_inpaint.generate_first_frame(
                args, gen_pipeline, depth_img, seg_img
            )

            video_writer.write(generated_img)

            output_frame_path = output_frames_clip_dir / f"{frame_id:05d}.png"
            cv2.imwrite(str(output_frame_path), generated_img)

        video_writer.release()
        print(f"Saved video: {output_video_path}")
        print(f"Saved frames: {output_frames_clip_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
