import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

import generate_sd15
import warp_and_inpaint
from warping import warp_frames_vkitti_2


def get_preprocessed_control_maps(
    args,
    vkitti_2_path,
    clip_dir_name,
    start_frame_id,
    end_frame_id,
    s,
    offset_x,
    diffusion_img_shape,
):
    depth_dir = vkitti_2_path / "depth"
    depth_raw_dir = vkitti_2_path / "depth_raw"
    seg_dir = vkitti_2_path / "segmentation"

    depth_clip_dir = depth_dir / clip_dir_name
    depth_raw_clip_dir = depth_raw_dir / clip_dir_name
    seg_clip_dir = seg_dir / clip_dir_name

    depths_raw = {}
    depth_maps = {}
    use_segmentation = (
        args.use_segmentation_for_generation or args.use_segmentation_for_inpainting
    )
    seg_maps = None
    if use_segmentation:
        seg_maps = {}

    for frame_id in range(start_frame_id, end_frame_id + 1):
        depth_raw = warp_and_inpaint.get_depth(depth_raw_clip_dir, frame_id)
        depth_raw_resized = warp_and_inpaint.get_resized_and_cropped_image(
            depth_raw,
            s,
            offset_x,
            diffusion_img_shape,
            interpolation=cv2.INTER_NEAREST,
        )

        depth_map = cv2.imread(
            str(depth_clip_dir / f"{frame_id:05d}.png"), cv2.IMREAD_GRAYSCALE
        )
        depths_raw[frame_id] = depth_raw_resized
        depth_maps[frame_id] = depth_map

        seg_map = cv2.imread(str(seg_clip_dir / f"{frame_id:05d}.png"))
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_BGR2RGB)
        if use_segmentation:
            seg_maps[frame_id] = seg_map

    return depths_raw, depth_maps, seg_maps


def parse_args():
    base_parser = warp_and_inpaint.get_base_parser()
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument(
        "--eval_dataset_csv_path", type=str, default="data/vkitti_2_eval_dataset.csv"
    )
    parser.add_argument("--rel_frame_ids", type=int, nargs="+", default=[0, 8, 15])
    parser.set_defaults(vkitti_2_path="~/data/VKITTI-2_processed/")
    parser.set_defaults(experiment_name="warp_and_inpaint_full")
    args = parser.parse_args()
    return args


def main(args):
    vkitti_2_path = Path(args.vkitti_2_path).expanduser()
    eval_dataset_csv_path = Path(args.eval_dataset_csv_path)
    output_root_dir = Path(args.output_root_dir).expanduser()

    orig_shape = (args.orig_width, args.orig_height)
    diffusion_img_shape = (args.diffusion_img_width, args.diffusion_img_height)

    output_dir = output_root_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_frames_dir = output_dir / "output_frames"
    output_frames_dir.mkdir(exist_ok=True)

    output_videos_dir = output_dir / "output_videos"
    output_videos_dir.mkdir(exist_ok=True)

    if args.debug:
        debug_output_dir = output_dir / "debug"
        debug_output_dir.mkdir(exist_ok=True)

    args_output_path = output_dir / "args.yaml"
    with open(args_output_path, "w") as file:
        yaml.dump(vars(args), file)

    # Save conditions in VBench config for later processing
    conditions = ["depth"]
    if args.use_segmentation_for_generation or args.use_segmentation_for_inpainting:
        conditions.append("segmentation")
    vbench_config = {"conditions": conditions}
    vbench_config["prompt"] = args.prompt
    vbench_config["negative_prompt"] = args.negative_prompt
    config_output_path = output_dir / "vbench_config.yaml"
    with open(config_output_path, "w") as file:
        yaml.dump(vbench_config, file)

    gen_pipeline = generate_sd15.init_generation_pipeline(
        use_segmentation=args.use_segmentation_for_generation,
        lora_weights_path=args.lora_weights_path,
    )

    inpaint_pipeline = generate_sd15.init_inpainting_pipeline(
        args.use_depth_for_inpainting,
        args.use_segmentation_for_inpainting,
        lora_weights_path=args.lora_weights_path,
    )

    eval_dataset_df = pd.read_csv(eval_dataset_csv_path)
    for idx, row in eval_dataset_df.iterrows():
        clip_id = row["clip_id"]
        scene_id = row["scene_id"]
        start_frame_id = row["start_frame_id"]
        end_frame_id = row["end_frame_id"]

        frame_ids = [start_frame_id + rel_id for rel_id in args.rel_frame_ids]

        print(
            f"Clip: {clip_id}, Scene: {scene_id}, Keyframes: {[int(x) for x in frame_ids]}"
        )

        clip_dir_name = f"Scene{scene_id:02d}_{args.variation}_Camera_{args.cam_id}_clip_{clip_id:04d}"

        cam = warp_frames_vkitti_2.CameraPerspective(
            vkitti_2_path, scene_id, args.variation, args.cam_id
        )
        # Update intrinsic matrix K for warping in diffusion image coordinates
        cam, s, offset_x = warp_and_inpaint.update_K(
            cam, orig_shape, diffusion_img_shape
        )

        depths_raw, depth_maps, seg_maps = get_preprocessed_control_maps(
            args,
            vkitti_2_path,
            clip_dir_name,
            start_frame_id,
            end_frame_id,
            s,
            offset_x,
            diffusion_img_shape,
        )

        src_frame_id = frame_ids[0]

        if args.generate_first_frame:
            src_frame = warp_and_inpaint.generate_first_frame(
                args,
                gen_pipeline,
                depth_maps[src_frame_id],
                (
                    seg_maps[src_frame_id]
                    if args.use_segmentation_for_generation
                    else None
                ),
            )
        else:
            src_frame_path = (
                vkitti_2_path / "rgb" / clip_dir_name / f"rgb_{src_frame_id:05d}.jpg"
            )
            src_frame = cv2.imread(str(src_frame_path))

        # Save debug output
        debug_clip_dir = None
        if args.debug:
            debug_clip_dir = debug_output_dir / clip_dir_name
            debug_clip_dir.mkdir(exist_ok=True)
            if args.generate_first_frame:
                norm_depth_out_path = (
                    debug_clip_dir / f"norm_depth_{src_frame_id:05d}.png"
                )
                cv2.imwrite(str(norm_depth_out_path), depth_maps[src_frame_id])

                if args.use_segmentation_for_generation:
                    seg_src_out_path = debug_clip_dir / f"seg_{src_frame_id:05d}.png"
                    cv2.imwrite(
                        str(seg_src_out_path),
                        cv2.cvtColor(seg_maps[src_frame_id], cv2.COLOR_RGB2BGR),
                    )

            src_frame_out_path = debug_clip_dir / f"{src_frame_id:05d}.png"
            cv2.imwrite(str(src_frame_out_path), src_frame)

        clip_output_frames = warp_and_inpaint.warp_and_inpaint_sequence(
            args,
            cam,
            src_frame,
            frame_ids,
            inpaint_pipeline,
            depths_raw,
            depth_maps,
            seg_maps,
            debug_clip_dir,
        )

        # Save output frames and write to video
        output_frames_clip_dir = output_frames_dir / clip_dir_name
        output_frames_clip_dir.mkdir(exist_ok=True)

        output_video_path = output_videos_dir / f"{clip_dir_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            args.fps,
            diffusion_img_shape,
        )

        frame_ids_full = np.arange(start_frame_id, end_frame_id + 1)
        for id, frame in zip(frame_ids_full, clip_output_frames):
            frame_out_path = output_frames_clip_dir / f"{id:05d}.png"
            cv2.imwrite(str(frame_out_path), frame)

            video_writer.write(frame)

        video_writer.release()


if __name__ == "__main__":
    args = parse_args()
    main(args)
