import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import vkitti_2_to_ade20k
import warp_and_inpaint
from warping import warp_frames_vkitti_2


def main():
    vkitti_2_path = Path("~/data/VKITTI-2/").expanduser()
    variation = "clone"
    cam_id = 0
    orig_shape = (1242, 375)  # w, h
    diffusion_img_shape = (512, 512)  # w, h
    min_gen_depth = 0.1
    max_gen_depth = 655.35
    vkitti_2_out_path = Path("~/data/VKITTI-2_processed/").expanduser()
    mp4_fps = 4

    eval_dataset_csv_path = Path("data/vkitti_2_eval_dataset.csv")
    eval_dataset_df = pd.read_csv(eval_dataset_csv_path)

    rgb_out_dir = vkitti_2_out_path / "raw_input"
    rgb_mp4_out_dir = vkitti_2_out_path / "raw_input_mp4"
    depth_out_dir = vkitti_2_out_path / "depth"
    depth_raw_out_dir = vkitti_2_out_path / "depth_raw"
    seg_out_dir = vkitti_2_out_path / "segmentation"
    textgt_out_dir = vkitti_2_out_path / "vkitti_2.0.3_textgt"
    rgb_out_dir.mkdir(parents=True, exist_ok=True)
    rgb_mp4_out_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)
    depth_raw_out_dir.mkdir(parents=True, exist_ok=True)
    seg_out_dir.mkdir(parents=True, exist_ok=True)
    textgt_out_dir.mkdir(parents=True, exist_ok=True)

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

        cam = warp_frames_vkitti_2.CameraPerspective(
            vkitti_2_path, scene_id, variation, cam_id
        )
        cam, s, offset_x = warp_and_inpaint.update_K(
            cam, orig_shape, diffusion_img_shape
        )

        clip_name = (
            f"Scene{scene_id:02d}_{variation}_Camera_{cam_id}_clip_{clip_id:04d}"
        )

        rgb_clip_out_dir = rgb_out_dir / clip_name
        depth_clip_out_dir = depth_out_dir / clip_name
        depth_raw_clip_out_dir = depth_raw_out_dir / clip_name
        seg_clip_out_dir = seg_out_dir / clip_name

        rgb_clip_out_dir.mkdir(parents=True, exist_ok=True)
        depth_clip_out_dir.mkdir(parents=True, exist_ok=True)
        depth_raw_clip_out_dir.mkdir(parents=True, exist_ok=True)
        seg_clip_out_dir.mkdir(parents=True, exist_ok=True)

        # Write frames into an mp4
        clip_mp4_out_path = rgb_mp4_out_dir / f"{clip_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(clip_mp4_out_path),
            fourcc,
            mp4_fps,
            diffusion_img_shape,
        )

        for frame_id in frame_ids:
            rgb_path = (
                vkitti_2_path
                / f"vkitti_2.0.3_rgb/Scene{scene_id:02d}/{variation}/frames/rgb/Camera_{cam_id}/rgb_{frame_id:05d}.jpg"
            )
            depth_dir = (
                vkitti_2_path
                / f"vkitti_2.0.3_depth/Scene{scene_id:02d}/{variation}/frames/depth/Camera_{cam_id}"
            )
            seg_dir = (
                vkitti_2_path
                / f"vkitti_2.0.3_classSegmentation/Scene{scene_id:02d}/{variation}/frames/classSegmentation/Camera_{cam_id}"
            )

            rgb_img = cv2.imread(str(rgb_path))
            rgb_img_resized = warp_and_inpaint.get_resized_and_cropped_image(
                rgb_img,
                s,
                offset_x,
                diffusion_img_shape,
                interpolation=cv2.INTER_NEAREST,
            )
            rgb_out_path = rgb_clip_out_dir / f"{frame_id:05d}.jpg"
            cv2.imwrite(str(rgb_out_path), rgb_img_resized)

            video_writer.write(rgb_img_resized)

            depth = warp_and_inpaint.get_depth(depth_dir, frame_id)
            depth_clipped = np.clip(depth, min_gen_depth, max_gen_depth)
            depth_resized = warp_and_inpaint.get_resized_and_cropped_image(
                depth_clipped,
                s,
                offset_x,
                diffusion_img_shape,
                interpolation=cv2.INTER_NEAREST,
            )
            norm_depth_img = warp_and_inpaint.get_normalized_depth_image(depth_resized)
            depth_out_path = depth_clip_out_dir / f"{frame_id:05d}.png"
            cv2.imwrite(str(depth_out_path), norm_depth_img)

            depth_raw_path = depth_dir / f"depth_{frame_id:05d}.png"
            depth_raw_out_path = depth_raw_clip_out_dir / f"depth_{frame_id:05d}.png"
            shutil.copy(str(depth_raw_path), str(depth_raw_out_path))

            seg_img = warp_and_inpaint.get_segmentation_map(seg_dir, frame_id)
            seg_img_resized = warp_and_inpaint.get_resized_and_cropped_image(
                seg_img,
                s,
                offset_x,
                diffusion_img_shape,
                interpolation=cv2.INTER_NEAREST,
            )
            seg_img_ade20k = vkitti_2_to_ade20k.map_vkitti2_to_ade20k(
                seg_img_resized,
            )
            seg_img_ade20k = cv2.cvtColor(seg_img_ade20k, cv2.COLOR_RGB2BGR)
            seg_out_path = seg_clip_out_dir / f"{frame_id:05d}.png"
            cv2.imwrite(str(seg_out_path), seg_img_ade20k)

        video_writer.release()

        print(f"Processed clip {clip_id} of scene {scene_id}")

    unique_scenes = eval_dataset_df["scene_id"].unique()
    for scene_id in unique_scenes:
        textgt_scene_dir = (
            vkitti_2_path / f"vkitti_2.0.3_textgt/Scene{scene_id:02d}/{variation}"
        )
        textgt_scene_out_dir = textgt_out_dir / f"Scene{scene_id:02d}/{variation}"
        textgt_scene_out_dir.mkdir(parents=True, exist_ok=True)
        intrinsic_path = textgt_scene_dir / "intrinsic.txt"
        extrinsic_path = textgt_scene_dir / "extrinsic.txt"
        intrinsic_out_path = textgt_scene_out_dir / "intrinsic.txt"
        extrinsic_out_path = textgt_scene_out_dir / "extrinsic.txt"
        shutil.copy(str(intrinsic_path), str(intrinsic_out_path))
        shutil.copy(str(extrinsic_path), str(extrinsic_out_path))

        print(f"Copied textgt files for scene {scene_id}")


if __name__ == "__main__":
    main()
