import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

import generate_sd15
import vkitti_2_to_ade20k
from warping import warp_frames_vkitti_2, warping_utils


def get_depth(depth_dir, frame_id):
    depth_path = depth_dir / f"depth_{frame_id:05d}.png"
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = depth_raw.astype(np.float32)
    # Divide to get meters
    depth /= 100.0
    return depth


def get_segmentation_map(seg_dir, frame_id):
    seg_path = seg_dir / f"classgt_{frame_id:05d}.png"
    seg_img = cv2.imread(str(seg_path))
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    return seg_img


def update_K(cam, orig_shape, tgt_shape):
    orig_w, orig_h = orig_shape
    tgt_w, tgt_h = tgt_shape
    s = tgt_h / orig_h
    cam.K[0, 0] *= s
    cam.K[1, 1] *= s
    new_w = orig_w * s
    offset_x = (new_w - tgt_w) / 2
    cam.K[0, 2] = (cam.K[0, 2] * s) - offset_x
    cam.K[1, 2] *= s
    return cam, s, offset_x


def get_resized_and_cropped_image(
    image, s, offset_x, tgt_shape, interpolation=cv2.INTER_NEAREST
):
    h, w = image.shape[:2]
    tgt_w, tgt_h = tgt_shape
    new_w = round(w * s)
    image_resized = cv2.resize(image, (new_w, tgt_h), interpolation=interpolation)
    offset_x = round(offset_x)
    image_cropped = image_resized[:, offset_x : offset_x + tgt_w]
    return image_cropped


def get_normalized_depth_image(depth_metric):
    disparity = 1.0 / depth_metric

    d_min = disparity.min()
    d_max = disparity.max()
    norm_disparity = (disparity - d_min) / (d_max - d_min)

    norm_depth_image = (norm_disparity * 255).astype(np.uint8)

    return norm_depth_image


def inpaint_image_classical(image, mask, opening_kernel_size=3, inpaint_kernel_size=3):
    kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_diff = cv2.absdiff(mask, opening)
    inpainted_image = cv2.inpaint(
        image, mask_diff, inpaint_kernel_size, cv2.INPAINT_TELEA
    )
    return inpainted_image, mask_diff, opening


def interpolate_frames(
    src_frame,
    tgt_frame,
    src_frame_id,
    tgt_frame_id,
    depth_src,
    depth_tgt,
    interp_ids,
    interp_depths,
    cam,
    depth_thr,
    diffusion_img_shape,
    inpaint_kernel_size=3,
):
    interpolated_frames = []
    for i, frame_id in enumerate(interp_ids):
        depth_interp = interp_depths[i]
        interp_frame_src, interp_missing_mask_src = warping_utils.warp_frame(
            src_frame,
            cam,
            src_frame_id,
            frame_id,
            depth_src,
            depth_interp,
            depth_thr,
        )
        interp_frame_tgt, interp_missing_mask_tgt = warping_utils.warp_frame(
            tgt_frame,
            cam,
            tgt_frame_id,
            frame_id,
            depth_tgt,
            depth_interp,
            depth_thr,
        )

        interp_frame = np.zeros_like(interp_frame_src)
        valid_src = interp_missing_mask_src == 0
        valid_tgt = interp_missing_mask_tgt == 0
        valid_both = valid_src & valid_tgt
        alpha = (frame_id - src_frame_id) / (tgt_frame_id - src_frame_id)
        interp_frame[valid_both] = (1 - alpha) * interp_frame_src[
            valid_both
        ] + alpha * interp_frame_tgt[valid_both]
        interp_frame[valid_src & ~valid_tgt] = interp_frame_src[valid_src & ~valid_tgt]
        interp_frame[~valid_src & valid_tgt] = interp_frame_tgt[~valid_src & valid_tgt]

        # Inpaint the remaining missing regions using classical inpainting
        interp_missing_mask = (
            interp_missing_mask_src * interp_missing_mask_tgt
        ).astype(np.uint8) * 255
        interp_frame_inpainted = cv2.inpaint(
            interp_frame, interp_missing_mask, inpaint_kernel_size, cv2.INPAINT_TELEA
        )
        interp_frame_inpainted_resized = cv2.resize(
            interp_frame_inpainted, diffusion_img_shape
        )
        interpolated_frames.append(interp_frame_inpainted_resized)

    return interpolated_frames


def get_preprocessed_control_maps(
    args, vkitti_2_path, s, offset_x, diffusion_img_shape
):
    rel_frames_dir = Path(f"Scene{args.location:02d}") / args.variation / "frames"
    depth_dir = (
        vkitti_2_path
        / "vkitti_2.0.3_depth"
        / rel_frames_dir
        / "depth"
        / f"Camera_{args.cam_id}"
    )
    seg_dir = (
        vkitti_2_path
        / "vkitti_2.0.3_classSegmentation"
        / rel_frames_dir
        / "classSegmentation"
        / f"Camera_{args.cam_id}"
    )

    depths_raw = {}
    depth_maps = {}
    use_segmentation = (
        args.use_segmentation_for_generation or args.use_segmentation_for_inpainting
    )
    seg_maps = {} if use_segmentation else None
    for frame_id in range(min(args.frame_ids), max(args.frame_ids) + 1):
        # Get depth for warping
        depth_raw = get_depth(depth_dir, frame_id)
        depth_raw_resized = get_resized_and_cropped_image(
            depth_raw,
            s,
            offset_x,
            diffusion_img_shape,
            interpolation=cv2.INTER_NEAREST,
        )
        # Get depth maps for ControlNet
        depth_clipped = np.clip(depth_raw, args.min_gen_depth, args.max_gen_depth)
        depth_resized = get_resized_and_cropped_image(
            depth_clipped,
            s,
            offset_x,
            diffusion_img_shape,
            interpolation=cv2.INTER_NEAREST,
        )
        norm_depth_image = get_normalized_depth_image(depth_resized)
        depths_raw[frame_id] = depth_raw_resized
        depth_maps[frame_id] = norm_depth_image

        if not use_segmentation:
            continue

        seg_map = get_segmentation_map(seg_dir, frame_id)
        seg_map_resized = get_resized_and_cropped_image(
            seg_map,
            s,
            offset_x,
            diffusion_img_shape,
            interpolation=cv2.INTER_NEAREST,
        )
        seg_map_ade20k = vkitti_2_to_ade20k.map_vkitti2_to_ade20k(
            seg_map_resized,
        )
        seg_maps[frame_id] = seg_map_ade20k

    return depths_raw, depth_maps, seg_maps


def generate_first_frame(args, gen_pipeline, norm_depth_image, seg_src_resized=None):
    if args.use_segmentation_for_generation:
        assert (
            seg_src_resized is not None
        ), "Segmentation map must be provided if using segmentation for generation"
        seg_image = Image.fromarray(seg_src_resized)

    depth_image = generate_sd15.preprocess_depth_image(norm_depth_image)

    gen_kwargs = {
        "num_inference_steps": args.gen_num_inference_steps,
        "guidance_scale": args.gen_guidance_scale,
    }
    gen_control_image = [depth_image]
    gen_cond_scale = [args.gen_depth_cond_scale]

    if args.use_segmentation_for_generation:
        gen_control_image.append(seg_image)
        gen_cond_scale.append(args.gen_segmentation_cond_scale)

    if len(gen_control_image) == 1:
        gen_control_image = gen_control_image[0]
        gen_cond_scale = gen_cond_scale[0]

    gen_kwargs["controlnet_conditioning_scale"] = gen_cond_scale

    src_frame_gen = generate_sd15.generate_image(
        gen_pipeline,
        args.prompt,
        args.negative_prompt,
        gen_control_image,
        seed=args.seed,
        **gen_kwargs,
    )
    src_frame = cv2.cvtColor(np.array(src_frame_gen), cv2.COLOR_RGB2BGR)

    return src_frame


def warp_and_inpaint_sequence(
    args,
    cam,
    src_frame,
    frame_ids,
    inpaint_pipeline,
    depths_raw,
    depth_maps,
    seg_maps=None,
    debug_dir=None,
):
    if args.use_segmentation_for_inpainting:
        assert (
            seg_maps is not None
        ), "Segmentation maps must be provided if using segmentation for inpainting"

    diffusion_img_shape = (args.diffusion_img_width, args.diffusion_img_height)

    src_frame_id = frame_ids[0]
    depth_src = depths_raw[src_frame_id]
    output_frames = [src_frame]
    for tgt_frame_id in frame_ids[1:]:
        depth_tgt = depths_raw[tgt_frame_id]

        src_frame_warped, warp_missing_mask = warping_utils.warp_frame(
            src_frame,
            cam,
            src_frame_id,
            tgt_frame_id,
            depth_src,
            depth_tgt,
            args.depth_thr,
        )

        warp_missing_mask = warp_missing_mask.astype(np.uint8) * 255

        src_frame_inpainted_classical, _, opening = inpaint_image_classical(
            src_frame_warped,
            warp_missing_mask,
            opening_kernel_size=args.opening_kernel_size,
            inpaint_kernel_size=args.classical_inpaint_kernel_size,
        )

        if args.mask_closing_kernel_size > 0:
            closing_kernel = np.ones(
                (args.mask_closing_kernel_size, args.mask_closing_kernel_size), np.uint8
            )
            diffusion_inpainting_mask = cv2.morphologyEx(
                opening, cv2.MORPH_CLOSE, closing_kernel
            )
        else:
            diffusion_inpainting_mask = opening

        if args.prefill_missing_regions:
            src_frame_inpainted_classical = cv2.inpaint(
                src_frame_inpainted_classical,
                diffusion_inpainting_mask,
                args.prefill_kernel_size,
                cv2.INPAINT_TELEA,
            )

        if args.diffusion_mask_dilation_kernel_size > 0:
            diffusion_mask_dilation_kernel = np.ones(
                (
                    args.diffusion_mask_dilation_kernel_size,
                    args.diffusion_mask_dilation_kernel_size,
                ),
                np.uint8,
            )
            diffusion_inpainting_mask = cv2.dilate(
                diffusion_inpainting_mask, diffusion_mask_dilation_kernel, iterations=1
            )

        input_image = Image.fromarray(
            cv2.cvtColor(src_frame_inpainted_classical, cv2.COLOR_BGR2RGB)
        )
        mask_image = Image.fromarray(diffusion_inpainting_mask)

        inpaint_kwargs = {
            "num_inference_steps": args.inpaint_num_inference_steps,
            "guidance_scale": args.inpaint_guidance_scale,
        }
        inpaint_control_image = []
        inpaint_cond_scale = []
        if args.use_depth_for_inpainting:
            depth_inpaint_normalized = depth_maps[tgt_frame_id]
            depth_inpaint_image = generate_sd15.preprocess_depth_image(
                depth_inpaint_normalized
            )
            inpaint_control_image.append(depth_inpaint_image)
            inpaint_cond_scale.append(args.inpaint_depth_cond_scale)

        if args.use_segmentation_for_inpainting:
            seg_inpaint_resized = seg_maps[tgt_frame_id]
            seg_inpaint_image = Image.fromarray(seg_inpaint_resized)
            inpaint_control_image.append(seg_inpaint_image)
            inpaint_cond_scale.append(args.inpaint_segmentation_cond_scale)

        if len(inpaint_control_image) == 1:
            inpaint_control_image = inpaint_control_image[0]
            inpaint_cond_scale = inpaint_cond_scale[0]

        inpaint_kwargs["control_image"] = inpaint_control_image
        inpaint_kwargs["controlnet_conditioning_scale"] = inpaint_cond_scale

        src_frame_inpainted_diffusion = generate_sd15.inpaint_image(
            inpaint_pipeline,
            args.prompt,
            args.negative_prompt,
            input_image,
            mask_image,
            seed=args.seed,
            **inpaint_kwargs,
        )
        src_frame_inpainted_diffusion = cv2.cvtColor(
            np.array(src_frame_inpainted_diffusion), cv2.COLOR_RGB2BGR
        )

        # Prepare depths for interpolation
        step = 1 if tgt_frame_id > src_frame_id else -1
        interp_ids = np.arange(src_frame_id + step, tgt_frame_id, step)
        interp_depths = []
        for frame_id in interp_ids:
            depth_interp = depths_raw[frame_id]
            interp_depths.append(depth_interp)

        interpolated_frames = interpolate_frames(
            src_frame,
            src_frame_inpainted_diffusion,
            src_frame_id,
            tgt_frame_id,
            depth_src,
            depth_tgt,
            interp_ids,
            interp_depths,
            cam,
            args.depth_thr,
            diffusion_img_shape,
            inpaint_kernel_size=args.interp_inpaint_kernel_size,
        )
        output_frames.extend(interpolated_frames)

        # Save debug outputs
        if debug_dir is not None:
            output_pair_path = debug_dir / f"{src_frame_id:05d}_to_{tgt_frame_id:05d}"
            output_pair_path.mkdir(parents=True, exist_ok=True)

            warped_frame_out_path = output_pair_path / f"warped_{tgt_frame_id:05d}.png"
            cv2.imwrite(str(warped_frame_out_path), src_frame_warped)

            warp_missing_mask_out_path = (
                output_pair_path / f"warp_missing_mask_{tgt_frame_id:05d}.png"
            )
            cv2.imwrite(str(warp_missing_mask_out_path), warp_missing_mask)

            inpainted_classical_frame_out_path = (
                output_pair_path / f"inpainted_classical_{tgt_frame_id:05d}.png"
            )
            cv2.imwrite(
                str(inpainted_classical_frame_out_path), src_frame_inpainted_classical
            )

            masked_input_image_out_path = (
                output_pair_path / f"masked_input_{tgt_frame_id:05d}.png"
            )
            masked_input_image = src_frame_inpainted_classical.copy()
            masked_input_image[diffusion_inpainting_mask == 255] = [0, 0, 255]
            cv2.imwrite(str(masked_input_image_out_path), masked_input_image)

            if args.use_depth_for_inpainting:
                depth_inpaint_out_path = (
                    output_pair_path / f"depth_inpaint_{tgt_frame_id:05d}.png"
                )
                cv2.imwrite(str(depth_inpaint_out_path), depth_inpaint_normalized)

            if args.use_segmentation_for_inpainting:
                seg_inpaint_out_path = (
                    output_pair_path / f"seg_inpaint_{tgt_frame_id:05d}.png"
                )
                cv2.imwrite(
                    str(seg_inpaint_out_path),
                    cv2.cvtColor(seg_inpaint_resized, cv2.COLOR_RGB2BGR),
                )

            mask_image_out_path = (
                output_pair_path / f"diffusion_inpainting_mask_{tgt_frame_id:05d}.png"
            )
            cv2.imwrite(str(mask_image_out_path), diffusion_inpainting_mask)

            inpainted_diffusion_frame_out_path = (
                output_pair_path / f"inpainted_diffusion_{tgt_frame_id:05d}.png"
            )
            cv2.imwrite(
                str(inpainted_diffusion_frame_out_path), src_frame_inpainted_diffusion
            )

            interpolated_frame_out_dir = output_pair_path / "interpolated_frames"
            interpolated_frame_out_dir.mkdir(exist_ok=True)
            for id, interp_frame in zip(interp_ids, interpolated_frames):
                interp_frame_out_path = (
                    interpolated_frame_out_dir / f"interp_{id:05d}.png"
                )
                cv2.imwrite(str(interp_frame_out_path), interp_frame)

        output_frames.append(src_frame_inpainted_diffusion)

        src_frame = src_frame_inpainted_diffusion
        src_frame_id = tgt_frame_id
        depth_src = depth_tgt

    return output_frames


def get_base_parser():
    parser = argparse.ArgumentParser(add_help=False)
    # General
    parser.add_argument("--vkitti_2_path", type=str, default="~/data/VKITTI-2/")
    parser.add_argument("--location", type=int, default=1)
    parser.add_argument("--variation", type=str, default="clone")
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--orig_width", type=int, default=1242)
    parser.add_argument("--orig_height", type=int, default=375)
    parser.add_argument("--diffusion_img_width", type=int, default=512)
    parser.add_argument("--diffusion_img_height", type=int, default=512)
    parser.add_argument("--min_gen_depth", type=float, default=0.1)
    parser.add_argument("--max_gen_depth", type=float, default=655.35)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A driving scene, day, clear weather, best quality, highly detailed",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bad quality, cartoon style, unrealistic, blurry, low resolution",
    )
    parser.add_argument(
        "--output_root_dir", type=str, default="outputs/warp_and_inpaint_vkitti_2"
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lora_weights_path", type=str, default=None)

    # First frame generation
    parser.add_argument("--generate_first_frame", action="store_true")
    parser.add_argument("--gen_num_inference_steps", type=int, default=50)
    parser.add_argument("--gen_depth_cond_scale", type=float, default=1.0)
    parser.add_argument("--use_segmentation_for_generation", action="store_true")
    parser.add_argument("--gen_segmentation_cond_scale", type=float, default=1.0)
    parser.add_argument("--gen_guidance_scale", type=float, default=7.5)

    # Warping
    parser.add_argument("--depth_thr", type=float, default=0.3)

    # Classical inpainting
    parser.add_argument("--opening_kernel_size", type=int, default=3)
    parser.add_argument("--classical_inpaint_kernel_size", type=int, default=3)

    # Diffusion inpainting
    parser.add_argument("--mask_closing_kernel_size", type=int, default=7)
    parser.add_argument("--prefill_missing_regions", action="store_true")
    parser.add_argument("--prefill_kernel_size", type=int, default=7)
    parser.add_argument("--diffusion_mask_dilation_kernel_size", type=int, default=9)
    parser.add_argument("--use_depth_for_inpainting", action="store_true")
    parser.add_argument("--use_segmentation_for_inpainting", action="store_true")
    parser.add_argument("--inpaint_num_inference_steps", type=int, default=50)
    parser.add_argument("--inpaint_depth_cond_scale", type=float, default=0.5)
    parser.add_argument("--inpaint_segmentation_cond_scale", type=float, default=0.5)
    parser.add_argument("--inpaint_guidance_scale", type=float, default=7.5)

    # Interpolation
    parser.add_argument("--interp_inpaint_kernel_size", type=int, default=3)

    return parser


def parse_args():
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument("--frame_ids", type=int, nargs="+", default=[7, 12])
    args = parser.parse_args()
    if args.experiment_name is None:
        args.experiment_name = f"{args.frame_ids[0]:05d}_to_{args.frame_ids[-1]:05d}"
    return args


def main(args):
    vkitti_2_path = Path(args.vkitti_2_path).expanduser()
    cam = warp_frames_vkitti_2.CameraPerspective(
        vkitti_2_path, args.location, args.variation, args.cam_id
    )
    orig_shape = (args.orig_width, args.orig_height)
    diffusion_img_shape = (args.diffusion_img_width, args.diffusion_img_height)
    # Update intrinsic matrix K for warping in diffusion image coordinates
    cam, s, offset_x = update_K(cam, orig_shape, diffusion_img_shape)
    depths_raw, depth_maps, seg_maps = get_preprocessed_control_maps(
        args,
        vkitti_2_path,
        s,
        offset_x,
        diffusion_img_shape,
    )

    output_root_dir = Path(args.output_root_dir).expanduser()

    output_dir = output_root_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    src_frame_id = args.frame_ids[0]

    src_frame = None
    gen_pipeline = None
    if args.generate_first_frame:
        gen_pipeline = generate_sd15.init_generation_pipeline(
            use_segmentation=args.use_segmentation_for_generation,
            lora_weights_path=args.lora_weights_path,
        )
        src_frame = generate_first_frame(
            args,
            gen_pipeline,
            depth_maps[src_frame_id],
            seg_maps[src_frame_id] if args.use_segmentation_for_generation else None,
        )
        norm_depth_out_path = output_dir / f"norm_depth_{src_frame_id:05d}.png"
        cv2.imwrite(str(norm_depth_out_path), depth_maps[src_frame_id])
        if args.use_segmentation_for_generation:
            seg_src_out_path = output_dir / f"seg_{src_frame_id:05d}.png"
            cv2.imwrite(
                str(seg_src_out_path),
                cv2.cvtColor(seg_maps[src_frame_id], cv2.COLOR_RGB2BGR),
            )
    else:
        rgb_dir = (
            vkitti_2_path
            / f"vkitti_2.0.3_rgb/Scene{args.location:02d}/{args.variation}/frames/rgb/Camera_{args.cam_id}"
        )
        src_frame_path = rgb_dir / f"rgb_{src_frame_id:05d}.jpg"
        src_frame = cv2.imread(str(src_frame_path))
        src_frame = get_resized_and_cropped_image(
            src_frame,
            s,
            offset_x,
            diffusion_img_shape,
            interpolation=cv2.INTER_NEAREST,
        )

    args_out_path = output_dir / "args.yaml"
    with open(args_out_path, "w") as f:
        yaml.dump(vars(args), f)

    src_frame_out_path = output_dir / f"{src_frame_id:05d}.png"
    cv2.imwrite(str(src_frame_out_path), src_frame)

    inpaint_pipeline = generate_sd15.init_inpainting_pipeline(
        args.use_depth_for_inpainting,
        args.use_segmentation_for_inpainting,
        args.lora_weights_path,
    )

    debug_dir = None
    if args.debug:
        debug_dir = output_dir
    output_frames = warp_and_inpaint_sequence(
        args,
        cam,
        src_frame,
        args.frame_ids,
        inpaint_pipeline,
        depths_raw,
        depth_maps,
        seg_maps,
        debug_dir,
    )

    # Save output video
    output_video_path = output_dir / f"{args.experiment_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        args.fps,
        diffusion_img_shape,
    )
    for frame in output_frames:
        video_writer.write(frame)
    video_writer.release()


if __name__ == "__main__":
    args = parse_args()
    main(args)
