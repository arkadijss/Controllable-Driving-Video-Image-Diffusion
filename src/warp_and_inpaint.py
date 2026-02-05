from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import generate_sd15
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


def get_normalized_depth_image(depth_image, max_gen_depth):
    norm_depth_image = depth_image / max_gen_depth
    norm_depth_image = (1 - norm_depth_image) * 255
    norm_depth_image = norm_depth_image.astype(np.uint8)
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


def main():
    vkitti_2_path = Path("~/data/VKITTI-2/").expanduser()
    location = 1
    variation = "clone"
    cam_id = 0
    frame_ids = [7, 12]
    depth_thr = 0.3
    generate_first_frame = True
    max_gen_depth = 80.0
    use_segmentation_for_generation = True
    diffusion_img_shape = (512, 512)  # w, h
    orig_shape = (1242, 375)  # w, h
    opening_kernel_size = 3
    classical_inpaint_kernel_size = 3
    mask_closing_kernel_size = 7
    prefill_missing_regions = False
    prefill_kernel_size = 7
    use_depth_for_inpainting = True
    use_segmentation_for_inpainting = True
    interp_inpaint_kernel_size = 3
    prompt = "A driving scene in a town, photorealistic, clear daylight, blue sky, highly detailed"
    negative_prompt = "Bad quality, worst quality, cartoon style, unrealistic, blurry"
    experiment_name = f"{frame_ids[0]:05d}_to_{frame_ids[-1]:05d}"
    fps = 4

    output_dir = Path(f"outputs") / "warp_and_inpaint_vkitti_2" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cam = warp_frames_vkitti_2.CameraPerspective(
        vkitti_2_path, location, variation, cam_id
    )
    # Update intrinsic matrix K for warping in diffusion image coordinates
    cam, s, offset_x = update_K(cam, orig_shape, diffusion_img_shape)

    rel_var_dir = Path(f"Scene{location:02d}") / variation
    rel_frames_dir = rel_var_dir / "frames"
    rgb_dir = (
        vkitti_2_path / "vkitti_2.0.3_rgb" / rel_frames_dir / "rgb" / f"Camera_{cam_id}"
    )

    src_frame_id = frame_ids[0]

    depth_dir = (
        vkitti_2_path
        / "vkitti_2.0.3_depth"
        / rel_frames_dir
        / "depth"
        / f"Camera_{cam_id}"
    )
    depth_src = get_depth(depth_dir, src_frame_id)

    seg_dir = (
        vkitti_2_path
        / "vkitti_2.0.3_classSegmentation_ADE20K"
        / rel_frames_dir
        / "classSegmentation_ADE20K"
        / f"Camera_{cam_id}"
    )

    if generate_first_frame:
        depth_clipped = np.clip(depth_src, 0, max_gen_depth)
        depth_resized = get_resized_and_cropped_image(
            depth_clipped,
            s,
            offset_x,
            diffusion_img_shape,
            interpolation=cv2.INTER_NEAREST,
        )
        norm_depth_image = get_normalized_depth_image(depth_resized, max_gen_depth)
        norm_depth_out_path = output_dir / f"norm_depth_{src_frame_id:05d}.png"
        cv2.imwrite(str(norm_depth_out_path), norm_depth_image)

        if use_segmentation_for_generation:
            seg_src = get_segmentation_map(seg_dir, src_frame_id)
            seg_src_resized = get_resized_and_cropped_image(
                seg_src,
                s,
                offset_x,
                diffusion_img_shape,
                interpolation=cv2.INTER_NEAREST,
            )
            seg_image = Image.fromarray(seg_src_resized)
            seg_src_out_path = output_dir / f"seg_{src_frame_id:05d}.png"
            cv2.imwrite(
                str(seg_src_out_path), cv2.cvtColor(seg_src_resized, cv2.COLOR_RGB2BGR)
            )

        gen_pipeline = generate_sd15.init_generation_pipeline(
            use_segmentation=use_segmentation_for_generation
        )

        depth_image = generate_sd15.preprocess_depth_image(norm_depth_image)

        control_image = [depth_image]

        if use_segmentation_for_generation:
            control_image.append(seg_image)

        if len(control_image) == 1:
            control_image = control_image[0]

        src_frame_gen = generate_sd15.generate_image(
            gen_pipeline, prompt, negative_prompt, control_image
        )
        src_frame = cv2.cvtColor(np.array(src_frame_gen), cv2.COLOR_RGB2BGR)
    else:
        src_frame_path = rgb_dir / f"rgb_{src_frame_id:05d}.jpg"
        src_frame = cv2.imread(str(src_frame_path))
        src_frame = get_resized_and_cropped_image(
            src_frame, s, offset_x, diffusion_img_shape, interpolation=cv2.INTER_NEAREST
        )

    src_frame_out_path = output_dir / f"{src_frame_id:05d}.png"
    cv2.imwrite(str(src_frame_out_path), src_frame)

    # Save output video
    output_video_path = output_dir / f"{experiment_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        diffusion_img_shape,
    )
    video_writer.write(src_frame)

    depth_src = get_resized_and_cropped_image(
        depth_src, s, offset_x, diffusion_img_shape, interpolation=cv2.INTER_NEAREST
    )
    inpaint_pipeline = generate_sd15.init_inpainting_pipeline(
        use_depth_for_inpainting, use_segmentation_for_inpainting
    )

    for tgt_frame_id in frame_ids[1:]:
        depth_tgt = get_depth(depth_dir, tgt_frame_id)
        depth_tgt = get_resized_and_cropped_image(
            depth_tgt, s, offset_x, diffusion_img_shape, interpolation=cv2.INTER_NEAREST
        )

        src_frame_warped, warp_missing_mask = warping_utils.warp_frame(
            src_frame,
            cam,
            src_frame_id,
            tgt_frame_id,
            depth_src,
            depth_tgt,
            depth_thr,
        )

        warp_missing_mask = warp_missing_mask.astype(np.uint8) * 255

        src_frame_warped = cv2.resize(
            src_frame_warped, diffusion_img_shape, interpolation=cv2.INTER_NEAREST
        )
        warp_missing_mask = cv2.resize(
            warp_missing_mask, diffusion_img_shape, interpolation=cv2.INTER_NEAREST
        )

        # Assert all masked regions are black
        assert np.all(
            src_frame_warped[warp_missing_mask == 255] == 0
        ), "Masked regions in warped frame are not black"

        # Assert the mask is binary
        unique_values = np.unique(warp_missing_mask)
        assert set(unique_values).issubset({0, 255}), "Warp missing mask is not binary"

        src_frame_inpainted_classical, _, opening = inpaint_image_classical(
            src_frame_warped,
            warp_missing_mask,
            opening_kernel_size=opening_kernel_size,
            inpaint_kernel_size=classical_inpaint_kernel_size,
        )

        if mask_closing_kernel_size > 0:
            closing_kernel = np.ones(
                (mask_closing_kernel_size, mask_closing_kernel_size), np.uint8
            )
            diffusion_inpainting_mask = cv2.morphologyEx(
                opening, cv2.MORPH_CLOSE, closing_kernel
            )
        else:
            diffusion_inpainting_mask = opening

        if prefill_missing_regions:
            src_frame_inpainted_classical = cv2.inpaint(
                src_frame_inpainted_classical,
                diffusion_inpainting_mask,
                prefill_kernel_size,
                cv2.INPAINT_TELEA,
            )

        input_image = Image.fromarray(src_frame_inpainted_classical)
        mask_image = Image.fromarray(diffusion_inpainting_mask)

        kwargs = {}
        control_image = []
        if use_depth_for_inpainting:
            depth_inpaint = get_depth(depth_dir, tgt_frame_id)
            depth_inpaint_clipped = np.clip(depth_inpaint, 0, max_gen_depth)
            depth_inpaint_resized = get_resized_and_cropped_image(
                depth_inpaint_clipped,
                s,
                offset_x,
                diffusion_img_shape,
                interpolation=cv2.INTER_NEAREST,
            )
            depth_inpaint_normalized = get_normalized_depth_image(
                depth_inpaint_resized, max_gen_depth
            )
            depth_inpaint_image = generate_sd15.preprocess_depth_image(
                depth_inpaint_normalized
            )
            control_image.append(depth_inpaint_image)

        if use_segmentation_for_inpainting:
            seg_inpaint = get_segmentation_map(seg_dir, tgt_frame_id)
            seg_inpaint_resized = get_resized_and_cropped_image(
                seg_inpaint,
                s,
                offset_x,
                diffusion_img_shape,
                interpolation=cv2.INTER_NEAREST,
            )
            seg_inpaint_image = Image.fromarray(seg_inpaint_resized)
            control_image.append(seg_inpaint_image)

        if len(control_image) == 1:
            kwargs["control_image"] = control_image[0]
        elif len(control_image) > 1:
            kwargs["control_image"] = control_image

        src_frame_inpainted_diffusion = generate_sd15.inpaint_image(
            inpaint_pipeline,
            prompt,
            negative_prompt,
            input_image,
            mask_image,
            **kwargs,
        )
        src_frame_inpainted_diffusion = np.array(src_frame_inpainted_diffusion)

        # Prepare depths for interpolation
        interp_ids = np.arange(src_frame_id + 1, tgt_frame_id)
        interp_depths = []
        for frame_id in interp_ids:
            depth_interp = get_depth(depth_dir, frame_id)
            depth_interp = get_resized_and_cropped_image(
                depth_interp,
                s,
                offset_x,
                diffusion_img_shape,
                interpolation=cv2.INTER_NEAREST,
            )
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
            depth_thr,
            diffusion_img_shape,
            inpaint_kernel_size=interp_inpaint_kernel_size,
        )

        # Save outputs
        output_pair_path = output_dir / f"{src_frame_id:05d}_to_{tgt_frame_id:05d}"
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
        interpolated_frame_out_dir.mkdir(parents=True, exist_ok=True)
        for id, interp_frame in zip(interp_ids, interpolated_frames):
            interp_frame_out_path = interpolated_frame_out_dir / f"interp_{id:05d}.png"
            cv2.imwrite(str(interp_frame_out_path), interp_frame)
            video_writer.write(interp_frame)

        video_writer.write(src_frame_inpainted_diffusion)

        src_frame = src_frame_inpainted_diffusion
        src_frame_id = tgt_frame_id
        depth_src = depth_tgt

    video_writer.release()


if __name__ == "__main__":
    main()
