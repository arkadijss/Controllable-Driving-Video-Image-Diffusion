import os
from pathlib import Path

import cv2
import numpy as np
from kitti360scripts.helpers.project import CameraPerspective

from warping import warping_utils


def main():
    kitti360_path = Path(os.environ["KITTI360_DATASET"])
    seq = 0
    cam_id = 0
    src_frame_id = 828
    tgt_frame_id = 832
    sequence = f"2013_05_28_drive_{seq:04d}_sync"
    depth_dir = Path("data") / "depths" / f"{sequence}_depths"
    depth_thr = 1
    output_dir = (
        Path("outputs")
        / "warped_frames_kitti360"
        / f"{src_frame_id:010d}_to_{tgt_frame_id:010d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    src_frame_path = (
        kitti360_path
        / "data_2d_raw"
        / sequence
        / f"image_{cam_id:02d}"
        / "data_rect"
        / f"{src_frame_id:010d}.png"
    )
    src_frame = cv2.imread(str(src_frame_path))

    cam = CameraPerspective(str(kitti360_path), sequence, cam_id)

    depth_tgt_path = depth_dir / f"{tgt_frame_id:010d}_raw_depth_meter.npy"
    depth_tgt = np.load(depth_tgt_path)

    depth_src_path = depth_dir / f"{src_frame_id:010d}_raw_depth_meter.npy"
    depth_src = np.load(depth_src_path)

    src_frame_warped, warp_missing_mask = warping_utils.warp_frame(
        src_frame,
        cam,
        src_frame_id,
        tgt_frame_id,
        depth_src,
        depth_tgt,
        depth_thr,
    )

    src_frame_warped_output_path = (
        output_dir / f"{src_frame_id:010d}_to_{tgt_frame_id:010d}.png"
    )
    cv2.imwrite(str(src_frame_warped_output_path), src_frame_warped)

    warp_missing_mask_output_path = (
        output_dir / f"{src_frame_id:010d}_to_{tgt_frame_id:010d}_mask.png"
    )
    cv2.imwrite(str(warp_missing_mask_output_path), warp_missing_mask * 255)


if __name__ == "__main__":
    main()
