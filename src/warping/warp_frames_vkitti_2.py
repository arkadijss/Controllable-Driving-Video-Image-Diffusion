# Copyright (c) 2026 Arkādijs Sergejevs
# Adapted from kitti360Scripts
# Original Copyright (c) 2021 Autonomous Vision Group
# Licensed under the MIT License.

from pathlib import Path

import cv2
import numpy as np

from warping import warping_utils


class CameraPerspective:
    def __init__(self, dataset_dir, location, variation, cam_id):
        self.dataset_dir = dataset_dir
        self.cam_id = cam_id

        rel_var_dir = Path(f"Scene{location:02d}") / variation

        textgt_var_dir = dataset_dir / "vkitti_2.0.3_textgt" / rel_var_dir
        extrinsics_file = textgt_var_dir / "extrinsic.txt"

        intrinsics_file = textgt_var_dir / "intrinsic.txt"
        intrinsics = np.loadtxt(intrinsics_file, skiprows=1)
        cam_rows = intrinsics[intrinsics[:, 1] == cam_id]
        K_coeffs = cam_rows[0, 2:6]
        self.K = np.array(
            [
                [K_coeffs[0], 0.0, K_coeffs[2]],
                [0.0, K_coeffs[1], K_coeffs[3]],
                [0.0, 0.0, 1.0],
            ]
        )
        self.raw_extrinsics_data = np.loadtxt(extrinsics_file, skiprows=1)
        self.initialize_matrices()

    def initialize_matrices(self):
        cam_data = self.raw_extrinsics_data[
            self.raw_extrinsics_data[:, 1] == self.cam_id
        ]

        coeffs = cam_data[:, 2:14]

        R_flat = coeffs[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]]
        R_matrices = R_flat.reshape((-1, 3, 3))

        t_vectors = coeffs[:, [3, 7, 11]].reshape((-1, 3, 1))

        self.R_matrices = R_matrices
        self.t_vectors = t_vectors

        N = len(cam_data)

        R_T_matrices = np.transpose(R_matrices, axes=(0, 2, 1))  # Transpose each 3x3
        t_inv_vectors = -np.matmul(R_T_matrices, t_vectors)  # -R_T * t

        self.cam2world = np.zeros((N, 4, 4))
        self.cam2world[:, 3, 3] = 1.0
        self.cam2world[:, :3, :3] = R_T_matrices
        self.cam2world[:, :3, 3] = t_inv_vectors.squeeze()

    def world2cam(self, points, R, T, inverse=False):
        assert points.ndim == R.ndim
        assert T.ndim == R.ndim or T.ndim == (R.ndim - 1)
        ndim = R.ndim
        if ndim == 2:
            R = np.expand_dims(R, 0)
            T = np.reshape(T, [1, -1, 3])
            points = np.expand_dims(points, 0)
        if not inverse:
            points = np.matmul(R, points.transpose(0, 2, 1)).transpose(0, 2, 1) + T
        else:
            points = np.matmul(R.transpose(0, 2, 1), (points - T).transpose(0, 2, 1))

        if ndim == 2:
            points = points[0]

        return points

    def cam2image(self, points):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(self.K[:3, :3].reshape([1, 3, 3]), points)
        depth = points_proj[:, 2, :]
        depth[depth == 0] = -1e-6
        u = np.round(points_proj[:, 0, :] / np.abs(depth)).astype(int)
        v = np.round(points_proj[:, 1, :] / np.abs(depth)).astype(int)

        if ndim == 2:
            u = u[0]
            v = v[0]
            depth = depth[0]
        return u, v, depth

    def project_vertices(self, vertices, frame_id, inverse=True):
        # current camera pose
        curr_pose = self.cam2world[frame_id]
        T = curr_pose[:3, 3]
        R = curr_pose[:3, :3]

        # convert points from world coordinate to local coordinate
        points_local = self.world2cam(vertices, R, T, inverse)

        # perspective projection
        u, v, depth = self.cam2image(points_local)

        return (u, v), depth


def main():
    vkitti_2_path = Path("~/data/VKITTI-2/").expanduser()
    location = 1
    variation = "clone"
    cam_id = 0
    src_frame_id = 7
    tgt_frame_id = 12
    depth_thr = 0.2
    output_dir = (
        Path("outputs")
        / "warped_frames_vkitti_2"
        / f"{src_frame_id:05d}_to_{tgt_frame_id:05d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cam = CameraPerspective(vkitti_2_path, location, variation, cam_id)

    rel_var_dir = Path(f"Scene{location:02d}") / variation
    rel_frames_dir = rel_var_dir / "frames"
    rgb_dir = (
        vkitti_2_path / "vkitti_2.0.3_rgb" / rel_frames_dir / "rgb" / f"Camera_{cam_id}"
    )
    src_frame_path = rgb_dir / f"rgb_{src_frame_id:05d}.jpg"
    src_frame = cv2.imread(str(src_frame_path))

    depth_dir = (
        vkitti_2_path
        / "vkitti_2.0.3_depth"
        / rel_frames_dir
        / "depth"
        / f"Camera_{cam_id}"
    )

    depth_tgt_path = depth_dir / f"depth_{tgt_frame_id:05d}.png"
    depth_tgt_raw = cv2.imread(
        str(depth_tgt_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
    )

    depth_src_path = depth_dir / f"depth_{src_frame_id:05d}.png"
    depth_src_raw = cv2.imread(
        str(depth_src_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
    )

    depth_tgt = depth_tgt_raw.astype(np.float32)
    depth_src = depth_src_raw.astype(np.float32)

    # Divide to get meters
    depth_tgt /= 100.0
    depth_src /= 100.0

    src_frame_warped, warp_missing_mask = warping_utils.warp_frame(
        src_frame,
        cam,
        src_frame_id,
        tgt_frame_id,
        depth_src,
        depth_tgt,
        depth_thr,
    )

    src_frame_out_path = output_dir / f"{src_frame_id:05d}.png"
    cv2.imwrite(str(src_frame_out_path), src_frame)

    tgt_frame_path = rgb_dir / f"rgb_{tgt_frame_id:05d}.jpg"
    tgt_frame = cv2.imread(str(tgt_frame_path))
    tgt_frame_out_path = output_dir / f"{tgt_frame_id:05d}.png"
    cv2.imwrite(str(tgt_frame_out_path), tgt_frame)

    src_frame_warped_output_path = (
        output_dir / f"{src_frame_id:05d}_to_{tgt_frame_id:05d}.png"
    )
    cv2.imwrite(str(src_frame_warped_output_path), src_frame_warped)

    warp_missing_mask_output_path = (
        output_dir / f"{src_frame_id:05d}_to_{tgt_frame_id:05d}_mask.png"
    )
    cv2.imwrite(str(warp_missing_mask_output_path), warp_missing_mask * 255)


if __name__ == "__main__":
    main()
