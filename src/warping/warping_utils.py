import numpy as np


def image2cam(depth, cam):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pts_img_norm = np.stack((u, v, np.ones((h, w))), axis=-1).reshape(-1, 3)
    pts_img = pts_img_norm * depth.reshape(-1, 1)
    K = cam.K[:3, :3]
    pts_cam = pts_img @ np.linalg.inv(K).T
    return pts_cam


def cam2world(pts_cam, cam, frame_id):
    pts_cam_homog = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world_homog = pts_cam_homog @ cam.cam2world[frame_id].T
    pts_world = (pts_world_homog / pts_world_homog[:, 3:4])[:, :3]
    return pts_world


def image2world(depth, cam, frame_id):
    pts_cam = image2cam(depth, cam)
    pts_world = cam2world(pts_cam, cam, frame_id)
    return pts_world


def get_visible_coords(
    pts_world, cam, frame_id_view, depth_view, depth_thr, error_factor=7.0
):
    (u_view, v_view), d_proj = cam.project_vertices(pts_world, frame_id_view, True)
    h, w = depth_view.shape
    valid = (d_proj > 0) & (u_view >= 0) & (u_view < w) & (v_view >= 0) & (v_view < h)

    u_view_valid = np.round(u_view[valid]).astype(int)
    v_view_valid = np.round(v_view[valid]).astype(int)

    d_view_valid = depth_view[v_view_valid, u_view_valid]
    d_proj_valid = d_proj[valid]

    d_vis = (
        np.abs(d_proj_valid - d_view_valid) < depth_thr * d_view_valid / error_factor
    )

    u_view_final = u_view_valid[d_vis].astype(int)
    v_view_final = v_view_valid[d_vis].astype(int)

    valid_final = np.zeros_like(valid)
    valid_final[valid] = d_vis

    return u_view_final, v_view_final, valid_final


def warp_frame(
    src_frame,
    cam,
    src_frame_id,
    tgt_frame_id,
    depth_src,
    depth_tgt,
    depth_thr,
):
    # Get world points for pixel depths ordered by rows and columns
    pts_tgt_world = image2world(depth_tgt, cam, tgt_frame_id)

    u_src_final, v_src_final, valid_final = get_visible_coords(
        pts_tgt_world, cam, src_frame_id, depth_src, depth_thr
    )

    h, w = depth_tgt.shape
    src_frame_warped = np.zeros_like(src_frame)
    src_frame_warped_flat = src_frame_warped.reshape(-1, 3)
    src_frame_warped_flat[valid_final] = src_frame[v_src_final, u_src_final]
    src_frame_warped = src_frame_warped_flat.reshape(h, w, 3)

    warp_missing_mask = np.ones_like(depth_tgt)
    warp_missing_mask_flat = warp_missing_mask.flatten()
    warp_missing_mask_flat[valid_final] = 0
    warp_missing_mask = warp_missing_mask_flat.reshape(h, w)

    return src_frame_warped, warp_missing_mask
