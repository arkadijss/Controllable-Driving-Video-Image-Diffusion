from pathlib import Path

import cv2
import numpy as np

from configs import ada_palette, vkitti_2_cfg


def map_vkitti2_to_ade20k(seg_img):
    seg_img_ade20k = np.zeros_like(seg_img)
    for class_name, vkitti_color in vkitti_2_cfg.VKITTI_2_PALETTE.items():
        ade20k_idx = vkitti_2_cfg.VKITTI_2_TO_ADE20K_IDX[class_name]
        ade20k_color = ada_palette.ada_palette[ade20k_idx]
        mask = np.all(seg_img == vkitti_color, axis=-1)
        seg_img_ade20k[mask] = ade20k_color
    return seg_img_ade20k


def main():
    vkitti_2_path = Path("~/data/VKITTI-2/").expanduser()
    locations = [1, 2, 6, 18, 20]
    variations = ["clone"]
    cam_ids = [0]

    seg_root_dir = vkitti_2_path / "vkitti_2.0.3_classSegmentation"
    seg_root_out_dir = vkitti_2_path / "vkitti_2.0.3_classSegmentation_ADE20K"
    seg_root_out_dir.mkdir(parents=True, exist_ok=True)

    for location in locations:
        for variation in variations:
            rel_frames_dir = Path(f"Scene{location:02d}") / variation / "frames"
            for cam_id in cam_ids:
                rel_seg_dir = rel_frames_dir / "classSegmentation" / f"Camera_{cam_id}"
                seg_dir = seg_root_dir / rel_seg_dir

                print(f"Processing segmentation dir: {seg_dir}")

                rel_seg_out_dir = (
                    rel_frames_dir / "classSegmentation_ADE20K" / f"Camera_{cam_id}"
                )
                seg_out_dir = seg_root_out_dir / rel_seg_out_dir
                seg_out_dir.mkdir(parents=True, exist_ok=True)

                for seg_path in sorted(seg_dir.glob("*.png")):
                    seg_img = cv2.imread(str(seg_path))
                    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
                    seg_img_ade20k = map_vkitti2_to_ade20k(seg_img)
                    seg_img_ade20k = cv2.cvtColor(seg_img_ade20k, cv2.COLOR_RGB2BGR)
                    save_path = seg_out_dir / seg_path.name
                    cv2.imwrite(str(save_path), seg_img_ade20k)

                    print(f"Saved converted segmentation map to: {save_path}")


if __name__ == "__main__":
    main()
