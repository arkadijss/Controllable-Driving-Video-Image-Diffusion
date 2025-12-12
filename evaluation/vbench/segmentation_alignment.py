from pathlib import Path
import numpy as np
from tqdm import tqdm

import cv2
from PIL import Image
from configs.ada_palette import ada_palette

from vbench.utils import load_dimension_info

from .distributed import (
    get_world_size,
    get_rank,
    distribute_list_to_rank,
    gather_list_of_dict,
)


def calculate_accuracy(pred_img, gt_img):
    correct_pixel_count = np.sum(np.all(pred_img == gt_img, axis=2)).item()
    h, w, _ = pred_img.shape
    total_pixel_count = h * w
    return correct_pixel_count / total_pixel_count


def calculate_mean_iou(pred_img, gt_img, palette, min_gt_ratio=0.03):
    ious = []
    dtype = pred_img.dtype
    h, w, _ = pred_img.shape
    total_pixel_count = h * w
    for color in palette:
        color = color.astype(dtype)
        pred_mask = np.all(pred_img == color, axis=2)
        gt_mask = np.all(gt_img == color, axis=2)
        intersection_count = np.logical_and(pred_mask, gt_mask).sum()
        union_count = np.logical_or(pred_mask, gt_mask).sum()
        if gt_mask.sum() < min_gt_ratio * total_pixel_count:
            continue
        iou = intersection_count / union_count
        ious.append(iou)
    mean_iou = np.mean(ious).item()
    return mean_iou


def segmentation_alignment(video_list, gt_segmentation_maps_dir, min_gt_ratio):
    mean_iou_sum = 0.0
    accuracy_sum = 0.0
    cnt = 0
    video_results = []

    for video_path in tqdm(video_list, disable=get_rank() > 0):
        cur_video_mean_iou = []
        cur_video_accuracy = []

        video_path = Path(video_path)
        video_stem = video_path.stem
        gen_condition_dir = video_path.parent.parent / "output_segmentation_frames" / video_stem
        gt_condition_dir = Path(gt_segmentation_maps_dir) / video_stem
        
        for gen_frame_path in sorted(gen_condition_dir.iterdir()):
            frame_name = gen_frame_path.name
            gt_frame_path = gt_condition_dir / frame_name
            gen_frame_img = np.array(Image.open(gen_frame_path))
            gt_frame_img = np.array(Image.open(gt_frame_path))
            h, w, _ = gt_frame_img.shape
            gen_frame_img = cv2.resize(gen_frame_img, (w, h), interpolation=cv2.INTER_NEAREST)

            mean_iou = calculate_mean_iou(gen_frame_img, gt_frame_img, ada_palette, min_gt_ratio)
            accuracy = calculate_accuracy(gen_frame_img, gt_frame_img)
            
            cur_video_mean_iou.append(mean_iou)
            cur_video_accuracy.append(accuracy)

            mean_iou_sum += mean_iou
            accuracy_sum += accuracy
            cnt += 1

        video_mean_iou = np.mean(cur_video_mean_iou)
        video_accuracy = np.mean(cur_video_accuracy)

        video_results.append(
            {
                "video_path": str(video_path),
                "gen_condition_dir": str(gen_condition_dir),
                "gt_condition_dir": str(gt_condition_dir),
                "video_results": video_mean_iou,
                "frame_results": cur_video_mean_iou,
                "video_accuracy_results": video_accuracy,
                "frame_accuracy_results": cur_video_accuracy,
            }
        )

    overall_mean_iou = mean_iou_sum / cnt
    overall_accuracy = accuracy_sum / cnt

    return overall_mean_iou, overall_accuracy, video_results


def compute_segmentation_alignment(json_dir, device, submodules_list, **kwargs):
    video_list, _ = load_dimension_info(
        json_dir, dimension="segmentation_alignment", lang="en"
    )
    video_list = distribute_list_to_rank(video_list)
    gt_segmentation_maps_dir = kwargs["gt_segmentation_maps_dir"]
    min_gt_ratio = kwargs["min_gt_ratio"]
    overall_mean_iou, overall_accuracy, video_results = segmentation_alignment(video_list, gt_segmentation_maps_dir, min_gt_ratio)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        overall_mean_iou = sum([d["video_results"] for d in video_results]) / len(
            video_results
        )
        overall_accuracy = sum([d["video_accuracy_results"] for d in video_results]) / len(
            video_results
        )
    
    additional_info = {"accuracy": overall_accuracy, "min_gt_ratio": min_gt_ratio}
    
    return overall_mean_iou, additional_info, video_results


def main():
    gen_img_path = "data/gen_segmentation.png"
    gt_img_path = "data/gt_segmentation.png"

    gen_img = np.array(Image.open(gen_img_path))
    gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
    gt_img = np.array(Image.open(gt_img_path))
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    h, w, _ = gt_img.shape
    gen_img = cv2.resize(gen_img, (w, h), interpolation=cv2.INTER_NEAREST)

    mean_iou = calculate_mean_iou(gen_img, gt_img, ada_palette)
    print(f"Mean IoU: {mean_iou}")

    accuracy = calculate_accuracy(gen_img, gt_img)
    print(f"Accuracy: {accuracy}")

    mask = np.any(gen_img != gt_img, axis=2)
    mask_vis = mask.astype(np.uint8) * 255
    cv2.imwrite("error_mask.png", mask_vis)


if __name__ == "__main__":
    main()
