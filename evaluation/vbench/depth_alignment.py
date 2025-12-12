import numpy as np
from tqdm import tqdm
import cv2
from vbench.utils import load_dimension_info
from pathlib import Path
from PIL import Image

from .distributed import (
    get_world_size,
    get_rank,
    distribute_list_to_rank,
    gather_list_of_dict,
)


def get_frames(video_dir):
    frames = []
    for frame_path in sorted(video_dir.iterdir()):
        frame = Image.open(str(frame_path)).convert("L")
        frame = np.array(frame)
        frames.append(frame)
    frames = np.array(frames)
    return frames


def calculate_mae(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    abs_diff = np.abs(img1 - img2)
    return np.mean(abs_diff, axis=(-2, -1))


def calculate_seq_mae(video_dir, video_gt_dir):
    frames = get_frames(video_dir)
    frames_gt = get_frames(video_gt_dir)
    _, h_gt, w_gt = frames_gt.shape
    frames_resized = []
    for frame in frames:
        frame_resized = cv2.resize(frame, (w_gt, h_gt), interpolation=cv2.INTER_LINEAR)
        frames_resized.append(frame_resized)
    frames_resized = np.array(frames_resized)
    seq_mae = calculate_mae(frames_resized, frames_gt)
    return seq_mae


def calculate_scores(seq_mae):    
    scores = (255.0 - seq_mae) / 255.0
    return scores


def depth_alignment(video_list, gt_depth_maps_dir):
    scores_total = []
    maes_total = []
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        video_path = Path(video_path)
        video_stem = video_path.stem
        gen_condition_dir = video_path.parent.parent / "output_depth_frames" / video_stem
        gt_condition_dir = Path(gt_depth_maps_dir) / video_stem

        seq_mae = calculate_seq_mae(gen_condition_dir, gt_condition_dir)
        mean_seq_mae = np.mean(seq_mae).item()
        scores = calculate_scores(seq_mae)
        mean_score = np.mean(scores).item()
        
        video_results.append({"video_path": str(video_path),
                              "gen_condition_dir": str(gen_condition_dir),
                              "gt_condition_dir": str(gt_condition_dir),
                              "video_results": mean_score,
                              "frame_results": scores.tolist(),
                              "video_mae_results": mean_seq_mae,
                              "frame_mae_results": seq_mae.tolist()})
        scores_total.extend(scores.tolist())
        maes_total.extend(seq_mae.tolist())

    overall_score = np.mean(scores_total)
    overall_mae = np.mean(maes_total)

    return overall_score, overall_mae, video_results


def compute_depth_alignment(json_dir, device, submodules_list, **kwargs):
    video_list, _ = load_dimension_info(json_dir, dimension="depth_alignment", lang="en")
    video_list = distribute_list_to_rank(video_list)
    gt_depth_maps_dir = kwargs["gt_depth_maps_dir"]
    overall_score, overall_mae, video_results = depth_alignment(video_list, gt_depth_maps_dir)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        overall_score = sum([d["video_results"] for d in video_results]) / len(video_results)
        overall_mae = sum([d["video_mae_results"] for d in video_results]) / len(video_results)

    mae_dict = {"mae": overall_mae}
    return overall_score, mae_dict, video_results


def main():
    gen_img_path = "data/gen_depth.png"
    gt_img_path = "data/gt_depth.png"
    gen_img = Image.open(gen_img_path).convert("L")
    gen_img = np.array(gen_img)
    gt_img = Image.open(gt_img_path).convert("L")
    gt_img = np.array(gt_img)
    h, w = gt_img.shape
    gen_img = cv2.resize(gen_img, (w, h), interpolation=cv2.INTER_LINEAR)
    mae = calculate_mae(gen_img, gt_img)
    print(f"MAE: {mae}")
    mae_cv2 = np.mean(cv2.absdiff(gen_img, gt_img))
    print(f"MAE using OpenCV: {mae_cv2}")


if __name__ == "__main__":
    main()
