import numpy as np
from tqdm import tqdm

import torch
import clip
from PIL import Image
from vbench.utils import load_video, load_dimension_info

from .distributed import (
    get_world_size,
    get_rank,
    distribute_list_to_rank,
    gather_list_of_dict,
)


def calculate_text_image_similarity(model, text, image):
    with torch.no_grad():
        text_features = model.encode_text(text)
        image_features = model.encode_image(image)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = text_features @ image_features.T

        return similarity


def framewise_consistency(clip_model, preprocess, video_dict, device):
    sim = 0.0
    cnt = 0
    video_results = []
    for info in tqdm(video_dict, disable=get_rank() > 0):
        query = info["prompt"]
        text = clip.tokenize([query]).to(device)
        video_list = info["video_list"]
        for video_path in video_list:
            cur_video = []
            with torch.no_grad():
                video_arrays = load_video(video_path, return_tensor=False)
                images = [Image.fromarray(i) for i in video_arrays]
                for image in images:
                    image = preprocess(image).unsqueeze(0).to(device)
                    cur_sim_t = calculate_text_image_similarity(clip_model, text, image)
                    cur_sim = float(cur_sim_t[0][0].cpu())
                    cur_video.append(cur_sim)
                    sim += cur_sim
                    cnt +=1
                video_sim = np.mean(cur_video)
                video_results.append({
                    "video_path": video_path, 
                    "video_results": video_sim, 
                    "frame_results": cur_video})
    sim_per_frame = sim / cnt
    return sim_per_frame, video_results


def compute_framewise_consistency(json_dir, device, submodules_list, **kwargs):
    clip_model, preprocess = clip.load(device=device, **submodules_list)
    _, video_dict = load_dimension_info(json_dir, dimension="framewise_consistency", lang="en")
    video_dict = distribute_list_to_rank(video_dict)
    all_results, video_results = framewise_consistency(clip_model, preprocess, video_dict, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d["video_results"] for d in video_results]) / len(video_results)
    return all_results, video_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    prompt = "a driving scene in a town, photorealistic, sunny weather"
    text = clip.tokenize([prompt]).to(device)
    image = preprocess(Image.open("data/driving_scene.png")).unsqueeze(0).to(device)
    similarity = calculate_text_image_similarity(model, text, image)
    print("Similarity:", float(similarity[0][0].cpu()))
