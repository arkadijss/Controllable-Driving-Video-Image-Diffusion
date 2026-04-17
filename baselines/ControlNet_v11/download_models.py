from huggingface_hub import hf_hub_download, snapshot_download

model_path = hf_hub_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    filename="v1-5-pruned.ckpt",
    local_dir="models",
)

files = [
    "control_v11f1p_sd15_depth.pth",
    "control_v11p_sd15_seg.pth",
]
for f in files:
    file_path = hf_hub_download(
        repo_id="lllyasviel/ControlNet-v1-1", filename=f, local_dir="models"
    )

local_dir = snapshot_download(
    repo_id="openai/clip-vit-large-patch14", local_dir="openai/clip-vit-large-patch14"
)
