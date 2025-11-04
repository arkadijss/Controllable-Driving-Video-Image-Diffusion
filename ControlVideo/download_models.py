from huggingface_hub import snapshot_download


# Download Stable Diffusion v1-5 model
snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_dir="./checkpoints/stable-diffusion-v1-5",
    local_dir_use_symlinks=False,
)

# Download ControlNet Depth model
snapshot_download(
    repo_id="lllyasviel/sd-controlnet-depth",
    local_dir="./checkpoints/sd-controlnet-depth",
    local_dir_use_symlinks=False,
)

# Download ControlNet 1.1 Depth model
snapshot_download(
    repo_id="lllyasviel/control_v11f1p_sd15_depth",
    local_dir="./checkpoints/control_v11f1p_sd15_depth",
    local_dir_use_symlinks=False,
)

# Download ControlNet Segmentation model
snapshot_download(
    repo_id="lllyasviel/sd-controlnet-seg",
    local_dir="./checkpoints/sd-controlnet-seg",
    local_dir_use_symlinks=False,
)

# Download ControlNet 1.1 Segmentation model
snapshot_download(
    repo_id="lllyasviel/control_v11p_sd15_seg",
    local_dir="./checkpoints/control_v11p_sd15_seg",
    local_dir_use_symlinks=False,
)
