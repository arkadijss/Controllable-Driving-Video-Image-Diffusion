from huggingface_hub import snapshot_download

# Download Canny model
snapshot_download(
    repo_id="lllyasviel/sd-controlnet-canny",
    local_dir="./checkpoints/sd-controlnet-canny",
    local_dir_use_symlinks=False,
)

# Download Depth model
snapshot_download(
    repo_id="lllyasviel/sd-controlnet-depth",
    local_dir="./checkpoints/sd-controlnet-depth",
    local_dir_use_symlinks=False,
)

# Download Stable Diffusion v1-5 model
snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_dir="./checkpoints/stable-diffusion-v1-5",
    local_dir_use_symlinks=False,
)

# Download ControlNet 1.1 Depth model
snapshot_download(
    repo_id="lllyasviel/control_v11f1p_sd15_depth",
    local_dir="./checkpoints/control_v11f1p_sd15_depth",
    local_dir_use_symlinks=False,
)

# Download ControlNet 1.1 Canny model
snapshot_download(
    repo_id="lllyasviel/control_v11p_sd15_canny",
    local_dir="./checkpoints/control_v11p_sd15_canny",
    local_dir_use_symlinks=False,
)
