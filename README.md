# Controllable-Driving-Video-Image-Diffusion

## Description

This repository contains the implementation of the research project "Controllable Photorealistic Driving Video Generation using Image Diffusion Models". It explores image diffusion models as a more cost effective solution to controllable photorealistic driving video generation compared to video diffuson models. The repository contains scripts to preprocess KITTI-360 and Virtual KITTI 2 datasets and evaluate different image and video diffusion-based methods on them. Additionally, it contains the implementation of a zero-shot image diffusion-based method inspired by VideoControlNet and SceneScape. It uses a warping-and-inpainting strategy and synthetic data from Virtual KITTI 2. Warping captures the motion through explicit geometric transformations, while first frame generation and inpainting is done by an image diffusion model conditioned on depth and segmentation control maps. The project was done by Arkādijs Sergejevs in the Autonomous Vision Group at the University of Tübingen under the supervision of Christina-Ourania Tze.

## Getting Started

The environment for our method can be set up using the following commands:

```
conda create -n cdvid python=3.10
conda activate cdvid
pip install -r requirements.txt
```

To install libraries for other methods and condition processors, please refer to the respective repositories.

## Usage

### VBench metrics

The methods are evaluated using [VBench](https://github.com/Vchitect/VBench). To use our custom metrics, please clone the VBench repository and copy `evaluation/vbench` and `evaluate/evaluate.py` to the repository.

### KITTI-360 evaluation

The methods we evaluate are a naive baseline generating frames independently using [ControlNet 1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly), zero-shot image diffusion-based [ControlVideo](https://github.com/YBYBZhang/ControlVideo) and video diffusion-based SVD, I2VGen-XL from [Ctrl-Adapter](https://ctrl-adapter.github.io/).

#### Preprocessing

First, you should download [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/).

The preprocessing scripts for KITTI-360 dataset are located in `preprocessing/kitti360`. To obtain depth maps, we use [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2). The segmentation maps in ADE20K format are obtained using [InternImage-H](https://github.com/opengvlab/internimage).

#### Inference

Inference scripts for each KITTI-360 baseline can be found in `baselines/{method}`. To use them, you should clone the corresponding method's repository and paste the scripts into its root directory.

#### Postprocessing

Postprocessing scripts for KITTI-360 evaluation can be found in `postprocessing`.

### Virtual KITTI 2 evaluation

We evaluate 3 methods - a naive baseline generating frames independently using ControlNet 1.1, ControlVideo and our method.

#### Preprocessing

Please refer to [Virtual KITTI 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/) for download instructions.

To preprocess the dataset, run the command `PYTHONPATH=.:src python preprocessing/preprocess_vkitti_2.py`.

#### Inference

To run inference with ControlNet 1.1, you can use the command `PYTHONPATH=. python src/controlnet_full.py --use_segmentation_for_generation`.

Inference with ControlVideo can be run using the scripts in `baselines/ControlVideo` by cloning ControlVideo and pasting the scripts into its root directory.

Our method on the full dataset can be run as follows:

```
PYTHONPATH=. python src/warp_and_inpaint_full.py \
--generate_first_frame \
--use_segmentation_for_generation \
--use_depth_for_inpainting \
--use_segmentation_for_inpainting
```

If you would like to run it on a single sequence without preprocessing, you can use the script `src/warp_and_inpaint.py` analogously.

#### LoRA

To improve the photorealism of the generated scenes, we have fine-tuned a LoRA on KITTI-360 scenes.

The preprocessing scripts for collecting the dataset and generating captions are available in `preprocessing/lora`.

For training we use the [Hugging Face LoRA](https://huggingface.co/blog/lora) script. To run fine-tuning, refer to the script `scripts/finetune_lora.sh`.

To run our method with LoRA, you can use `src/warp_and_inpaint.py` or `src/warp_and_inpaint_full.py`, and pass `--lora_weights_path data/lora/pytorch_lora_weights.safetensors`.

#### Postprocessing

The outputs for ControlNet 1.1 and our method are already returned in the VBench format.

To convert the ControlVideo output to VBench format, please refer to `postprocessing/controlvideo_to_vbench.py`.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

This repository also includes code from other open-source projects. Please refer to the [NOTICE](NOTICE) file for the references and project licenses.

## Acknowledgments

Many thanks to everyone whose work was used in this project. Special thanks to:

- [VideoControlNet](https://vcg-aigc.github.io/), [SceneScape](https://scenescape.github.io/), whose work our method is built upon
- ControlNet 1.1, ControlVideo and Ctrl-Adapter, which were used as baselines
- KITTI-360 and Virtual Kitti 2, which provided datasets for experiments
- Depth Anything V2 and InternImage, which were used to generate control maps for the KITTI-360 dataset
- [kitti360Scripts](https://github.com/autonomousvision/kitti360scripts), which was used to process the KITTI-360 dataset and implement warping
- VBench, which was used for evaluating the methods
- The Hugging Face Inc. team for providing the LoRA fine-tuning scripts
