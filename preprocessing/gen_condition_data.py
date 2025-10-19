import glob
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from InternImage.segmentation.image_demo import test_single_image
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor
from mmseg.core import get_classes
from mmseg.core.evaluation import get_palette


@dataclass
class DepthAnythingArgs:
    img_path: str = None
    outdir: str = None
    encoder: str = 'vitl'
    input_size: int = 518
    pred_only: bool = True
    grayscale: bool = True


@dataclass
class InternImageArgs:
    img: str = None
    config: str = None
    checkpoint: str = None
    out: str = None
    device: str = 'cuda:0'
    palette: str = 'ade20k'
    opacity: float = 1.0


def load_depth_anything_v2(args: DepthAnythingArgs):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'Depth_Anything_V2/checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    return depth_anything


def run_depth_anything_v2(args: DepthAnythingArgs, depth_anything: DepthAnythingV2):
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)


def load_internimage_h(args: InternImageArgs):
    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)
    return model


def run_internimage_h(args: InternImageArgs, model: nn.Module):
    # check arg.img is directory of a single image.
    if os.path.isdir(args.img):
        for img in sorted(os.listdir(args.img)):
            test_single_image(model, os.path.join(args.img, img),
                              args.out, get_palette(args.palette), args.opacity)
    else:
        test_single_image(model, args.img, args.out,
                          get_palette(args.palette), args.opacity)
    

def get_condition_data(input_dir: Path, output_dir: Path, method: str):
    cameras = ["image_00"]

    if method == "depth":
        args = DepthAnythingArgs(encoder='vitl',
                                 input_size=518,
                                 pred_only=True,
                                 grayscale=True)
        model = load_depth_anything_v2(args)
    elif method == "segmentation":
        args = InternImageArgs(config='InternImage/segmentation/configs/ade20k/mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py',
                               checkpoint='InternImage/segmentation/checkpoint_dir/seg/mask2former_internimage_h_896_80k_cocostuff2ade20k.pth',
                               device='cuda:0',
                               palette='ade20k',
                               opacity=1.0)
        model = load_internimage_h(args)

    for sequence_dir in input_dir.iterdir():
        for cam in cameras:
            cam_dir = sequence_dir / cam

            if method == "depth":
                args.img_path = str(cam_dir)
                args.outdir = str(output_dir / sequence_dir.name / cam)
                run_depth_anything_v2(args, model)

            elif method == "segmentation":
                args.img = str(cam_dir)
                args.out = str(output_dir / sequence_dir.name / cam)
                run_internimage_h(args, model)


if __name__ == "__main__":
    input_dir = Path("~/data/KITTI-360_proc/val/data_2d_raw_center_cropped").expanduser()
    output_dir = Path("~/data/KITTI-360_proc/val").expanduser()
    get_condition_data(input_dir, output_dir / "depth", "depth")
    get_condition_data(input_dir, output_dir / "segmentation", "segmentation")
