import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch.nn as nn
from configs.ada_palette import ada_palette
from InternImage.segmentation.image_demo import test_single_image
from mmcv.runner import load_checkpoint
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core import get_classes
from PIL import Image


@dataclass
class InternImageArgs:
    img: str = None
    config: str = None
    checkpoint: str = None
    out: str = None
    device: str = "cuda:0"
    palette: Union[str, np.ndarray] = "ade20k"
    opacity: Optional[float] = 1.0


def load_internimage_h(args: InternImageArgs):
    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = get_classes(args.palette)
    return model


def run_internimage_h(args: InternImageArgs, model: nn.Module, test_func):
    # check arg.img is directory of a single image.
    if os.path.isdir(args.img):
        for img in sorted(os.listdir(args.img)):
            test_func(
                model, os.path.join(args.img, img), args.out, ada_palette, args.opacity
            )
    else:
        test_func(model, args.img, args.out, ada_palette, args.opacity)


def test_single_image_ctrl(model, img_name, out_dir, color_palette, opacity):
    # check img_name is an image file or not
    assumed_imgformat = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
    if not img_name.lower().endswith(assumed_imgformat):
        print(f"Skip {img_name} because it is not an image file.")
        return

    result = inference_segmentor(model, img_name)

    seg = result[0]  # height, width
    color_seg = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3
    for label, color in enumerate(color_palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)
    control_image = Image.fromarray(color_seg)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(img_name))
    control_image.save(out_path)
    print(f"Result is save at {out_path}")


def gen_segmentation_data(input_dir: Path, output_dir: Path, method: str):
    cameras = ["image_00"]

    if method == "segmentation":
        args = InternImageArgs(
            config="InternImage/segmentation/configs/ade20k/mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py",
            checkpoint="InternImage/segmentation/checkpoint_dir/seg/mask2former_internimage_h_896_80k_cocostuff2ade20k.pth",
            device="cuda:0",
            palette="ade20k",
            opacity=1.0,
        )
        model = load_internimage_h(args)
    elif method == "segmentation_ctrl":
        args = InternImageArgs(
            config="InternImage/segmentation/configs/ade20k/mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py",
            checkpoint="InternImage/segmentation/checkpoint_dir/seg/mask2former_internimage_h_896_80k_cocostuff2ade20k.pth",
            device="cuda:0",
            palette=ada_palette,
            opacity=None,
        )

    for sequence_dir in input_dir.iterdir():
        for cam in cameras:
            cam_dir = sequence_dir / cam

            if method == "segmentation":
                args.img = str(cam_dir)
                args.out = str(output_dir / sequence_dir.name / cam)
                run_internimage_h(args, model, test_single_image)
            elif method == "segmentation_ctrl":
                args.img = str(cam_dir)
                args.out = str(output_dir / sequence_dir.name / cam)
                run_internimage_h(args, model, test_single_image_ctrl)


if __name__ == "__main__":
    input_dir = Path(
        "~/data/KITTI-360_proc/val/data_2d_raw_center_cropped"
    ).expanduser()
    output_dir = Path("~/data/KITTI-360_proc/val").expanduser()
    gen_segmentation_data(input_dir, output_dir / "segmentation", "segmentation")
    gen_segmentation_data(
        input_dir, output_dir / "segmentation_ctrl", "segmentation_ctrl"
    )
