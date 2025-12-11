from pathlib import Path

import yaml
from mmseg.core.evaluation import get_palette
from preprocessing import gen_segmentation_data


def gen_segmentation_maps(root_dir):
    segmentation_args = gen_segmentation_data.InternImageArgs(
        config="InternImage/segmentation/configs/ade20k/mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py",
        checkpoint="InternImage/segmentation/checkpoint_dir/seg/mask2former_internimage_h_896_80k_cocostuff2ade20k.pth",
        device="cuda:0",
        palette="ade20k",
        opacity=1.0,
    )
    segmentation_model = gen_segmentation_data.load_internimage_h(segmentation_args)
    color_palette = get_palette(segmentation_args.palette)

    for method_dir in root_dir.iterdir():
        for submethod_dir in method_dir.iterdir():
            config_path = submethod_dir / "vbench_config.yaml"
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)

            conditions = config["conditions"]
            if "segmentation" not in conditions:
                print("InternImage-H maps not used as condition, skipping")
                continue

            segmentation_output_dir = submethod_dir / "output_segmentation_frames"
            segmentation_output_dir.mkdir(exist_ok=True)
            output_frame_dir = submethod_dir / "output_frames"
            for clip_dir in output_frame_dir.iterdir():
                segmentation_args.img = str(clip_dir)
                segmentation_args.out = str(segmentation_output_dir / clip_dir.name)
                gen_segmentation_data.run_internimage_h(
                    segmentation_args,
                    segmentation_model,
                    color_palette,
                    gen_segmentation_data.test_single_image,
                )
                print(f"Generated segmentation maps: {segmentation_args.out}")


if __name__ == "__main__":
    root_dir = Path("~/data/KITTI-360_output/VBench").expanduser()
    gen_segmentation_maps(root_dir)
