from pathlib import Path

import yaml
from preprocessing.kitti360 import gen_depth_data


def gen_depth_maps(root_dir):
    depth_args = gen_depth_data.DepthAnythingArgs(
        encoder="vitl", input_size=518, pred_only=True, grayscale=True
    )
    depth_model = gen_depth_data.load_depth_anything_v2(depth_args)

    for method_dir in root_dir.iterdir():
        for submethod_dir in method_dir.iterdir():
            config_path = submethod_dir / "vbench_config.yaml"
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)

            conditions = config["conditions"]
            if "depth" not in conditions:
                print("Depth Anything V2 maps not used as condition, skipping")
                continue

            depth_output_dir = submethod_dir / "output_depth_frames"
            depth_output_dir.mkdir(exist_ok=True)
            output_frame_dir = submethod_dir / "output_frames"
            for clip_dir in output_frame_dir.iterdir():
                depth_args.img_path = str(clip_dir)
                depth_args.outdir = str(depth_output_dir / clip_dir.name)
                gen_depth_data.run_depth_anything_v2(depth_args, depth_model)
                print(f"Generated depth maps: {depth_args.outdir}")


if __name__ == "__main__":
    root_dir = Path("~/data/KITTI-360_output/VBench").expanduser()
    gen_depth_maps(root_dir)
