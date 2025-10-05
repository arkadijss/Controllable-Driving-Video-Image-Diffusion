from pathlib import Path

import cv2

method = "ControlNet"

input_ds_dir = Path.home() / "data/KITTI_test"
output_ds_dir = Path.home() / "data/KITTI_output"
suffix = ".png"

rel_dir = "data_tracking_image_2/testing/image_02/0018"
prompt = "a driving scene in a town, photo-realistic, sunny weather"
a_prompt = "best quality, photo-realistic, extremely detailed"
n_prompt = "bad quality, cartoon style, unrealistic"
ddim_steps = 40

args_dict = {
    "prompt": prompt,
    "a_prompt": a_prompt,
    "n_prompt": n_prompt,
    "num_samples": 1,
    "image_resolution": 512,
    # "detect_resolution": 512,
    "ddim_steps": ddim_steps,
    "guess_mode": False,
    "strength": 1.0,
    "scale": 9.0,
    "seed": 42,
    "eta": 0.0,
    # "low_threshold": 100,
    # "high_threshold": 200,
}


def process_sequence(condition, args_dict):
    print(f"Starting to process {condition} sequence with arguments:")
    for k, v in args_dict.items():
        print(f"{k}: {v}")

    if condition == "seg":
        from gradio_seg2image import process
    elif condition == "depth":
        from gradio_depth2image import process
    elif condition == "canny":
        from gradio_canny2image import process

    input_dir = input_ds_dir / rel_dir
    output_dir = output_ds_dir / rel_dir / method / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    for input_img_path in input_dir.glob(f"*{suffix}"):
        input_img = cv2.imread(str(input_img_path))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        args_dict_sample = args_dict.copy()
        args_dict_sample["input_image"] = input_img
        results = process(**args_dict_sample)

        detected_cond = results[0]
        generated_images = results[1:]

        input_img_stem = input_img_path.stem
        detected_cond_out_path = output_dir / f"{input_img_stem}_{condition}{suffix}"
        cv2.imwrite(
            str(detected_cond_out_path), cv2.cvtColor(detected_cond, cv2.COLOR_RGB2BGR)
        )

        for i, generated_img in enumerate(generated_images):
            generated_img_out_path = (
                output_dir / f"{input_img_stem}_{condition}_{i}{suffix}"
            )
            cv2.imwrite(
                str(generated_img_out_path),
                cv2.cvtColor(generated_img, cv2.COLOR_RGB2BGR),
            )

        print(f"Processed image {input_img_stem}")

    print(f"Finished processing sequence {rel_dir} with condition {condition}")


if __name__ == "__main__":
    args_dict_seg_depth = args_dict.copy()
    args_dict_seg_depth["detect_resolution"] = 512
    process_sequence("seg", args_dict_seg_depth)
    process_sequence("depth", args_dict_seg_depth)
    args_dict_canny = args_dict.copy()
    args_dict_canny["low_threshold"] = 100
    args_dict_canny["high_threshold"] = 200
    process_sequence("canny", args_dict_canny)
