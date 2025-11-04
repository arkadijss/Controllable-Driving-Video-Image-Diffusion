from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

color_palette = plt.cm.tab20b.colors

depth_label_map = {
    "ControlNet_v11_Depth_Midas": ("ControlNet 1.1, Depth Midas, Num. Steps 50", 0),
    "ControlNet_v11_Depth_Anything_V2": (
        "ControlNet 1.1, Depth Anything V2, Num. Steps 50",
        1,
    ),
    "ControlVideo_v11_control_v11f1p_sd15_depth_num_inference_steps_50": (
        "ControlVideo, Depth Midas, Num. steps 50",
        2,
    ),
    "ControlVideo_v11_Depth_Anything_V2_num_inference_steps_50": (
        "ControlVideo, Depth Anything V2, Num. steps 50",
        3,
    ),
    "ControlVideo_v11_Depth_Anything_V2_num_inference_steps_100": (
        "ControlVideo, Depth Anything V2, Num. steps 100",
        4,
    ),
    "i2vgenxl_Depth_Midas_num_inference_steps_50": (
        "I2VGen-XL, Depth Midas, Num. steps 50",
        5,
    ),
    "i2vgenxl_Depth_Anything_V2_num_inference_steps_50": (
        "I2VGen-XL, Depth Anything V2, Num. steps 50",
        6,
    ),
    "svd_Depth_Midas_num_inference_steps_25": ("SVD, Depth Midas, Num. steps 25", 7),
    "svd_Depth_Anything_V2_num_inference_steps_25": (
        "SVD, Depth Anything V2, Num. steps 25",
        8,
    ),
    "i2vgenxl_multi_control_adapter_Depth_Midas_SegFormer_num_inference_steps_50": (
        "I2VGen-XL Multi, Depth Midas + SegFormer, Num. steps 50",
        9,
    ),
    "i2vgenxl_multi_control_adapter_Depth_Anything_V2_InternImage-H_num_inference_steps_50": (
        "I2VGen-XL Multi, Depth Anything V2 + InternImage-H, Num. steps 50",
        10,
    ),
}

seg_label_map = {
    "ControlNet_v11_Seg_OFADE20K": ("ControlNet 1.1, OFADE20K, Num. Steps 50", 0),
    "ControlNet_v11_InternImage-H": ("ControlNet 1.1, InternImage-H, Num. Steps 50", 1),
    "ControlVideo_v11_InternImage-H_num_inference_steps_50": (
        "ControlVideo, InternImage-H, Num. steps 50",
        3,
    ),
    "ControlVideo_v11_InternImage-H_num_inference_steps_100": (
        "ControlVideo, InternImage-H, Num. steps 100",
        4,
    ),
    "i2vgenxl_multi_control_adapter_SegFormer_num_inference_steps_50": (
        "I2VGen-XL Multi, SegFormer, Num. steps 50",
        5,
    ),
    "i2vgenxl_multi_control_adapter_InternImage-H_num_inference_steps_50": (
        "I2VGen-XL Multi, InternImage-H, Num. steps 50",
        6,
    ),
    "i2vgenxl_multi_control_adapter_Depth_Midas_SegFormer_num_inference_steps_50": (
        "I2VGen-XL Multi, Depth Midas + SegFormer, Num. steps 50",
        9,
    ),
    "i2vgenxl_multi_control_adapter_Depth_Anything_V2_InternImage-H_num_inference_steps_50": (
        "I2VGen-XL Multi, Depth Anything V2 + InternImage-H, Num. steps 50",
        10,
    ),
}


def plot_vbench_bar(
    csv_path, label_map, dimensions, output_dir, title_suffix, figsize=(10, 10)
):
    df = pd.read_csv(csv_path)
    submethods_to_plot = list(label_map.keys())
    short_labels = [label_map[m][0] for m in submethods_to_plot]
    color_indices = [label_map[m][1] for m in submethods_to_plot]

    colors = [color_palette[idx] for idx in color_indices]
    df_plot = df[df["submethod"].isin(submethods_to_plot)].copy()

    for dimension in dimensions:
        scores_to_plot = [
            df_plot.loc[df_plot["submethod"] == m, dimension].values[0]
            for m in submethods_to_plot
        ]

        plt.figure(figsize=figsize)
        bars = plt.bar(
            range(len(submethods_to_plot)),
            scores_to_plot,
            color=colors,
            edgecolor="black",
        )

        plt.xticks(
            range(len(submethods_to_plot)),
            [str(i + 1) for i in range(len(submethods_to_plot))],
        )

        for bar, value in zip(bars, scores_to_plot):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.005,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
            )

        plt.ylim(0, max(scores_to_plot) * 1.2)
        plt.legend(
            bars,
            short_labels,
            loc="upper right",
            ncol=1,
            fontsize=8,
            frameon=False,
            labelspacing=0.3,
        )

        plt.title(
            f"{dimension.replace('_', ' ').title()} – {title_suffix}",
            fontsize=18,
            pad=15,
        )
        plt.ylabel("Score")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        output_path = output_dir / f"{dimension}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot for {dimension} at {output_path}")


csv_path = "results/VBench_summary/all_methods_summary.csv"
dimensions = [
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "overall_consistency",
]

output_root = Path("results/VBench_plots")

depth_output_dir = output_root / "depth"
depth_output_dir.mkdir(exist_ok=True, parents=True)
plot_vbench_bar(
    csv_path, depth_label_map, dimensions, depth_output_dir, "Depth", figsize=(10, 10)
)

segmentation_output_dir = output_root / "segmentation"
segmentation_output_dir.mkdir(exist_ok=True, parents=True)
plot_vbench_bar(
    csv_path,
    seg_label_map,
    dimensions,
    segmentation_output_dir,
    "Segmentation",
    figsize=(10, 10),
)
