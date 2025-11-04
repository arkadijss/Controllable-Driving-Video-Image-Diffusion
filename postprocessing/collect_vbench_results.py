import json
from pathlib import Path

import pandas as pd


def collect_vbench_results(results_root: Path, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)

    combined_rows = []

    for method_dir in sorted(results_root.iterdir()):
        print(f"Processing method: {method_dir.name}")
        rows = []

        for submethod_dir in sorted(method_dir.iterdir()):
            json_files = list(submethod_dir.glob("*results_*_eval_results.json"))
            if not json_files:
                print(f"No results JSON found in {submethod_dir}")
                continue

            result_path = json_files[0]
            with open(result_path, "r") as f:
                data = json.load(f)

            result_row = {"submethod": submethod_dir.name, "method": method_dir.name}
            for dim, value in data.items():
                # average score
                result_row[dim] = value[0]

            rows.append(result_row)
            combined_rows.append(result_row)

        df = pd.DataFrame(rows).set_index("submethod")

        method_out_dir = output_root / method_dir.name
        method_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = method_out_dir / f"{method_dir.name}_summary.csv"

        df.to_csv(out_path, index=True)
        print(df.round(3))
        print(f"Saved summary to {out_path}")

    if combined_rows:
        combined_df = pd.DataFrame(combined_rows).set_index(["method", "submethod"])
        combined_out_path = output_root / "all_methods_summary.csv"
        combined_df.to_csv(combined_out_path)
        print(f"Saved combined summary for all methods to {combined_out_path}")
    else:
        print("No results found to create combined summary")


if __name__ == "__main__":
    results_root = Path("/mnt/data/KITTI-360_output/VBench_results")
    output_root = Path("./results/VBench_summary")
    collect_vbench_results(results_root, output_root)
