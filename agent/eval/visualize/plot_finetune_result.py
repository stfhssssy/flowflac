# MIT License

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def load_eval_points(result_path):
    with result_path.open("rb") as result_file:
        run_results = pickle.load(result_file)

    if not isinstance(run_results, list):
        raise ValueError(f"Expected {result_path} to contain a list, got {type(run_results)}.")

    eval_points = [
        result
        for result in run_results
        if "eval/success_rate" in result and "itr" in result
    ]
    if not eval_points:
        raise ValueError(f"No eval/success_rate points found in {result_path}.")
    return eval_points


def plot_success_rate(eval_points, output_prefix, label, title, x_axis):
    x_key = "step" if x_axis == "step" else "itr"
    x_values = [point[x_key] for point in eval_points]
    success_rates = [point["eval/success_rate"] for point in eval_points]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, success_rates, color="#177E89", linewidth=3, marker="o", label=label)
    ax.set_xlabel("Environment steps" if x_axis == "step" else "Iteration", fontsize=18)
    ax.set_ylabel("Success Rate", fontsize=18)
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=14, loc="lower right")
    ax.set_title(title, fontsize=20)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot eval success rate from a finetune result.pkl file."
    )
    parser.add_argument("--result-pkl", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--label", default="Finetune")
    parser.add_argument("--title", default="Finetune success rate")
    parser.add_argument("--x-axis", choices=("step", "itr"), default="step")
    return parser.parse_args()


def main():
    args = parse_args()
    eval_points = load_eval_points(args.result_pkl)
    png_path, pdf_path = plot_success_rate(
        eval_points=eval_points,
        output_prefix=args.output_prefix,
        label=args.label,
        title=args.title,
        x_axis=args.x_axis,
    )
    print(f"Plotted {len(eval_points)} eval points from {args.result_pkl}.")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
