# MIT License

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt


SUCCESS_RATE_KEYS = {
    "eval": ("eval/success_rate", "eval/success rate"),
    "train": ("train/success_rate", "train/success rate"),
}


def get_success_rate(result, data_source):
    for key in SUCCESS_RATE_KEYS[data_source]:
        value = result.get(key)
        if value is not None:
            return value
    raise KeyError(f"Expected one of {SUCCESS_RATE_KEYS[data_source]} in {data_source} point.")


def has_success_rate(result, data_source):
    return any(result.get(key) is not None for key in SUCCESS_RATE_KEYS[data_source])


def load_result_points(result_path, data_source):
    with result_path.open("rb") as result_file:
        run_results = pickle.load(result_file)

    if not isinstance(run_results, list):
        raise ValueError(f"Expected {result_path} to contain a list, got {type(run_results)}.")

    points = [
        result
        for result in run_results
        if has_success_rate(result, data_source) and "itr" in result
    ]
    if not points:
        raise ValueError(f"No success-rate {data_source} points found in {result_path}.")
    return points


def infer_wandb_step_scale(config):
    try:
        return config["env"]["n_envs"] * config["train"]["n_steps"] * config["act_steps"]
    except (KeyError, TypeError):
        return None


def load_wandb_points(run_path, data_source):
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("WandB is required for --wandb-series.") from exc

    run = wandb.Api().run(run_path)
    step_scale = infer_wandb_step_scale(run.config)
    points = []
    for row in run.scan_history():
        if not has_success_rate(row, data_source):
            continue

        itr = row.get("itr", row.get("_step"))
        if itr is None:
            continue

        point = {
            "itr": itr,
            f"{data_source}/success_rate": get_success_rate(row, data_source),
        }
        step = row.get("step", row.get("train/total env step"))
        if step is None and step_scale is not None:
            step = itr * step_scale
        if step is not None:
            point["step"] = step
        points.append(point)

    if not points:
        raise ValueError(f"No success-rate {data_source} points found in WandB run {run_path}.")
    return points


def smooth_values(values, window):
    if window <= 1:
        return values

    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        window_values = values[start : index + 1]
        smoothed.append(sum(window_values) / len(window_values))
    return smoothed


def plot_success_rate(series, output_prefix, title, x_axis, data_source, smooth_window):
    x_key = "step" if x_axis == "step" else "itr"

    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if style_name in plt.style.available:
            plt.style.use(style_name)
            break
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for index, (label, points) in enumerate(series):
        missing_x_points = [point for point in points if x_key not in point]
        if missing_x_points:
            raise ValueError(f"{len(missing_x_points)} {data_source} points for {label} have no {x_key}.")

        x_values = [point[x_key] for point in points]
        success_rates = [get_success_rate(point, data_source) for point in points]
        color = colors[index % len(colors)]
        if smooth_window > 1:
            ax.plot(
                x_values,
                success_rates,
                color=color,
                linewidth=1.5,
                alpha=0.22,
            )
            success_rates = smooth_values(success_rates, smooth_window)

        ax.plot(
            x_values,
            success_rates,
            color=color,
            linewidth=3,
            marker="o" if smooth_window <= 1 else None,
            label=label,
        )

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
        description="Plot eval or train success rate from finetune result.pkl files and WandB runs."
    )
    parser.add_argument(
        "--series",
        action="append",
        nargs=2,
        metavar=("RESULT_PKL", "LABEL"),
        help="Result file and legend label. Repeat to compare multiple runs.",
    )
    parser.add_argument(
        "--wandb-series",
        action="append",
        nargs=2,
        metavar=("RUN_PATH", "LABEL"),
        help="WandB entity/project/run_id and legend label. Repeat for remote runs.",
    )
    parser.add_argument(
        "--result-pkl",
        type=Path,
        help="Single result file. Kept for the one-run command form.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--label", default="Finetune")
    parser.add_argument("--title", default="Finetune success rate")
    parser.add_argument("--data-source", choices=("eval", "train"), default="eval")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Trailing moving-average window in logged points. Use 1 for raw curves.",
    )
    parser.add_argument("--x-axis", choices=("step", "itr"), default="step")
    args = parser.parse_args()
    if not args.series and not args.wandb_series and args.result_pkl is None:
        parser.error("provide --result-pkl, --series, or --wandb-series")
    if args.smooth_window < 1:
        parser.error("--smooth-window must be >= 1")
    return args


def main():
    args = parse_args()
    series_specs = [(Path(result_path), label) for result_path, label in args.series or []]
    if args.result_pkl is not None:
        series_specs.append((args.result_pkl, args.label))

    series = []
    for result_path, label in series_specs:
        points = load_result_points(result_path, args.data_source)
        series.append((label, points))
        print(f"Loaded {len(points)} {args.data_source} points from {result_path}.")

    for run_path, label in args.wandb_series or []:
        points = load_wandb_points(run_path, args.data_source)
        series.append((label, points))
        print(f"Loaded {len(points)} {args.data_source} points from WandB run {run_path}.")

    png_path, pdf_path = plot_success_rate(
        series=series,
        output_prefix=args.output_prefix,
        title=args.title,
        x_axis=args.x_axis,
        data_source=args.data_source,
        smooth_window=args.smooth_window,
    )
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
