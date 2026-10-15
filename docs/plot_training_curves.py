"""Render dev-1.0 measured training history; never interpolate measurements.

Plotting-only dependency: Matplotlib 3.10.0. Model training is not executed.
Run from the repository root: python docs/plot_training_curves.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

HERE = Path(__file__).resolve().parent
WINDOW = 50
BLUE = "#2563a6"
TEAL = "#137d78"
ORANGE = "#d57420"
INK = "#15263d"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(history: dict) -> None:
    if history.get("schema") != "dev-1.0-measured-training-history-1":
        raise ValueError("Unknown measured-history schema")
    train, dev = history["training_records"], history["development_records"]
    if [row["step"] for row in train] != list(range(1, 2449)):
        raise ValueError("Require every original optimizer update exactly once")
    if [row["step"] for row in dev] != [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2448]:
        raise ValueError("Require the 11 original development measurements")
    if sum(row["examples"] for row in train) != 626546 or any(row["count"] != 1497 for row in dev):
        raise ValueError("Training or development population differs")
    for row in train:
        if any(not math.isfinite(row[key]) or row[key] < 0 for key in ("loss", "learning_rate")):
            raise ValueError("Training measurements must be finite and nonnegative")
    for row in dev:
        if any(not math.isfinite(row[key]) or row[key] < 0 for key in ("accuracy", "calibrated_nll", "calibrated_ece", "calibrated_brier", "temperature")):
            raise ValueError("Development measurements must be finite and nonnegative")


def moving_mean(values: list[float], window: int = WINDOW) -> list[float]:
    """Trailing full-window mean; first output belongs to update `window`."""
    if window < 1 or len(values) < window:
        raise ValueError("A complete positive rolling window is required")
    return [math.fsum(values[end - window:end]) / window for end in range(window, len(values) + 1)]


def style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12, weight="bold")
    ax.set_xlabel("Optimizer update", labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_xlim(0, 2500)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.grid(axis="y", color="#dfe5ec", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9d1dc")


def save_figure(fig, output: Path, stem: str) -> None:
    fig.savefig(output / (stem + ".png"), dpi=180, facecolor="white", metadata={"Software": "Matplotlib " + matplotlib.__version__})
    fig.savefig(output / (stem + ".svg"), facecolor="white", metadata={"Date": None, "Creator": "dev-1.0 measured-history plotting script"})
    plt.close(fig)


def plot_loss(history: dict, output: Path) -> None:
    train, dev = history["training_records"], history["development_records"]
    x = [row["step"] for row in train]
    loss = [row["loss"] for row in train]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.23, top=0.77, wspace=0.30)
    fig.suptitle("dev-1.0 | Loss during supervised training", x=0.075, y=0.965, ha="left", fontsize=18, weight="bold", color=INK)
    fig.text(0.075, 0.88, "2,448 recorded training updates  /  11 measured development checkpoints", fontsize=10, color="#536275")
    ax = axes[0]
    ax.plot(x, loss, color="#9bb9d9", linewidth=0.65, alpha=0.75, label="Actual update loss (all 2,448)")
    ax.plot(x[WINDOW-1:], moving_mean(loss), color=BLUE, linewidth=2.1, label="Trailing mean (50 actual updates)")
    style_axis(ax, "Training loss", "Soft-label cross-entropy")
    ax.set_ylim(0.30, 1.65)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax = axes[1]
    steps = [row["step"] for row in dev]
    values = [row["calibrated_nll"] for row in dev]
    ax.plot(steps, values, "o-", color=TEAL, linewidth=1.8, markersize=5, label="Measured calibrated NLL (11 points)")
    style_axis(ax, "Development loss", "Negative log-likelihood")
    ax.set_ylim(0.25, 1.35)
    ax.annotate(f"{values[-1]:.4f}", xy=(steps[-1], values[-1]), xytext=(-10, 15), textcoords="offset points", ha="right", fontsize=10, color=TEAL, weight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.text(0.075, 0.095, "Training CE and development NLL use different targets; they are not the same loss population.", fontsize=9, color="#536275")
    fig.text(0.075, 0.045, "Markers are the 11 observed checkpoints. Connecting segments are visual guides, not added measurements.", fontsize=9, color="#536275")
    save_figure(fig, output, "loss-curves")


def plot_metrics(history: dict, output: Path) -> None:
    dev = history["development_records"]
    steps = [row["step"] for row in dev]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.7))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.15, top=0.83, wspace=0.30, hspace=0.57)
    fig.suptitle("dev-1.0 | Measured development metrics", x=0.075, y=0.965, ha="left", fontsize=18, weight="bold", color=INK)
    fig.text(0.075, 0.91, "1,497 development examples  /  temperature fitted on 983 separate examples", fontsize=10, color="#536275")
    panels = [
        ("accuracy", "Development accuracy", "Accuracy", BLUE, True, (0.30, 0.94)),
        ("calibrated_ece", "Calibration error (ECE)", "Expected calibration error", ORANGE, True, (0, 0.215)),
        ("calibrated_brier", "Calibrated Brier score", "Brier score", TEAL, False, (0.15, 0.73)),
        ("temperature", "Saved calibration temperature", "Temperature", "#7655a8", False, (0.65, 2.45)),
    ]
    for ax, (key, title, ylabel, color, percent, limits) in zip(axes.flat, panels, strict=True):
        values = [row[key] for row in dev]
        ax.plot(steps, values, "o-", color=color, linewidth=1.8, markersize=5)
        style_axis(ax, title, ylabel)
        ax.set_ylim(*limits)
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        label = f"{values[-1]*100:.2f}%" if percent else f"{values[-1]:.4f}"
        ax.annotate(label, xy=(steps[-1], values[-1]), xytext=(-10, 13), textcoords="offset points", ha="right", fontsize=10, color=color, weight="bold")
    fig.text(0.075, 0.075, "Only 11 measured checkpoints are shown; no synthetic or interpolated evaluation samples.", fontsize=9, color="#536275")
    fig.text(0.075, 0.035, "Development data supports checkpoint selection. These curves are not independent BFCL benchmark results.", fontsize=9, color="#536275")
    save_figure(fig, output, "training-metrics")


def write_csv(output: Path, name: str, rows: list[dict]) -> None:
    with (output / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "training-history.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    history = json.loads(args.input.read_text(encoding="utf-8-sig"))
    validate(history)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": "#31445c", "xtick.color": "#536275", "ytick.color": "#536275", "svg.hashsalt": "dev-1.0-measured-training-history", "svg.fonttype": "none", "path.simplify": False})
    write_csv(args.output_dir, "training-history.csv", history["training_records"])
    write_csv(args.output_dir, "development-history.csv", history["development_records"])
    plot_loss(history, args.output_dir)
    plot_metrics(history, args.output_dir)
    names = ["training-history.csv", "development-history.csv", "loss-curves.png", "loss-curves.svg", "training-metrics.png", "training-metrics.svg"]
    manifest = {"schema": "dev-1.0-training-plot-provenance-1", "model": "dev-1.0", "input_sha256": sha256(args.input), "source_events_sha256": history["source_events_sha256"], "plot_script_sha256": sha256(Path(__file__)), "matplotlib_version": matplotlib.__version__, "training_updates": len(history["training_records"]), "development_measurements": len(history["development_records"]), "smoothing": {"training_loss_only": "trailing arithmetic mean over 50 actual optimizer updates", "window": WINDOW, "first_smoothed_step": WINDOW, "no_padding_or_extrapolation": True}, "evaluation_interpolation": False, "files": {name: {"bytes": (args.output_dir/name).stat().st_size, "sha256": sha256(args.output_dir/name)} for name in names}}
    (args.output_dir / "training-plot-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "rendered", "training_updates": len(history["training_records"]), "development_measurements": len(history["development_records"]), "files": names}))


if __name__ == "__main__":
    main()
