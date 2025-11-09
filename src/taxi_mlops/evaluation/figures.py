"""Figures for chronological baseline and training-window comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def write_baseline_backtest_figures(
    weekly: pd.DataFrame, output_dir: str | Path
) -> list[Path]:
    """Plot time-varying baseline error and zone-tail amplification."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    for model_id, frame in weekly.groupby("model_id", sort=True):
        frame = frame.sort_values("evaluation_week")
        ax.plot(frame["evaluation_week"], frame["mae"], label=model_id, linewidth=1.5)
    positions = np.arange(0, weekly["evaluation_week"].nunique(), 8)
    labels = sorted(weekly["evaluation_week"].unique())
    ax.set_xticks(positions, [labels[index] for index in positions], rotation=45, ha="right")
    ax.set(
        title="Weekly chronological baseline error",
        xlabel="Evaluation week",
        ylabel="MAE (pickups per zone-hour)",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    path = output_dir / "weekly_baseline_mae.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    tail = (
        weekly.groupby("model_id", as_index=False)
        .agg(overall_mae=("mae", "mean"), p95_zone_mae=("zone_mae_p95", "mean"))
        .sort_values("overall_mae")
    )
    x = np.arange(len(tail))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x - width / 2, tail["overall_mae"], width, label="Overall MAE", color="#1565c0")
    ax.bar(
        x + width / 2,
        tail["p95_zone_mae"],
        width,
        label="95th-percentile zone MAE",
        color="#ef6c00",
    )
    ax.set_xticks(x, tail["model_id"], rotation=20, ha="right")
    ax.set(
        title="Average error can hide the zone tail",
        xlabel="Baseline",
        ylabel="MAE (pickups per zone-hour)",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    path = output_dir / "baseline_zone_tail.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    return written


def write_window_experiment_figures(
    results: pd.DataFrame, output_dir: str | Path
) -> list[Path]:
    """Plot accuracy stability and compute trade-offs across history windows."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    ordered_cutoffs = sorted(results["cutoff"].unique())

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for window, frame in results.groupby("training_window", sort=True):
        frame = frame.sort_values("cutoff")
        ax.plot(frame["cutoff"], frame["mae"], marker="o", label=window, linewidth=1.5)
    ax.set_xticks(ordered_cutoffs, [item[:10] for item in ordered_cutoffs], rotation=45, ha="right")
    ax.set(
        title="Training-window performance across future periods",
        xlabel="Evaluation cutoff",
        ylabel="MAE (pickups per zone-hour)",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Training history", frameon=False, ncol=3)
    path = output_dir / "window_mae_by_cutoff.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    tradeoff = (
        results.groupby("training_window", as_index=False)
        .agg(mean_mae=("mae", "mean"), mean_training_seconds=("training_seconds", "mean"))
        .sort_values("mean_training_seconds")
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(tradeoff["mean_training_seconds"], tradeoff["mean_mae"], color="#2e7d32", s=55)
    for row in tradeoff.itertuples(index=False):
        ax.annotate(
            row.training_window,
            (row.mean_training_seconds, row.mean_mae),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set(
        title="Accuracy versus training cost",
        xlabel="Mean training time (seconds)",
        ylabel="Mean MAE (pickups per zone-hour)",
    )
    ax.grid(alpha=0.25)
    path = output_dir / "window_accuracy_cost.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    return written
