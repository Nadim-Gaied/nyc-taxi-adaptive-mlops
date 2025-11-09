"""Static scientific figures for development-period regime analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def write_regime_figures(
    monthly: pd.DataFrame,
    profiles: pd.DataFrame,
    similarity: pd.DataFrame,
    zone_summary: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Write a compact, reproducible figure set and return its paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    x = np.arange(len(monthly))
    ax.plot(x, monthly["mean_hourly_pickups"], color="#1565c0", linewidth=2)
    ax.scatter(x, monthly["mean_hourly_pickups"], color="#1565c0", s=18)
    ax.set(title="Monthly mean citywide hourly demand", ylabel="Mean pickups per hour")
    ax.set_xticks(x[::3], monthly["month"].iloc[::3], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / "monthly_mean_demand.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    profile_matrix = profiles.pivot(
        index="month", columns="hour_of_week", values="normalized_demand"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    image = ax.imshow(profile_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set(
        title="Normalized weekly demand profile by month",
        xlabel="Local hour of week (Monday 00:00 = 0)",
        ylabel="Month",
    )
    ax.set_yticks(np.arange(len(profile_matrix.index)), profile_matrix.index)
    ax.set_xticks(np.arange(0, 168, 24), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    fig.colorbar(image, ax=ax, label="Demand / monthly mean")
    path = output_dir / "monthly_weekly_profiles.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = ax.imshow(similarity, vmin=0.7, vmax=1.0, cmap="magma", interpolation="nearest")
    tick_positions = np.arange(0, len(similarity), 3)
    tick_labels = similarity.index[tick_positions]
    ax.set(
        title="Month-to-month weekly-profile similarity",
        xlabel="Comparison month",
        ylabel="Month",
    )
    ax.set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
    ax.set_yticks(tick_positions, tick_labels)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    path = output_dir / "monthly_profile_similarity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    ordered = zone_summary.sort_values("volume_rank")
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(
        ordered["volume_rank"],
        ordered["cumulative_demand_share"] * 100,
        color="#6a1b9a",
        linewidth=2,
    )
    ax.axhline(80, color="#c62828", linestyle="--", linewidth=1.2, label="80% of demand")
    ax.set(
        title="Demand concentration across taxi zones",
        xlabel="Zones ordered from highest to lowest demand",
        ylabel="Cumulative demand share (%)",
        xlim=(1, len(ordered)),
        ylim=(0, 101),
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    path = output_dir / "zone_demand_concentration.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    return written
