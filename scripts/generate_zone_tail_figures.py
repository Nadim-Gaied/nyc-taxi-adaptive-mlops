"""Generate reproducible figures explaining aggregate and zone-tail metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from taxi_mlops.baselines import daily_weekly_blend_predictions, weekly_seasonal_predictions
from taxi_mlops.evaluation import regression_metrics, zone_regression_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data/processed/zone_hour_2019-01.parquet"
ZONE_LOOKUP = PROJECT_ROOT / "data/raw/tlc/metadata/taxi_zone_lookup.csv"
OUTPUT_DIR = PROJECT_ROOT / "figures/zone_tail"
EVALUATION_START = pd.Timestamp("2019-01-08T05:00:00Z")

BASELINE_LABELS = {
    "seasonal_lag_1h": "Previous hour",
    "seasonal_lag_24h": "Previous day",
    "seasonal_lag_168h": "Previous week",
    "daily_weekly_blend_0.50_0.50": "Daily/weekly blend",
}


def _candidate_predictions(data: pd.DataFrame) -> list[pd.DataFrame]:
    return [
        weekly_seasonal_predictions(data, lag_hours=1),
        weekly_seasonal_predictions(data, lag_hours=24),
        weekly_seasonal_predictions(data, lag_hours=168),
        daily_weekly_blend_predictions(data, daily_weight=0.5),
    ]


def _load_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_parquet(DATASET)
    summary_rows: list[dict[str, object]] = []
    zone_frames: list[pd.DataFrame] = []
    for candidate in _candidate_predictions(data):
        evaluation = candidate.loc[candidate["pickup_hour"] >= EVALUATION_START].copy()
        model_id = str(candidate["model_id"].iloc[0])
        summary = regression_metrics(evaluation)
        summary["model_id"] = model_id
        summary["baseline"] = BASELINE_LABELS[model_id]
        summary_rows.append(summary)

        zones = zone_regression_metrics(evaluation)
        zones["model_id"] = model_id
        zones["baseline"] = BASELINE_LABELS[model_id]
        zone_frames.append(zones)

    summaries = pd.DataFrame(summary_rows)
    zone_metrics = pd.concat(zone_frames, ignore_index=True)
    return summaries, zone_metrics


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)


def _plot_baseline_error_profile(summaries: pd.DataFrame) -> None:
    order = ["Daily/weekly blend", "Previous week", "Previous hour", "Previous day"]
    frame = summaries.set_index("baseline").loc[order].reset_index()
    metrics = [
        ("zone_mae_p50", "Median zone", "o", "#4C78A8"),
        ("mae", "Overall", "s", "#59A14F"),
        ("zone_mae_p95", "95th percentile zone", "^", "#F28E2B"),
        ("zone_mae_max", "Worst zone", "D", "#E15759"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    y = np.arange(len(frame))
    for column, label, marker, color in metrics:
        ax.scatter(frame[column], y, label=label, marker=marker, s=65, color=color, zorder=3)
    for index, row in frame.iterrows():
        ax.plot(
            [row["zone_mae_p50"], row["zone_mae_max"]],
            [index, index],
            color="#B8B8B8",
            linewidth=1.2,
            zorder=1,
        )

    ax.set_xscale("log")
    ax.set_yticks(y, frame["baseline"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error (pickups per zone-hour, log scale)")
    ax.set_ylabel("Baseline")
    ax.set_title("A good average can coexist with a severe zone-level tail", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=4, loc="lower center")
    _style_axis(ax)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUTPUT_DIR / "baseline-error-profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_best_baseline_tail(zone_metrics: pd.DataFrame, summaries: pd.DataFrame) -> None:
    best_id = str(summaries.sort_values("mae").iloc[0]["model_id"])
    best = zone_metrics.loc[zone_metrics["model_id"] == best_id].sort_values("mae")
    percentile = np.linspace(0, 100, len(best))
    overall_mae = float(summaries.loc[summaries["model_id"] == best_id, "mae"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(percentile, best["mae"], color="#4C78A8", linewidth=2)
    ax.fill_between(percentile, best["mae"], color="#4C78A8", alpha=0.12)
    ax.axhline(
        overall_mae,
        color="#59A14F",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall MAE = {overall_mae:.2f}",
    )
    for quantile, color in [(50, "#76B7B2"), (90, "#F28E2B"), (95, "#E15759")]:
        value = float(best["mae"].quantile(quantile / 100))
        ax.scatter([quantile], [value], color=color, s=55, zorder=3)
        ax.annotate(
            f"p{quantile}: {value:.2f}",
            xy=(quantile, value),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
        )

    ax.set_xlabel("Zone percentile, ordered from lowest to highest MAE")
    ax.set_ylabel("Zone MAE (pickups per hour)")
    ax.set_title("Most zones are easy; a small tail is much harder", loc="left")
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "best-baseline-zone-tail.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_demand_vs_error(zone_metrics: pd.DataFrame, summaries: pd.DataFrame) -> None:
    best_id = str(summaries.sort_values("mae").iloc[0]["model_id"])
    best = zone_metrics.loc[zone_metrics["model_id"] == best_id].copy()
    lookup = pd.read_csv(ZONE_LOOKUP)[["LocationID", "Borough", "Zone"]]
    best = best.merge(lookup, left_on="pickup_zone_id", right_on="LocationID", how="left")
    best = best.loc[best["actual_pickups"].gt(0) & best["wape"].notna()].copy()
    best["wape_percent"] = best["wape"] * 100

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.scatter(
        best["actual_pickups"],
        best["wape_percent"],
        s=30,
        color="#4C78A8",
        alpha=0.72,
        edgecolors="none",
    )
    ax.set_xscale("log")
    highest_error = best.nlargest(6, "wape_percent")
    offsets = [(8, 9), (8, 4), (8, -3), (8, -10), (-8, 5), (-8, -10)]
    for row, offset in zip(highest_error.itertuples(index=False), offsets, strict=True):
        label = str(row.Zone) if pd.notna(row.Zone) else f"Zone {row.pickup_zone_id}"
        ax.annotate(
            label,
            xy=(row.actual_pickups, row.wape_percent),
            xytext=offset,
            textcoords="offset points",
            ha="left" if offset[0] > 0 else "right",
            fontsize=8,
        )

    ax.set_xlabel("Actual pickups during evaluation period (log scale)")
    ax.set_ylabel("Zone WAPE (%)")
    ax.set_title("Relative error is highest in low-demand zones", loc="left")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "zone-demand-vs-error.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_demand_vs_wape(zone_metrics: pd.DataFrame, summaries: pd.DataFrame) -> None:
    best_id = str(summaries.sort_values("mae").iloc[0]["model_id"])
    best = zone_metrics.loc[zone_metrics["model_id"] == best_id].copy()
    lookup = pd.read_csv(ZONE_LOOKUP)[["LocationID", "Borough", "Zone"]]
    best = best.merge(lookup, left_on="pickup_zone_id", right_on="LocationID", how="left")
    excluded_zero_demand = int(best["actual_pickups"].eq(0).sum())
    best = best.loc[best["actual_pickups"].gt(0) & best["wape"].notna()].copy()
    best["wape_percent"] = best["wape"] * 100

    best["demand_bin"] = pd.qcut(best["actual_pickups"], q=8, duplicates="drop")
    binned = (
        best.groupby("demand_bin", observed=True, as_index=False)
        .agg(actual_pickups=("actual_pickups", "median"), wape_percent=("wape_percent", "median"))
        .sort_values("actual_pickups")
    )

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.scatter(
        best["actual_pickups"],
        best["wape_percent"],
        s=30,
        color="#F28E2B",
        alpha=0.58,
        edgecolors="none",
        label="Taxi zone",
    )
    ax.plot(
        binned["actual_pickups"],
        binned["wape_percent"],
        color="#9C3D10",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Median within demand group",
    )
    overall_wape = float(
        summaries.loc[summaries["model_id"] == best_id, "wape"].iloc[0] * 100
    )
    ax.axhline(
        overall_wape,
        color="#4C78A8",
        linestyle="--",
        linewidth=1.4,
        label=f"Overall WAPE = {overall_wape:.1f}%",
    )
    mean_zone_wape = float(best["wape_percent"].mean())
    ax.axhline(
        mean_zone_wape,
        color="#59A14F",
        linestyle=":",
        linewidth=1.6,
        label=f"Mean zone WAPE = {mean_zone_wape:.1f}%",
    )
    highest_relative_error = best.nlargest(3, "wape_percent")
    for row in highest_relative_error.itertuples(index=False):
        label = str(row.Zone) if pd.notna(row.Zone) else f"Zone {row.pickup_zone_id}"
        ax.annotate(
            label,
            xy=(row.actual_pickups, row.wape_percent),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([20, 30, 40, 60, 100, 200, 300, 400])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Actual pickups during evaluation period (log scale)")
    ax.set_ylabel("Zone WAPE (log scale)")
    ax.set_title("Relative error falls as zone demand grows", loc="left")
    ax.text(
        0.02,
        0.04,
        f"{excluded_zero_demand} zero-demand zones excluded because WAPE is undefined",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    ax.legend(frameon=False, loc="upper right")
    _style_axis(ax)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "zone-demand-vs-wape.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries, zone_metrics = _load_metrics()
    summaries.to_csv(OUTPUT_DIR / "baseline-summary.csv", index=False)
    zone_metrics.to_csv(OUTPUT_DIR / "zone-metrics.csv", index=False)
    _plot_baseline_error_profile(summaries)
    _plot_best_baseline_tail(zone_metrics, summaries)
    _plot_demand_vs_error(zone_metrics, summaries)
    _plot_demand_vs_wape(zone_metrics, summaries)
    print(f"Wrote zone-tail figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
