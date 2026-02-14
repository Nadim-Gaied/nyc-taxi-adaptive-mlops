"""Post-hoc reporting for a completed locked historical replay."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_LABELS = {
    "never_retrain": "Never retrain",
    "fixed_periodic_28d": "Fixed 28-day",
    "performance_triggered": "Performance-triggered",
    "adaptive_reuse_and_retrain": "Adaptive reuse + retrain",
}
_COLORS = {
    "never_retrain": "#616161",
    "fixed_periodic_28d": "#1565c0",
    "performance_triggered": "#ef6c00",
    "adaptive_reuse_and_retrain": "#2e7d32",
}


def _label(strategy: str) -> str:
    return _LABELS.get(strategy, strategy.replace("_", " ").title())


def _write_overall_table(comparison: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    never = comparison.loc[comparison["strategy"].eq("never_retrain")]
    if len(never) != 1:
        raise ValueError("comparison must contain exactly one never_retrain row")
    never_wape = float(never.iloc[0]["wape"])
    table = comparison.copy()
    table["strategy_label"] = table["strategy"].map(_label)
    table["wape_percent"] = 100 * table["wape"]
    table["wape_delta_vs_never_percentage_points"] = 100 * (table["wape"] - never_wape)
    table["relative_wape_improvement_vs_never_percent"] = (
        100 * (never_wape - table["wape"]) / never_wape
    )
    table["total_compute_seconds"] = table["training_seconds"] + table["inference_seconds"]
    table["total_compute_hours"] = table["total_compute_seconds"] / 3600
    table = table.sort_values("wape").reset_index(drop=True)
    table.to_csv(output_dir / "overall_strategy_metrics.csv", index=False)
    return table


def _write_annual_table(replay_dir: Path, strategies: list[str], output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy in strategies:
        weekly = pd.read_csv(
            replay_dir / strategy / "weekly_metrics.csv", parse_dates=["week_start"]
        )
        weekly["year"] = weekly["week_start"].dt.year
        weekly["absolute_error"] = weekly["wape"] * weekly["actual_pickups"]
        weekly["signed_error"] = weekly["predicted_pickups"] - weekly["actual_pickups"]
        for year, frame in weekly.groupby("year", sort=True):
            observations = int(frame["observations"].sum())
            actual = float(frame["actual_pickups"].sum())
            rows.append(
                {
                    "strategy": strategy,
                    "strategy_label": _label(strategy),
                    "year": int(year),
                    "weeks": len(frame),
                    "observations": observations,
                    "actual_pickups": actual,
                    "wape": float(frame["absolute_error"].sum() / actual),
                    "mae": float(frame["mae"].mul(frame["observations"]).sum() / observations),
                    "mean_bias": float(frame["signed_error"].sum() / observations),
                }
            )
    annual = pd.DataFrame(rows)
    annual.to_csv(output_dir / "annual_strategy_metrics.csv", index=False)
    return annual


def _write_tier_table(replay_dir: Path, strategies: list[str], output_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for strategy in strategies:
        frame = pd.read_csv(replay_dir / strategy / "tier_metrics.csv")
        frame.insert(0, "strategy", strategy)
        frame.insert(1, "strategy_label", _label(strategy))
        frames.append(frame)
    tiers = pd.concat(frames, ignore_index=True)
    tiers.to_csv(output_dir / "tier_strategy_metrics.csv", index=False)
    return tiers


def _plot_weekly_wape(replay_dir: Path, strategies: list[str], figure_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    maximum_smoothed_wape = 0.0
    for strategy in strategies:
        weekly = pd.read_csv(
            replay_dir / strategy / "weekly_metrics.csv", parse_dates=["week_start"]
        ).sort_values("week_start")
        smooth = weekly["wape"].rolling(4, min_periods=1).mean()
        maximum_smoothed_wape = max(maximum_smoothed_wape, float(smooth.max()))
        ax.plot(
            weekly["week_start"],
            smooth,
            label=_label(strategy),
            color=_COLORS.get(strategy),
            linewidth=1.7,
        )
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"), color="#b71c1c", alpha=0.08)
    ax.text(
        pd.Timestamp("2020-03-08"),
        maximum_smoothed_wape * 0.96,
        "COVID demand shock",
        color="#8e0000",
        va="top",
    )
    ax.set(
        title="Locked replay: four-week rolling WAPE",
        xlabel="Evaluation week",
        ylabel="WAPE",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    path = figure_dir / "weekly_wape.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_annual_wape(annual: pd.DataFrame, strategies: list[str], figure_dir: Path) -> Path:
    years = sorted(annual["year"].unique())
    x = np.arange(len(years))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for index, strategy in enumerate(strategies):
        frame = annual.loc[annual["strategy"].eq(strategy)].set_index("year").loc[years]
        offset = (index - (len(strategies) - 1) / 2) * width
        ax.bar(
            x + offset,
            100 * frame["wape"],
            width,
            label=_label(strategy),
            color=_COLORS.get(strategy),
        )
    ax.set_xticks(x, years)
    ax.set(
        title="Annual forecasting error by lifecycle strategy",
        xlabel="Year",
        ylabel="WAPE (%)",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    path = figure_dir / "annual_wape.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_accuracy_cost(overall: pd.DataFrame, figure_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for row in overall.itertuples(index=False):
        ax.scatter(
            row.total_compute_hours,
            100 * row.wape,
            s=75,
            color=_COLORS.get(row.strategy),
        )
        align_right = row.total_compute_hours == overall["total_compute_hours"].max()
        ax.annotate(
            _label(row.strategy),
            (row.total_compute_hours, 100 * row.wape),
            xytext=((-6 if align_right else 6), 6),
            textcoords="offset points",
            ha="right" if align_right else "left",
        )
    ax.set(
        title="Forecast quality versus measured replay compute",
        xlabel="Training + inference time (hours)",
        ylabel="WAPE (%) — lower is better",
    )
    ax.grid(alpha=0.25)
    ax.margins(x=0.05, y=0.08)
    path = figure_dir / "accuracy_vs_compute.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_operations(overall: pd.DataFrame, figure_dir: Path) -> Path:
    ordered = overall.sort_values("wape")
    x = np.arange(len(ordered))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.bar(x - width, ordered["training_jobs"], width, label="Training jobs", color="#1565c0")
    ax.bar(x, ordered["model_switches"], width, label="Model switches", color="#ef6c00")
    ax.bar(x + width, ordered["degraded_weeks"], width, label="Degraded weeks", color="#8e24aa")
    ax.set_xticks(x, ordered["strategy_label"], rotation=15, ha="right")
    ax.set(
        title="Operational activity and time in degraded state",
        xlabel="Lifecycle strategy",
        ylabel="Count",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    path = figure_dir / "operational_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_tier_wape(tiers: pd.DataFrame, strategies: list[str], figure_dir: Path) -> Path:
    tier_order = [tier for tier in ("low", "medium", "high") if tier in set(tiers["demand_tier"])]
    x = np.arange(len(tier_order))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for index, strategy in enumerate(strategies):
        frame = tiers.loc[tiers["strategy"].eq(strategy)].set_index("demand_tier").loc[tier_order]
        offset = (index - (len(strategies) - 1) / 2) * width
        ax.bar(
            x + offset,
            100 * frame["wape"],
            width,
            label=_label(strategy),
            color=_COLORS.get(strategy),
        )
    ax.set_xticks(x, [tier.title() for tier in tier_order])
    ax.set(
        title="Error by training-defined demand tier",
        xlabel="Zone demand tier",
        ylabel="WAPE (%)",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    path = figure_dir / "tier_wape.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_locked_replay_report(
    replay_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Create deterministic tables and figures from a completed locked replay."""

    replay_dir = Path(replay_dir)
    output_dir = Path(output_dir) if output_dir is not None else replay_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(replay_dir / "strategy_comparison.csv")
    if comparison.empty or comparison["observations"].nunique() != 1:
        raise ValueError("strategies must contain one common, non-empty target set")
    strategies = comparison["strategy"].tolist()
    for strategy in strategies:
        for name in ("summary.json", "weekly_metrics.csv", "tier_metrics.csv"):
            if not (replay_dir / strategy / name).is_file():
                raise FileNotFoundError(replay_dir / strategy / name)

    overall = _write_overall_table(comparison, output_dir)
    annual = _write_annual_table(replay_dir, strategies, output_dir)
    tiers = _write_tier_table(replay_dir, strategies, output_dir)
    figures = [
        _plot_weekly_wape(replay_dir, strategies, figure_dir),
        _plot_annual_wape(annual, strategies, figure_dir),
        _plot_accuracy_cost(overall, figure_dir),
        _plot_operations(overall, figure_dir),
        _plot_tier_wape(tiers, strategies, figure_dir),
    ]

    best = overall.iloc[0]
    never = overall.loc[overall["strategy"].eq("never_retrain")].iloc[0]
    fixed = overall.loc[overall["strategy"].eq("fixed_periodic_28d")].iloc[0]
    adaptive = overall.loc[overall["strategy"].eq("adaptive_reuse_and_retrain")].iloc[0]
    summary: dict[str, object] = {
        "status": "complete",
        "strategies": strategies,
        "observations_per_strategy": int(comparison["observations"].iloc[0]),
        "best_overall_wape_strategy": str(best["strategy"]),
        "best_overall_wape": float(best["wape"]),
        "best_relative_wape_improvement_vs_never_percent": float(
            best["relative_wape_improvement_vs_never_percent"]
        ),
        "fixed_training_jobs": int(fixed["training_jobs"]),
        "fixed_model_switches": int(fixed["model_switches"]),
        "fixed_degraded_weeks": int(fixed["degraded_weeks"]),
        "never_degraded_weeks": int(never["degraded_weeks"]),
        "adaptive_registered_models_evaluated": int(adaptive["registered_models_evaluated"]),
        "adaptive_wape_delta_vs_fixed_percentage_points": float(
            100 * (adaptive["wape"] - fixed["wape"])
        ),
        "figures": [str(path) for path in figures],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary
