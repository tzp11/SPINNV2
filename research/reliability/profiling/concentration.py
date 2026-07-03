"""Quantify whether task-level failures are concentrated in a few runtime objects."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConcentrationSummary:
    scope: str
    candidate_count: int
    critical_failures: int
    gini_critical_failures: float
    task_risk_auc: float
    exposure_proxy_auc: float
    top_10_percent_coverage: float
    top_20_percent_coverage: float
    top_25_percent_coverage: float
    candidates_for_50_percent_coverage: int
    fraction_for_50_percent_coverage: float
    candidates_for_80_percent_coverage: int
    fraction_for_80_percent_coverage: float


def gini(values: np.ndarray) -> float:
    """Return the Gini coefficient for non-negative failure counts."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0 or np.all(data == 0):
        return 0.0
    if np.any(data < 0):
        raise ValueError("Gini input must be non-negative")
    ordered = np.sort(data)
    index = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * ordered) / np.sum(ordered) - ordered.size - 1.0) / ordered.size)


def build_rank_curve(frame: pd.DataFrame, *, rank_by: str, scope: str, ranking: str) -> pd.DataFrame:
    """Build a cumulative critical-failure coverage curve for one ranking rule."""
    required = {"candidate", "critical_failures", rank_by}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing concentration columns: {sorted(missing)}")
    ordered = frame.sort_values([rank_by, "critical_failures"], ascending=[False, False]).reset_index(drop=True).copy()
    if ordered.empty:
        return pd.DataFrame(
            columns=["scope", "ranking", "rank", "candidate", "candidate_fraction", "cumulative_failure_coverage"]
        )
    total_failures = float(ordered["critical_failures"].sum())
    ordered["scope"] = scope
    ordered["ranking"] = ranking
    ordered["rank"] = np.arange(1, len(ordered) + 1)
    ordered["candidate_fraction"] = ordered["rank"] / len(ordered)
    if total_failures > 0.0:
        ordered["cumulative_failure_coverage"] = ordered["critical_failures"].cumsum() / total_failures
    else:
        ordered["cumulative_failure_coverage"] = 0.0
    return ordered[
        [
            "scope",
            "ranking",
            "rank",
            "candidate",
            "critical_failures",
            "candidate_fraction",
            "cumulative_failure_coverage",
        ]
    ]


def summarize_concentration(frame: pd.DataFrame, *, scope: str) -> tuple[ConcentrationSummary, pd.DataFrame]:
    """Return primary metrics and risk/exposure ranking curves for one candidate scope."""
    if frame.empty:
        raise ValueError(f"cannot summarize empty scope: {scope}")
    task_curve = build_rank_curve(frame, rank_by="risk", scope=scope, ranking="task_risk")
    exposure_curve = build_rank_curve(frame, rank_by="activation_bytes", scope=scope, ranking="activation_bytes")
    curves = pd.concat([task_curve, exposure_curve], ignore_index=True)
    failures = int(frame["critical_failures"].sum())

    def coverage_at(fraction: float) -> float:
        count = max(1, math.ceil(len(task_curve) * fraction))
        return float(task_curve.iloc[count - 1]["cumulative_failure_coverage"])

    def candidates_for(target: float) -> tuple[int, float]:
        meets = task_curve[task_curve["cumulative_failure_coverage"] >= target]
        if meets.empty:
            return len(task_curve), 1.0
        count = int(meets.iloc[0]["rank"])
        return count, count / len(task_curve)

    count_50, fraction_50 = candidates_for(0.50)
    count_80, fraction_80 = candidates_for(0.80)
    x = np.concatenate(([0.0], task_curve["candidate_fraction"].to_numpy()))
    task_y = np.concatenate(([0.0], task_curve["cumulative_failure_coverage"].to_numpy()))
    exposure_y = np.concatenate(([0.0], exposure_curve["cumulative_failure_coverage"].to_numpy()))
    summary = ConcentrationSummary(
        scope=scope,
        candidate_count=len(frame),
        critical_failures=failures,
        gini_critical_failures=gini(frame["critical_failures"].to_numpy()),
        task_risk_auc=float(np.trapezoid(task_y, x)),
        exposure_proxy_auc=float(np.trapezoid(exposure_y, x)),
        top_10_percent_coverage=coverage_at(0.10),
        top_20_percent_coverage=coverage_at(0.20),
        top_25_percent_coverage=coverage_at(0.25),
        candidates_for_50_percent_coverage=count_50,
        fraction_for_50_percent_coverage=fraction_50,
        candidates_for_80_percent_coverage=count_80,
        fraction_for_80_percent_coverage=fraction_80,
    )
    return summary, curves
