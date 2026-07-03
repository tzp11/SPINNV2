"""Utilities for screening reliability-mechanism variants."""

from __future__ import annotations

from dataclasses import dataclass
import math


ADOPT_AS_MAIN = "adopt_as_main"
KEEP_AS_ABLATION = "keep_as_ablation"
REJECT = "reject"


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return max(0.0, (centre - margin) / denom)


@dataclass(frozen=True)
class ScreeningDecision:
    decision: str
    reason: str


def decide_guard_gated_dmr(*, latency_reduction: float, recovery_retention: float) -> ScreeningDecision:
    if latency_reduction >= 0.30 and recovery_retention >= 0.80:
        return ScreeningDecision(
            ADOPT_AS_MAIN,
            "latency reduction and recovery retention pass the Guard-Gated DMR thresholds",
        )
    if latency_reduction >= 0.15 and recovery_retention >= 0.60:
        return ScreeningDecision(
            KEEP_AS_ABLATION,
            "partial gain but below the main-method threshold",
        )
    return ScreeningDecision(REJECT, "latency saving or recovery retention is too low")


def decide_adaptive_range(*, false_positive_rate: float, coverage_gain: float) -> ScreeningDecision:
    if false_positive_rate <= 0.01 and coverage_gain >= 0.10:
        return ScreeningDecision(
            ADOPT_AS_MAIN,
            "coverage gain passes the threshold under the false-positive constraint",
        )
    if false_positive_rate <= 0.02 and coverage_gain > 0.0:
        return ScreeningDecision(KEEP_AS_ABLATION, "small gain or relaxed false-positive result")
    return ScreeningDecision(REJECT, "no verified coverage gain under the false-positive constraint")


def decide_task_utility_guard(
    *,
    false_positive_rate: float,
    reduction_gain: float,
    trigger_reduction: float,
) -> ScreeningDecision:
    if false_positive_rate <= 0.005 and (reduction_gain >= 0.10 or trigger_reduction >= 0.20):
        return ScreeningDecision(
            ADOPT_AS_MAIN,
            "task utility guard improves reduction or reduces rerun triggers under false-positive constraint",
        )
    if false_positive_rate <= 0.01 and (reduction_gain > 0.0 or trigger_reduction > 0.0):
        return ScreeningDecision(KEEP_AS_ABLATION, "measurable but below main-method threshold")
    return ScreeningDecision(REJECT, "no verified task-utility improvement")


def decide_robust_memory_ilp(*, risk_retention: float, memory_saving_ratio: float) -> ScreeningDecision:
    if risk_retention >= 0.95 and memory_saving_ratio >= 0.25:
        return ScreeningDecision(
            ADOPT_AS_MAIN,
            "risk retention and memory saving pass the robust low-memory optimization thresholds",
        )
    if risk_retention >= 0.90 and memory_saving_ratio > 0.0:
        return ScreeningDecision(KEEP_AS_ABLATION, "low-memory tradeoff exists but below main threshold")
    return ScreeningDecision(REJECT, "insufficient risk retention or memory saving")
