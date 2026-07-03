"""Estimate task-level runtime-object risk and uncertainty."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from collections import defaultdict


@dataclass(frozen=True)
class InjectionObservation:
    node_id: int
    tensor_id: int
    activation_bytes: int
    critical_failure: bool
    severity: float


@dataclass(frozen=True)
class RiskRecord:
    node_id: int
    tensor_id: int
    injections: int
    critical_failures: int
    critical_probability: float
    ci_low: float
    ci_high: float
    mean_severity: float
    activation_bytes: int
    exposure_ratio: float
    risk: float


def build_risk_profile(
    observations: list[InjectionObservation],
    *,
    critical_weight: float = 0.65,
    severity_weight: float = 0.25,
    exposure_weight: float = 0.10,
) -> list[RiskRecord]:
    if not observations:
        raise ValueError("observations must not be empty")
    if not math.isclose(critical_weight + severity_weight + exposure_weight, 1.0, abs_tol=1e-9):
        raise ValueError("risk weights must sum to 1")
    grouped: dict[tuple[int, int], list[InjectionObservation]] = defaultdict(list)
    for observation in observations:
        grouped[(observation.node_id, observation.tensor_id)].append(observation)
    total_bytes = sum(items[0].activation_bytes for items in grouped.values())
    results = []
    for (node_id, tensor_id), items in grouped.items():
        injections = len(items)
        failures = sum(int(item.critical_failure) for item in items)
        probability = failures / injections
        low, high = wilson_interval(failures, injections)
        severity = sum(item.severity for item in items) / injections
        exposure = items[0].activation_bytes / total_bytes if total_bytes else 0.0
        risk = critical_weight * probability + severity_weight * severity + exposure_weight * exposure
        results.append(
            RiskRecord(
                node_id=node_id,
                tensor_id=tensor_id,
                injections=injections,
                critical_failures=failures,
                critical_probability=probability,
                ci_low=low,
                ci_high=high,
                mean_severity=severity,
                activation_bytes=items[0].activation_bytes,
                exposure_ratio=exposure,
                risk=risk,
            )
        )
    return sorted(results, key=lambda record: (-record.risk, record.node_id))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def write_risk_profile(records: list[RiskRecord], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8")
