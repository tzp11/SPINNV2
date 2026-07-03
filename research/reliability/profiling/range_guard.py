"""Calibration and coverage metrics for scalar activation range guards."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RangeBounds:
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class RangeGuardRecord:
    node_id: int
    runtime_node_id: int
    runtime_tensor_id: int
    module_name: str
    invocation_index: int
    lower_bound: float
    upper_bound: float
    critical_faults: int
    detected_critical_faults: int
    critical_coverage: float
    critical_coverage_ci_low: float
    critical_coverage_ci_high: float
    injected_faults: int
    detected_faults: int
    false_positive_samples: int
    clean_holdout_samples: int
    false_positive_rate: float
    false_positive_ci_low: float
    false_positive_ci_high: float

    def as_dict(self) -> dict:
        return asdict(self)


def calibrate_scalar_bounds(
    minima: list[float],
    maxima: list[float],
    *,
    margin_ratio: float = 0.05,
    minimum_margin: float = 1e-6,
) -> RangeBounds:
    if not minima or not maxima or len(minima) != len(maxima):
        raise ValueError("calibration extrema must be non-empty and paired")
    lower = float(np.min(minima))
    upper = float(np.max(maxima))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("clean calibration outputs must be finite")
    span = max(upper - lower, abs(lower), abs(upper), minimum_margin)
    margin = max(span * margin_ratio, minimum_margin)
    return RangeBounds(lower - margin, upper + margin)


def outside_scalar_bounds(values: np.ndarray, bounds: RangeBounds) -> bool:
    array = np.asarray(values)
    return bool(
        not np.all(np.isfinite(array))
        or np.any(array < bounds.lower_bound)
        or np.any(array > bounds.upper_bound)
    )
