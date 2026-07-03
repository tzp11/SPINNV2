"""Calibrate risk model from Monte Carlo fault injection results.

Two-phase adaptive ILP: use pilot MC observations to replace the initial
critical_probability=1.0 assumption with measured per-node failure rates,
and replace the conservative range_guard coverage=0.5 with observed values.

Smoothing strategies
--------------------
wilson (default)
    Wilson score CI upper bound for every node.  Conservative — inflates CP
    for nodes with few events, causing ILP to waste budget on cheap nodes
    with high apparent risk/cost efficiency but zero actual critical failures.
observed
    Wilson upper bound for nodes with ≥1 observed CF; CP=0 for others.
    Precise but can plateau if unobserved CF nodes exist in validation.
observed-floor
    Wilson upper bound for CF nodes; min(CF_wilson)×0.1 for others.
    Recommended: combines CF-node precision with fallback coverage for
    unobserved nodes, avoiding both overestimation and hard zeroing.
laplace
    Laplace (add-1) smoothing.  Produces uniform non-zero estimates but
    does not distinguish observed-CF from observed-non-CF nodes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson_upper(successes: int, trials: int, z: float = 1.96) -> float:
    """Wilson score interval upper bound (95% by default)."""
    if trials == 0:
        return 1.0
    p_hat = successes / trials
    denom = 1 + z * z / trials
    centre = p_hat + z * z / (2 * trials)
    spread = z * math.sqrt(p_hat * (1 - p_hat) / trials + z * z / (4 * trials * trials))
    return min(1.0, (centre + spread) / denom)


def calibrate_candidates(
    mc_observations: list[dict],
    candidates: list[dict],
    *,
    smoothing: str = "wilson",
) -> tuple[list[dict], dict]:
    """Return calibrated candidates and calibration metadata.

    Parameters
    ----------
    mc_observations : list[dict]
        The ``observations`` array from an MC result JSON.
    candidates : list[dict]
        Original candidate mode list (heterogeneous).
    smoothing : str
        One of ``"wilson"``, ``"observed"``, ``"observed-floor"``, or
        ``"laplace"``.  See module docstring for trade-offs.

    Returns
    -------
    calibrated : list[dict]
        New candidates with updated ``risk_reduction`` and ``critical_probability``.
    metadata : dict
        Per-node calibration statistics including smoothing used, event counts,
        per-node CF rates, and global range-guard coverage.
    """
    node_events: dict[int, int] = {}
    node_cf: dict[int, int] = {}
    node_rg_events: dict[int, int] = {}
    node_rg_detected_cf: dict[int, int] = {}

    for obs in mc_observations:
        nid = obs["node_id"]
        node_events[nid] = node_events.get(nid, 0) + 1
        if obs["unprotected_critical_failure"]:
            node_cf[nid] = node_cf.get(nid, 0) + 1

        if obs.get("protection_mode") == "range_guard_rerun":
            node_rg_events[nid] = node_rg_events.get(nid, 0) + 1
            if obs["unprotected_critical_failure"] and not obs["protected_critical_failure"]:
                node_rg_detected_cf[nid] = node_rg_detected_cf.get(nid, 0) + 1

    total_bytes = 0
    node_bytes: dict[int, int] = {}
    for c in candidates:
        if c["mode"] == "dmr_compare_rerun":
            nbytes = c["extra_memory_bytes"] // 2
            node_bytes[c["node_id"]] = nbytes
            total_bytes += nbytes

    calibrated_cp: dict[int, float] = {}
    cf_wilson_values = []
    for nid in node_bytes:
        events = node_events.get(nid, 0)
        cf = node_cf.get(nid, 0)
        if cf > 0:
            cf_wilson_values.append(wilson_upper(cf, events))

    for nid in node_bytes:
        events = node_events.get(nid, 0)
        cf = node_cf.get(nid, 0)
        if smoothing == "wilson":
            calibrated_cp[nid] = wilson_upper(cf, events)
        elif smoothing == "observed":
            calibrated_cp[nid] = wilson_upper(cf, events) if cf > 0 else 0.0
        elif smoothing == "observed-floor":
            if cf > 0:
                calibrated_cp[nid] = wilson_upper(cf, events)
            else:
                floor = min(cf_wilson_values) * 0.1 if cf_wilson_values else 0.01
                calibrated_cp[nid] = floor
        else:
            calibrated_cp[nid] = (cf + 1) / (events + 2)

    total_rg_cf = sum(node_cf.get(nid, 0) for nid in node_rg_events)
    total_rg_mitigated = sum(node_rg_detected_cf.values())
    global_rg_coverage = total_rg_mitigated / total_rg_cf if total_rg_cf > 0 else 0.5

    per_node_rg_cov: dict[int, float] = {}
    for nid in node_bytes:
        nid_cf = node_cf.get(nid, 0)
        nid_rg_mit = node_rg_detected_cf.get(nid, 0)
        if nid_cf > 0 and nid in node_rg_events:
            per_node_rg_cov[nid] = nid_rg_mit / nid_cf
        else:
            per_node_rg_cov[nid] = global_rg_coverage

    calibrated = []
    for c in [dict(c) for c in candidates]:
        nid = c["node_id"]
        cp = calibrated_cp.get(nid, 1.0)
        fault_prior = node_bytes.get(nid, 0) / total_bytes if total_bytes > 0 else 0.0
        c["critical_probability"] = cp
        if c["mode"] == "dmr_compare_rerun":
            c["risk_reduction"] = fault_prior * cp
        elif c["mode"] == "range_guard_rerun":
            c["risk_reduction"] = fault_prior * cp * per_node_rg_cov.get(nid, global_rg_coverage)
        calibrated.append(c)

    per_node = {}
    for nid in sorted(node_bytes):
        events = node_events.get(nid, 0)
        cf = node_cf.get(nid, 0)
        per_node[str(nid)] = {
            "events": events,
            "critical_failures": cf,
            "raw_critical_probability": cf / events if events > 0 else 0.0,
            "calibrated_critical_probability": calibrated_cp.get(nid, 1.0),
            "fault_prior": node_bytes.get(nid, 0) / total_bytes if total_bytes > 0 else 0.0,
            "rg_coverage": per_node_rg_cov.get(nid, global_rg_coverage),
        }

    metadata = {
        "smoothing": smoothing,
        "total_events": sum(node_events.values()),
        "total_critical_failures": sum(node_cf.values()),
        "global_rg_coverage": global_rg_coverage,
        "rg_cf_total": total_rg_cf,
        "rg_cf_mitigated": total_rg_mitigated,
        "per_node_rg_coverage": per_node_rg_cov,
        "per_node": per_node,
    }
    return calibrated, metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate risk model from MC fault injection results."
    )
    parser.add_argument("--mc-results", required=True, nargs="+",
                        help="One or more MC result JSON files (observations are merged).")
    parser.add_argument("--candidates", required=True,
                        help="Original candidates_heterogeneous.json.")
    parser.add_argument("--smoothing", choices=("wilson", "laplace", "observed", "observed-floor"), default="wilson")
    parser.add_argument("--output", required=True,
                        help="Output calibrated candidates JSON.")
    parser.add_argument("--output-metadata",
                        help="Optional calibration metadata JSON.")
    args = parser.parse_args()

    all_observations = []
    for mc_path in args.mc_results:
        mc = json.loads(Path(mc_path).read_text(encoding="utf-8"))
        all_observations.extend(mc["observations"])

    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))

    calibrated, metadata = calibrate_candidates(
        all_observations, candidates, smoothing=args.smoothing
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calibrated, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_metadata:
        meta_out = Path(args.output_metadata)
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        meta_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    total_risk = sum(c["risk_reduction"] for c in calibrated if c["mode"] == "dmr_compare_rerun")
    print(json.dumps({
        "calibrated_candidates": len(calibrated),
        "total_events_used": metadata["total_events"],
        "total_critical_failures": metadata["total_critical_failures"],
        "global_rg_coverage": metadata["global_rg_coverage"],
        "total_dmr_risk_reduction": total_risk,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
