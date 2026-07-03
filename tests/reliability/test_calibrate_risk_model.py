"""Tests for risk model calibration from MC fault injection results."""

from __future__ import annotations

import pytest

from research.reliability.calibrate_risk_model import calibrate_candidates, wilson_upper


def _make_observations(node_cf_map: dict[int, tuple[int, int]], *, protection_mode: str = "range_guard_rerun") -> list[dict]:
    """Build synthetic MC observations.

    node_cf_map: {node_id: (total_events, critical_failures)}
    """
    obs = []
    seq = 0
    for nid, (events, cf) in node_cf_map.items():
        for i in range(events):
            is_cf = i < cf
            detected = is_cf and protection_mode == "range_guard_rerun"
            obs.append({
                "sequence": seq,
                "node_id": nid,
                "protection_mode": protection_mode,
                "unprotected_critical_failure": is_cf,
                "protected_critical_failure": False if detected else is_cf,
                "stats": {"detected_faults": int(detected), "recovered_faults": int(detected),
                          "unrecovered_faults": 0, "rerun_count": int(detected)},
            })
            seq += 1
    return obs


def _make_candidates(node_bytes: dict[int, int]) -> list[dict]:
    """Build minimal heterogeneous candidates."""
    total = sum(node_bytes.values())
    cands = []
    for nid, nbytes in node_bytes.items():
        fp = nbytes / total
        cands.append({
            "node_id": nid, "tensor_id": 0, "mode": "dmr_compare_rerun",
            "risk_reduction": fp, "latency_overhead_ms": 0.2,
            "extra_memory_bytes": nbytes * 2, "critical_probability": 1.0,
        })
        cands.append({
            "node_id": nid, "tensor_id": 0, "mode": "range_guard_rerun",
            "risk_reduction": fp * 0.5, "latency_overhead_ms": 0.05,
            "extra_memory_bytes": 0, "critical_probability": 1.0,
            "lower_bound": -1.0, "upper_bound": 1.0,
        })
    return cands


class TestWilsonUpper:
    def test_zero_trials(self):
        assert wilson_upper(0, 0) == 1.0

    def test_all_successes(self):
        assert wilson_upper(10, 10) == 1.0

    def test_no_successes(self):
        result = wilson_upper(0, 100)
        assert 0.0 < result < 0.05

    def test_monotone_in_successes(self):
        assert wilson_upper(1, 100) < wilson_upper(5, 100) < wilson_upper(10, 100)

    def test_shrinks_with_more_trials(self):
        assert wilson_upper(0, 10) > wilson_upper(0, 100) > wilson_upper(0, 1000)


class TestCalibrateCandidates:
    def test_updates_critical_probability(self):
        obs = _make_observations({0: (100, 5), 1: (100, 0)})
        cands = _make_candidates({0: 1000, 1: 1000})
        cal, meta = calibrate_candidates(obs, cands, smoothing="wilson")
        cp0 = next(c["critical_probability"] for c in cal if c["node_id"] == 0 and c["mode"] == "dmr_compare_rerun")
        cp1 = next(c["critical_probability"] for c in cal if c["node_id"] == 1 and c["mode"] == "dmr_compare_rerun")
        assert cp0 > cp1

    def test_risk_reduction_uses_calibrated_cp(self):
        obs = _make_observations({0: (100, 10), 1: (100, 0)})
        cands = _make_candidates({0: 1000, 1: 1000})
        cal, _ = calibrate_candidates(obs, cands, smoothing="wilson")
        rr0 = next(c["risk_reduction"] for c in cal if c["node_id"] == 0 and c["mode"] == "dmr_compare_rerun")
        rr1 = next(c["risk_reduction"] for c in cal if c["node_id"] == 1 and c["mode"] == "dmr_compare_rerun")
        assert rr0 > rr1

    def test_rg_risk_reduction_includes_coverage(self):
        obs = _make_observations({0: (100, 10)})
        cands = _make_candidates({0: 1000})
        cal, meta = calibrate_candidates(obs, cands, smoothing="wilson")
        rr_dmr = next(c["risk_reduction"] for c in cal if c["mode"] == "dmr_compare_rerun")
        rr_rg = next(c["risk_reduction"] for c in cal if c["mode"] == "range_guard_rerun")
        assert rr_rg == rr_dmr * meta["global_rg_coverage"]

    def test_rg_coverage_from_observations(self):
        obs = _make_observations({0: (100, 10)}, protection_mode="range_guard_rerun")
        cands = _make_candidates({0: 1000})
        _, meta = calibrate_candidates(obs, cands)
        assert meta["global_rg_coverage"] == 1.0

    def test_no_cf_rg_coverage_defaults(self):
        obs = _make_observations({0: (100, 0)}, protection_mode="range_guard_rerun")
        cands = _make_candidates({0: 1000})
        _, meta = calibrate_candidates(obs, cands)
        assert meta["global_rg_coverage"] == 0.5

    def test_laplace_smoothing(self):
        obs = _make_observations({0: (10, 0)})
        cands = _make_candidates({0: 1000})
        cal, _ = calibrate_candidates(obs, cands, smoothing="laplace")
        cp = next(c["critical_probability"] for c in cal if c["node_id"] == 0 and c["mode"] == "dmr_compare_rerun")
        assert abs(cp - 1 / 12) < 1e-9

    def test_metadata_per_node(self):
        obs = _make_observations({0: (50, 3), 1: (50, 0)})
        cands = _make_candidates({0: 1000, 1: 2000})
        _, meta = calibrate_candidates(obs, cands)
        assert meta["total_events"] == 100
        assert meta["total_critical_failures"] == 3
        assert meta["per_node"]["0"]["events"] == 50
        assert meta["per_node"]["0"]["critical_failures"] == 3
        assert meta["per_node"]["1"]["critical_failures"] == 0

    def test_preserves_extra_fields(self):
        cands = _make_candidates({0: 1000})
        cands[0]["macs_fraction"] = 0.5
        cands[0]["taylor_importance"] = 0.01
        obs = _make_observations({0: (100, 5)})
        cal, _ = calibrate_candidates(obs, cands)
        assert cal[0]["macs_fraction"] == 0.5
        assert cal[0]["taylor_importance"] == 0.01

    def test_unobserved_node_gets_conservative_cp(self):
        obs = _make_observations({0: (100, 5)})
        cands = _make_candidates({0: 1000, 1: 1000})
        cal, _ = calibrate_candidates(obs, cands, smoothing="wilson")
        cp1 = next(c["critical_probability"] for c in cal if c["node_id"] == 1 and c["mode"] == "dmr_compare_rerun")
        assert cp1 == 1.0

    def test_observed_smoothing_zeros_non_cf_nodes(self):
        obs = _make_observations({0: (100, 5), 1: (100, 0)})
        cands = _make_candidates({0: 1000, 1: 1000})
        cal, _ = calibrate_candidates(obs, cands, smoothing="observed")
        cp0 = next(c["critical_probability"] for c in cal if c["node_id"] == 0 and c["mode"] == "dmr_compare_rerun")
        cp1 = next(c["critical_probability"] for c in cal if c["node_id"] == 1 and c["mode"] == "dmr_compare_rerun")
        assert cp0 > 0
        assert cp1 == 0.0

    def test_observed_smoothing_uses_wilson_for_cf_nodes(self):
        obs = _make_observations({0: (100, 5)})
        cands = _make_candidates({0: 1000})
        cal_obs, _ = calibrate_candidates(obs, cands, smoothing="observed")
        cal_wil, _ = calibrate_candidates(obs, cands, smoothing="wilson")
        cp_obs = next(c["critical_probability"] for c in cal_obs if c["node_id"] == 0)
        cp_wil = next(c["critical_probability"] for c in cal_wil if c["node_id"] == 0)
        assert cp_obs == cp_wil

    def test_observed_floor_gives_small_nonzero_cp(self):
        obs = _make_observations({0: (100, 5), 1: (100, 0)})
        cands = _make_candidates({0: 1000, 1: 1000})
        cal, _ = calibrate_candidates(obs, cands, smoothing="observed-floor")
        cp0 = next(c["critical_probability"] for c in cal if c["node_id"] == 0 and c["mode"] == "dmr_compare_rerun")
        cp1 = next(c["critical_probability"] for c in cal if c["node_id"] == 1 and c["mode"] == "dmr_compare_rerun")
        assert cp0 > cp1 > 0
        assert cp1 < cp0 * 0.2
