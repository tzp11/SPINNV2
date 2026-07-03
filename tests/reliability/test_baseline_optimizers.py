"""Unit tests for baseline protection-plan optimizers."""

from __future__ import annotations

import pytest

from research.reliability.optimizer.plan_optimizer import (
    CandidateMode,
    optimize_ahmadilivani,
    optimize_aspis,
    optimize_filr,
    optimize_greedy,
    optimize_ilp,
    optimize_ilp_static,
    optimize_ilp_v2,
    optimize_ranger,
    optimize_random_dmr,
    optimize_ruospo,
    optimize_topk_single_mode,
    optimize_uniform_dmr,
    optimize_vf_greedy,
)


def _make_candidates() -> list[CandidateMode]:
    """Heterogeneous candidates: 4 nodes, each with DMR and range_guard options.

    Node 0: high VF (0.8), small tensor  -> risk_reduction = 0.04
            macs_fraction=0.15, activation_magnitude=15.0, taylor_importance=0.005
    Node 1: medium VF (0.4), large tensor -> risk_reduction = 0.20
            macs_fraction=0.40, activation_magnitude=8.0, taylor_importance=0.020
    Node 2: low VF (0.1), medium tensor   -> risk_reduction = 0.02
            macs_fraction=0.30, activation_magnitude=25.0, taylor_importance=0.001
    Node 3: medium VF (0.3), medium tensor -> risk_reduction = 0.06
            macs_fraction=0.25, activation_magnitude=3.0, taylor_importance=0.010

    Designed so that each native metric produces a DIFFERENT ranking:
      - macs_fraction*VF: node1(0.16) > node0(0.12) > node3(0.075) > node2(0.03)
      - activation_magnitude: node2(25) > node0(15) > node1(8) > node3(3)
      - taylor_importance: node1(0.020) > node3(0.010) > node0(0.005) > node2(0.001)
      - risk_reduction: node1(0.20) > node3(0.06) > node0(0.04) > node2(0.02)
    """
    return [
        CandidateMode(node_id=0, tensor_id=0, mode="dmr_compare_rerun",
                       risk_reduction=0.04, latency_overhead_ms=1.0,
                       extra_memory_bytes=1000, critical_probability=0.8,
                       macs_fraction=0.15, activation_magnitude=15.0, taylor_importance=0.005),
        CandidateMode(node_id=0, tensor_id=0, mode="range_guard_rerun",
                       risk_reduction=0.03, latency_overhead_ms=0.2,
                       extra_memory_bytes=0, critical_probability=0.8,
                       lower_bound=-1.0, upper_bound=1.0,
                       macs_fraction=0.15, activation_magnitude=15.0, taylor_importance=0.005),
        CandidateMode(node_id=1, tensor_id=0, mode="dmr_compare_rerun",
                       risk_reduction=0.20, latency_overhead_ms=5.0,
                       extra_memory_bytes=10000, critical_probability=0.4,
                       macs_fraction=0.40, activation_magnitude=8.0, taylor_importance=0.020),
        CandidateMode(node_id=1, tensor_id=0, mode="range_guard_rerun",
                       risk_reduction=0.15, latency_overhead_ms=1.0,
                       extra_memory_bytes=0, critical_probability=0.4,
                       lower_bound=-2.0, upper_bound=2.0,
                       macs_fraction=0.40, activation_magnitude=8.0, taylor_importance=0.020),
        CandidateMode(node_id=2, tensor_id=0, mode="dmr_compare_rerun",
                       risk_reduction=0.02, latency_overhead_ms=2.0,
                       extra_memory_bytes=4000, critical_probability=0.1,
                       macs_fraction=0.30, activation_magnitude=25.0, taylor_importance=0.001),
        CandidateMode(node_id=2, tensor_id=0, mode="range_guard_rerun",
                       risk_reduction=0.01, latency_overhead_ms=0.3,
                       extra_memory_bytes=0, critical_probability=0.1,
                       lower_bound=-0.5, upper_bound=0.5,
                       macs_fraction=0.30, activation_magnitude=25.0, taylor_importance=0.001),
        CandidateMode(node_id=3, tensor_id=0, mode="dmr_compare_rerun",
                       risk_reduction=0.06, latency_overhead_ms=1.5,
                       extra_memory_bytes=3000, critical_probability=0.3,
                       macs_fraction=0.25, activation_magnitude=3.0, taylor_importance=0.010),
        CandidateMode(node_id=3, tensor_id=0, mode="range_guard_rerun",
                       risk_reduction=0.04, latency_overhead_ms=0.4,
                       extra_memory_bytes=0, critical_probability=0.3,
                       lower_bound=-1.5, upper_bound=1.5,
                       macs_fraction=0.25, activation_magnitude=3.0, taylor_importance=0.010),
    ]


BUDGET_LATENCY = 10.0
BUDGET_MEMORY = 20000


class TestOptimizeFilr:
    """FILR: Vfmap-ranked (risk_reduction = Vorig * Pprop) DMR only."""

    def test_selects_dmr_only(self):
        result = optimize_filr(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert all(c.mode == "dmr_compare_rerun" for c in result.selected)

    def test_risk_reduction_ranking_order(self):
        """FILR ranks by risk_reduction, so node 1 (0.20) should come before
        node 0 (0.04)."""
        result = optimize_filr(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        selected_ids = [c.node_id for c in result.selected]
        if 0 in selected_ids and 1 in selected_ids:
            assert selected_ids.index(1) < selected_ids.index(0)

    def test_respects_latency_budget(self):
        result = optimize_filr(_make_candidates(), latency_budget_ms=2.0, memory_budget_bytes=BUDGET_MEMORY)
        assert result.total_latency_overhead_ms <= 2.0

    def test_matches_topk_dmr(self):
        """FILR (risk_reduction ranking) is equivalent to topk_dmr."""
        candidates = _make_candidates()
        filr = optimize_filr(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        topk = optimize_topk_single_mode(candidates, mode="dmr_compare_rerun",
                                          latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        filr_ids = [c.node_id for c in filr.selected]
        topk_ids = [c.node_id for c in topk.selected]
        assert filr_ids == topk_ids


class TestOptimizeRanger:
    """Ranger: uniform range guard, no vulnerability ranking."""

    def test_selects_range_guard_only(self):
        result = optimize_ranger(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert all(c.mode == "range_guard_rerun" for c in result.selected)

    def test_node_id_order(self):
        """Ranger should pack in node_id order, not vulnerability order."""
        result = optimize_ranger(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        ids = [c.node_id for c in result.selected]
        assert ids == sorted(ids)

    def test_no_range_candidates_returns_empty(self):
        dmr_only = [c for c in _make_candidates() if c.mode == "dmr_compare_rerun"]
        result = optimize_ranger(dmr_only, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert len(result.selected) == 0


class TestOptimizeVfGreedy:
    """VF-only greedy (ablation): ranks by Pprop alone, no exposure weighting."""

    def test_allows_both_modes(self):
        result = optimize_vf_greedy(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        modes = {c.mode for c in result.selected}
        # With enough budget, can pick both modes across different nodes
        assert len(result.selected) > 0
        assert modes.issubset({"dmr_compare_rerun", "range_guard_rerun"})

    def test_vf_ranking(self):
        """Highest VF node should be selected first."""
        result = optimize_vf_greedy(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        if result.selected:
            assert result.selected[0].node_id == 0  # VF=0.8 is highest


class TestOptimizeUniformDmr:
    """Uniform DMR: node_id order, no vulnerability info."""

    def test_selects_dmr_only(self):
        result = optimize_uniform_dmr(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert all(c.mode == "dmr_compare_rerun" for c in result.selected)

    def test_node_id_order(self):
        result = optimize_uniform_dmr(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        ids = [c.node_id for c in result.selected]
        assert ids == sorted(ids)

    def test_ilp_dominates_uniform(self):
        """ILP should achieve >= risk reduction compared to uniform DMR."""
        candidates = _make_candidates()
        ilp = optimize_ilp(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        uniform = optimize_uniform_dmr(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert ilp.total_risk_reduction >= uniform.total_risk_reduction - 1e-9


class TestIlpDominatesBaselines:
    """ILP should always achieve >= risk reduction compared to all heuristics."""

    @pytest.fixture
    def candidates(self):
        return _make_candidates()

    def test_ilp_vs_filr(self, candidates):
        ilp = optimize_ilp(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        filr = optimize_filr(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert ilp.total_risk_reduction >= filr.total_risk_reduction - 1e-9

    def test_ilp_vs_greedy(self, candidates):
        ilp = optimize_ilp(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        greedy = optimize_greedy(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert ilp.total_risk_reduction >= greedy.total_risk_reduction - 1e-9

    def test_ilp_vs_vf_greedy(self, candidates):
        ilp = optimize_ilp(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        vf = optimize_vf_greedy(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert ilp.total_risk_reduction >= vf.total_risk_reduction - 1e-9

    def test_ilp_vs_random_dmr(self, candidates):
        ilp = optimize_ilp(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        rand = optimize_random_dmr(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY, seed=42)
        assert ilp.total_risk_reduction >= rand.total_risk_reduction - 1e-9

    def test_ilp_vs_ranger(self, candidates):
        ilp = optimize_ilp(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        ranger = optimize_ranger(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert ilp.total_risk_reduction >= ranger.total_risk_reduction - 1e-9

    def test_ilp_vs_uniform_dmr(self, candidates):
        ilp = optimize_ilp(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        uniform = optimize_uniform_dmr(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert ilp.total_risk_reduction >= uniform.total_risk_reduction - 1e-9

    def test_ilp_vs_aspis(self, candidates):
        ilp = optimize_ilp(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        aspis = optimize_aspis(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert ilp.total_risk_reduction >= aspis.total_risk_reduction - 1e-9

    def test_ilp_vs_ruospo(self, candidates):
        ilp = optimize_ilp(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        ruospo = optimize_ruospo(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert ilp.total_risk_reduction >= ruospo.total_risk_reduction - 1e-9

    def test_ilp_vs_ahmadilivani(self, candidates):
        ilp = optimize_ilp(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        ahm = optimize_ahmadilivani(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert ilp.total_risk_reduction >= ahm.total_risk_reduction - 1e-9


class TestAspisRuospo:
    """Aspis uses taylor_importance; Ruospo uses activation_magnitude."""

    def test_aspis_selects_dmr_only(self):
        result = optimize_aspis(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert all(c.mode == "dmr_compare_rerun" for c in result.selected)

    def test_ruospo_selects_dmr_only(self):
        result = optimize_ruospo(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert all(c.mode == "dmr_compare_rerun" for c in result.selected)

    def test_aspis_taylor_ranking(self):
        """Aspis ranks by taylor_importance: node1(0.020) > node3(0.010) > node0(0.005) > node2(0.001)."""
        result = optimize_aspis(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert result.selected[0].node_id == 1

    def test_ruospo_activation_magnitude_ranking(self):
        """Ruospo ranks by activation_magnitude: node2(25) > node0(15) > node1(8) > node3(3)."""
        result = optimize_ruospo(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert result.selected[0].node_id == 2

    def test_aspis_differs_from_ruospo(self):
        """Aspis (taylor) and Ruospo (activation_magnitude) now produce different rankings."""
        candidates = _make_candidates()
        aspis = optimize_aspis(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        ruospo = optimize_ruospo(candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert [c.node_id for c in aspis.selected] != [c.node_id for c in ruospo.selected]


class TestAhmadilivani:
    """Ahmadilivani: taylor_importance-ranked heterogeneous greedy (DMR + range_guard)."""

    def test_allows_both_modes(self):
        result = optimize_ahmadilivani(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        modes = {c.mode for c in result.selected}
        assert modes.issubset({"dmr_compare_rerun", "range_guard_rerun"})

    def test_taylor_ranking(self):
        """Highest taylor_importance node (node1=0.020) should be selected first."""
        result = optimize_ahmadilivani(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        if result.selected:
            assert result.selected[0].node_id == 1

    def test_prefers_dmr_over_range_guard(self):
        """For top-ranked nodes, DMR should be selected when budget allows."""
        result = optimize_ahmadilivani(_make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        if result.selected:
            assert result.selected[0].mode == "dmr_compare_rerun"

    def test_falls_back_to_range_guard_when_dmr_too_expensive(self):
        """When DMR doesn't fit, range guard should be used instead."""
        candidates = _make_candidates()
        result = optimize_ahmadilivani(candidates, latency_budget_ms=6.0, memory_budget_bytes=BUDGET_MEMORY)
        modes_by_node = {c.node_id: c.mode for c in result.selected}
        assert 1 in modes_by_node
        has_range = any(c.mode == "range_guard_rerun" for c in result.selected)
        assert has_range


class TestCriticalProbabilityFallback:
    """Methods should fall back gracefully when native metrics are None."""

    def test_filr_without_macs_fraction(self):
        """FILR falls back to risk_reduction when macs_fraction is None."""
        candidates = [
            CandidateMode(node_id=0, tensor_id=0, mode="dmr_compare_rerun",
                           risk_reduction=0.1, latency_overhead_ms=1.0,
                           extra_memory_bytes=1000),
            CandidateMode(node_id=1, tensor_id=0, mode="dmr_compare_rerun",
                           risk_reduction=0.2, latency_overhead_ms=1.0,
                           extra_memory_bytes=1000),
        ]
        result = optimize_filr(candidates, latency_budget_ms=5.0, memory_budget_bytes=5000)
        assert len(result.selected) == 2
        assert result.selected[0].node_id == 1  # higher risk_reduction first

    def test_aspis_without_taylor_importance(self):
        """Aspis falls back to activation_magnitude when taylor_importance is None."""
        candidates = [
            CandidateMode(node_id=0, tensor_id=0, mode="dmr_compare_rerun",
                           risk_reduction=0.10, latency_overhead_ms=1.0,
                           extra_memory_bytes=1000, activation_magnitude=5.0),
            CandidateMode(node_id=1, tensor_id=0, mode="dmr_compare_rerun",
                           risk_reduction=0.05, latency_overhead_ms=1.0,
                           extra_memory_bytes=1000, activation_magnitude=20.0),
        ]
        result = optimize_aspis(candidates, latency_budget_ms=5.0, memory_budget_bytes=5000)
        assert len(result.selected) == 2
        assert result.selected[0].node_id == 1  # higher activation_magnitude first

    def test_ruospo_without_activation_magnitude(self):
        """Ruospo falls back to risk_reduction when activation_magnitude is None."""
        candidates = [
            CandidateMode(node_id=0, tensor_id=0, mode="dmr_compare_rerun",
                           risk_reduction=0.05, latency_overhead_ms=1.0,
                           extra_memory_bytes=1000),
            CandidateMode(node_id=1, tensor_id=0, mode="dmr_compare_rerun",
                           risk_reduction=0.10, latency_overhead_ms=1.0,
                           extra_memory_bytes=1000),
        ]
        result = optimize_ruospo(candidates, latency_budget_ms=5.0, memory_budget_bytes=5000)
        assert len(result.selected) == 2
        assert result.selected[0].node_id == 1


class TestIlpStatic:
    """ILP-static: Taylor/ActMag proxy, MCKP optimization, no MC data needed."""

    def test_respects_latency_budget(self):
        result = optimize_ilp_static(
            _make_candidates(), latency_budget_ms=2.0, memory_budget_bytes=BUDGET_MEMORY)
        assert result.total_latency_overhead_ms <= 2.0

    def test_selects_heterogeneous_modes(self):
        """ILP-static should use both DMR and RG when beneficial."""
        result = optimize_ilp_static(
            _make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        modes = {c.mode for c in result.selected}
        assert len(result.selected) > 0
        assert modes.issubset({"dmr_compare_rerun", "range_guard_rerun"})

    def test_dominates_aspis(self):
        """ILP-static with same proxy should achieve >= risk_reduction vs ASPIS (greedy DMR-only)."""
        candidates = _make_candidates()
        static = optimize_ilp_static(
            candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        aspis = optimize_aspis(
            candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert static.total_risk_reduction >= aspis.total_risk_reduction - 1e-9

    def test_method_name(self):
        result = optimize_ilp_static(
            _make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert result.method == "ilp_static"

    def test_empty_candidates(self):
        result = optimize_ilp_static(
            [], latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert len(result.selected) == 0


class TestIlpV2:
    """ILP-v2: two-tier (ILP on CF nodes + RG backfill on rest)."""

    def test_respects_latency_budget(self):
        result = optimize_ilp_v2(
            _make_candidates(), latency_budget_ms=2.0, memory_budget_bytes=BUDGET_MEMORY,
            cf_node_ids={0, 1})
        assert result.total_latency_overhead_ms <= 2.0

    def test_cf_nodes_get_ilp_treatment(self):
        """CF nodes should be selected by ILP (potentially DMR)."""
        candidates = _make_candidates()
        result = optimize_ilp_v2(
            candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY,
            cf_node_ids={1})
        selected_ids = {c.node_id for c in result.selected}
        assert 1 in selected_ids

    def test_non_cf_nodes_get_rg_backfill(self):
        """Non-CF nodes should only get RG (never DMR) from backfill tier."""
        candidates = _make_candidates()
        result = optimize_ilp_v2(
            candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY,
            cf_node_ids={1})
        for c in result.selected:
            if c.node_id != 1:
                assert c.mode == "range_guard_rerun", \
                    f"Non-CF node {c.node_id} should only get RG, got {c.mode}"

    def test_empty_cf_nodes_becomes_pure_rg_backfill(self):
        """With no CF nodes, ILP-v2 degenerates to RG backfill only."""
        candidates = _make_candidates()
        result = optimize_ilp_v2(
            candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY,
            cf_node_ids=set())
        assert all(c.mode == "range_guard_rerun" for c in result.selected)

    def test_per_node_rg_coverage_suppresses_dmr(self):
        """When a CF node has high RG coverage, DMR should be excluded for it."""
        candidates = _make_candidates()
        result = optimize_ilp_v2(
            candidates, latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY,
            cf_node_ids={0, 1, 2, 3},
            per_node_rg_coverage={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
            rg_coverage_threshold=0.95)
        for c in result.selected:
            assert c.mode == "range_guard_rerun", \
                f"Node {c.node_id} with 100% RG coverage should not get DMR"

    def test_method_name(self):
        result = optimize_ilp_v2(
            _make_candidates(), latency_budget_ms=BUDGET_LATENCY, memory_budget_bytes=BUDGET_MEMORY)
        assert result.method == "ilp_v2"
