"""Multiple-choice protection-plan optimization under latency and memory budgets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import pulp


@dataclass(frozen=True)
class CandidateMode:
    node_id: int
    tensor_id: int
    mode: str
    risk_reduction: float
    latency_overhead_ms: float
    extra_memory_bytes: int
    lower_bound: float | None = None
    upper_bound: float | None = None
    critical_probability: float | None = None
    macs_fraction: float | None = None
    activation_magnitude: float | None = None
    taylor_importance: float | None = None


@dataclass(frozen=True)
class OptimizationResult:
    method: str
    selected: tuple[CandidateMode, ...]
    total_risk_reduction: float
    total_latency_overhead_ms: float
    total_extra_memory_bytes: int

    def to_protection_plan(
        self,
        *,
        model_id: str,
        platform_profile: str,
        latency_budget_ms: float,
        memory_budget_bytes: int,
        fault_prior: str = "activation_bytes_weighted_single_bit",
        workload_scope: str | None = None,
    ) -> dict:
        nodes = []
        for choice in self.selected:
            node = {"node_id": choice.node_id, "tensor_id": choice.tensor_id, "mode": choice.mode}
            if choice.lower_bound is not None:
                node["lower_bound"] = choice.lower_bound
            if choice.upper_bound is not None:
                node["upper_bound"] = choice.upper_bound
            nodes.append(node)
        plan = {
            "version": 1,
            "model_id": model_id,
            "platform_profile": platform_profile,
            "fault_prior": fault_prior,
            "budgets": {
                "latency_overhead_ms": latency_budget_ms,
                "extra_memory_bytes": memory_budget_bytes,
            },
            "optimizer": {
                "method": self.method,
                "predicted_risk_reduction": self.total_risk_reduction,
                "predicted_latency_overhead_ms": self.total_latency_overhead_ms,
                "peak_extra_memory_bytes": self.total_extra_memory_bytes,
            },
            "nodes": nodes,
        }
        if workload_scope is not None:
            plan["workload_scope"] = workload_scope
        return plan


def optimize_ilp(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    """Optimal selective protection via Multiple-Choice Knapsack ILP.

    **Original contribution.**  Formulates budgeted protection-plan selection
    as a multiple-choice knapsack: one candidate per node-id may be selected,
    maximising total risk_reduction subject to a latency budget.  Ties broken
    by minimising latency, so the solution is both risk-optimal and efficient.

    When used with calibrated candidates (two-phase adaptive ILP), the
    risk_reduction values encode empirically measured critical_probability and
    range-guard coverage from a pilot MC run, replacing the initial uniform
    assumption.  The ILP formulation is identical in both phases; only the
    input data differs.
    """
    if latency_budget_ms < 0 or memory_budget_bytes < 0:
        raise ValueError("budgets must be non-negative")
    problem = pulp.LpProblem("selective_software_protection", pulp.LpMaximize)
    variables = {
        index: problem.add_variable(f"x_{candidate.node_id}_{candidate.mode}_{index}", cat="Binary")
        for index, candidate in enumerate(candidates)
    }
    max_reduction = max((candidate.risk_reduction for candidate in candidates), default=0.0)
    objective_scale = 1.0 / max_reduction if max_reduction > 0.0 else 1.0
    risk_objective = pulp.lpSum(
        variables[i] * candidate.risk_reduction * objective_scale for i, candidate in enumerate(candidates)
    )
    latency_objective = pulp.lpSum(
        variables[i] * candidate.latency_overhead_ms for i, candidate in enumerate(candidates)
    )
    problem += risk_objective
    problem += latency_objective <= latency_budget_ms
    for index, candidate in enumerate(candidates):
        if candidate.extra_memory_bytes > memory_budget_bytes:
            problem += variables[index] == 0
    for node_id in {candidate.node_id for candidate in candidates}:
        problem += pulp.lpSum(variables[i] for i, candidate in enumerate(candidates) if candidate.node_id == node_id) <= 1
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"optimizer failed: {pulp.LpStatus[status]}")
    optimal_risk = float(pulp.value(risk_objective) or 0.0)
    problem += risk_objective >= optimal_risk - 1e-9
    problem.sense = pulp.LpMinimize
    problem.setObjective(latency_objective)
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"optimizer tie-break failed: {pulp.LpStatus[status]}")
    selected = tuple(candidate for i, candidate in enumerate(candidates) if pulp.value(variables[i]) > 0.5)
    return _summarize("ilp", selected)


def optimize_bounded_loss_memory_ilp(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
    risk_loss_tolerance: float = 0.05,
) -> OptimizationResult:
    if not 0.0 <= risk_loss_tolerance < 1.0:
        raise ValueError("risk_loss_tolerance must be in [0, 1)")
    optimal = optimize_ilp(
        candidates,
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
    )
    if optimal.total_risk_reduction <= 0.0:
        return _summarize(f"bounded_loss_memory_ilp_tol{risk_loss_tolerance:g}", ())

    min_risk = optimal.total_risk_reduction * (1.0 - risk_loss_tolerance)
    problem = pulp.LpProblem("bounded_loss_memory_protection", pulp.LpMinimize)
    variables = {
        index: problem.add_variable(f"x_{candidate.node_id}_{candidate.mode}_{index}", cat="Binary")
        for index, candidate in enumerate(candidates)
    }
    peak_memory = problem.add_variable("peak_extra_memory_bytes", lowBound=0)
    latency = pulp.lpSum(variables[i] * candidate.latency_overhead_ms for i, candidate in enumerate(candidates))
    risk = pulp.lpSum(variables[i] * candidate.risk_reduction for i, candidate in enumerate(candidates))
    problem += peak_memory
    problem += latency <= latency_budget_ms
    problem += peak_memory <= memory_budget_bytes
    problem += risk >= min_risk
    for index, candidate in enumerate(candidates):
        problem += peak_memory >= variables[index] * candidate.extra_memory_bytes
    for node_id in {candidate.node_id for candidate in candidates}:
        problem += pulp.lpSum(variables[i] for i, candidate in enumerate(candidates) if candidate.node_id == node_id) <= 1
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"bounded-loss memory optimizer failed: {pulp.LpStatus[status]}")

    optimal_peak = float(pulp.value(peak_memory) or 0.0)
    problem += peak_memory <= optimal_peak + 1e-6
    problem.sense = pulp.LpMaximize
    problem.setObjective(risk)
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"bounded-loss memory optimizer tie-break failed: {pulp.LpStatus[status]}")

    best_risk = float(pulp.value(risk) or 0.0)
    problem += risk >= best_risk - 1e-9
    problem.sense = pulp.LpMinimize
    problem.setObjective(latency)
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"bounded-loss memory optimizer latency tie-break failed: {pulp.LpStatus[status]}")

    selected = tuple(candidate for i, candidate in enumerate(candidates) if pulp.value(variables[i]) > 0.5)
    return _summarize(f"bounded_loss_memory_ilp_tol{risk_loss_tolerance:g}", selected)


def optimize_greedy(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    selected: list[CandidateMode] = []
    used_nodes: set[int] = set()
    used_latency = 0.0
    peak_memory = 0
    ordered = sorted(
        candidates,
        key=lambda choice: (
            -(choice.risk_reduction / max(choice.latency_overhead_ms, 1e-12)),
            -choice.risk_reduction,
        ),
    )
    for choice in ordered:
        if choice.node_id in used_nodes:
            continue
        if used_latency + choice.latency_overhead_ms > latency_budget_ms:
            continue
        if max(peak_memory, choice.extra_memory_bytes) > memory_budget_bytes:
            continue
        selected.append(choice)
        used_nodes.add(choice.node_id)
        used_latency += choice.latency_overhead_ms
        peak_memory = max(peak_memory, choice.extra_memory_bytes)
    return _summarize("greedy", tuple(selected))


def optimize_topk_single_mode(
    candidates: list[CandidateMode],
    *,
    mode: str,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    """Select highest-benefit objects using one fixed protection primitive."""
    return _pack_ordered(
        [candidate for candidate in candidates if candidate.mode == mode],
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
        method=f"topk_{mode}",
        order_key=lambda choice: (-choice.risk_reduction, choice.latency_overhead_ms),
    )


def optimize_random_dmr(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
    seed: int,
) -> OptimizationResult:
    """Random DMR selection baseline with deterministic seed."""
    dmr = [candidate for candidate in candidates if candidate.mode == "dmr_compare_rerun"]
    generator = random.Random(seed)
    generator.shuffle(dmr)
    return _pack_ordered(
        dmr,
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
        method="random_dmr",
        order_key=None,
    )


def optimize_filr(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    """FILR-style baseline: rank by Vfmap = Vorig * Pprop, DMR only, greedy.

    **Faithful.** Mahmoud et al., "Optimizing Selective Protection for CNN
    Resilience", ISSRE 2021.  FLR ranks feature maps by Vfmap (exposure
    Vorig × propagation probability Pprop) and greedily selects for DWC.
    Our risk_reduction = fault_prior × critical_prob maps directly to Vfmap.
    """
    return _pack_ordered(
        [c for c in candidates if c.mode == "dmr_compare_rerun"],
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
        method="filr",
        order_key=lambda c: (-c.risk_reduction, c.latency_overhead_ms),
    )


def optimize_ranger(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    """Ranger-style baseline: apply range guard uniformly, no vulnerability ranking.

    **Faithful.** Chen et al., "Ranger: Boosting Error Resilience of DNNs
    through Range Restriction", DSN 2021.  Ranger clips activations at
    every layer without selective ordering.  We pack range-guard candidates
    in node-id order, matching the original non-selective approach.
    """
    return _pack_ordered(
        [c for c in candidates if c.mode == "range_guard_rerun"],
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
        method="ranger",
        order_key=lambda c: (c.node_id, c.tensor_id),
    )


def optimize_vf_greedy(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    """VF-only greedy: rank all modes by critical_probability alone (no exposure).

    Ablation baseline showing the importance of exposure-weighting.  Ranks
    purely by Pprop (critical_probability) without Vorig, unlike FILR which
    multiplies both.  Falls back to risk_reduction when critical_probability
    is unavailable.
    """
    return _pack_ordered(
        candidates,
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
        method="vf_greedy",
        order_key=lambda c: (-(c.critical_probability if c.critical_probability is not None else c.risk_reduction), c.latency_overhead_ms),
    )


def _gradient_proxy(c: CandidateMode) -> float:
    """Taylor importance → activation magnitude → risk_reduction fallback."""
    if c.taylor_importance is not None:
        return c.taylor_importance
    if c.activation_magnitude is not None:
        return c.activation_magnitude
    return c.risk_reduction


def optimize_aspis(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    """Aspis-style baseline: Taylor importance ranking, DMR only, greedy.

    **Adapted.** Schmedding et al., "Aspis: Lightweight Neural Network
    Protection Against Soft Errors", ISSRE 2024.  Original: ranks layers
    by Taylor criterion |dL/dw × w| and protects weights via TMR.
    Adapted: same Taylor ranking but protects activations via DMR, since
    our fault model targets activation tensors, not stored weights.
    Fallback: taylor_importance → activation_magnitude → risk_reduction.
    """
    return _pack_ordered(
        [c for c in candidates if c.mode == "dmr_compare_rerun"],
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
        method="aspis",
        order_key=lambda c: (-_gradient_proxy(c), c.latency_overhead_ms),
    )


def optimize_ruospo(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    """Ruospo-style baseline: activation magnitude ranking, DMR only, greedy.

    **Adapted.** Ruospo et al., "Selective Hardening of Critical Neurons
    in Deep Neural Networks", DDECS 2022.  Original: ranks neurons by
    mean |activation| and applies TMR at neuron granularity.  Adapted:
    same activation-magnitude ranking but at tensor granularity with DMR,
    since our runtime operates on whole tensors, not individual neurons.
    Falls back to risk_reduction when activation_magnitude is unavailable.
    """
    return _pack_ordered(
        [c for c in candidates if c.mode == "dmr_compare_rerun"],
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
        method="ruospo",
        order_key=lambda c: (-(c.activation_magnitude if c.activation_magnitude is not None else c.risk_reduction), c.latency_overhead_ms),
    )


def optimize_ahmadilivani(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    """Ahmadilivani-style baseline: gradient importance ranking, heterogeneous, greedy.

    **Adapted.** Ahmadilivani et al., "Cost-Effective Fault Tolerance for
    CNNs Using Parameter Vulnerability Based Hardening and Pruning", IOLTS
    2024.  Original: ranks channels by gradient-based vulnerability, applies
    TMR to top-k% and EDAC intervals to the remainder.  Adapted: same
    vulnerability ranking but uses DMR (not TMR) and range guard (not EDAC),
    visiting nodes in vulnerability order and trying DMR first then RG.
    Fallback: taylor_importance → activation_magnitude → risk_reduction.
    """
    by_node: dict[int, dict[str, CandidateMode]] = {}
    for c in candidates:
        by_node.setdefault(c.node_id, {})[c.mode] = c
    node_order = sorted(by_node.keys(),
                        key=lambda nid: -_gradient_proxy(next(iter(by_node[nid].values()))))
    selected: list[CandidateMode] = []
    used_latency = 0.0
    peak_memory = 0
    for nid in node_order:
        modes = by_node[nid]
        for mode in ("dmr_compare_rerun", "range_guard_rerun"):
            c = modes.get(mode)
            if c is None:
                continue
            if used_latency + c.latency_overhead_ms > latency_budget_ms:
                continue
            if max(peak_memory, c.extra_memory_bytes) > memory_budget_bytes:
                continue
            selected.append(c)
            used_latency += c.latency_overhead_ms
            peak_memory = max(peak_memory, c.extra_memory_bytes)
            break
    return _summarize("ahmadilivani", tuple(selected))


def optimize_ilp_static(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    """ILP with static vulnerability proxy — no MC fault injection data needed.

    Uses Taylor importance (or activation magnitude fallback) as a proxy for
    critical probability.  Maintains MCKP optimality and heterogeneous DMR+RG
    selection, but derived entirely from the model itself.
    """
    total_bytes = sum(
        c.extra_memory_bytes // 2
        for c in candidates
        if c.mode == "dmr_compare_rerun"
    )
    if total_bytes == 0:
        return _summarize("ilp_static", ())

    proxy_vals = [_gradient_proxy(c) for c in candidates]
    max_proxy = max(proxy_vals) if proxy_vals else 1.0
    if max_proxy <= 0:
        max_proxy = 1.0

    proxy_candidates = []
    for c, pv in zip(candidates, proxy_vals):
        proxy_norm = pv / max_proxy
        fault_prior = (c.extra_memory_bytes // 2) / total_bytes if c.mode == "dmr_compare_rerun" else (
            next((c2.extra_memory_bytes // 2 for c2 in candidates
                  if c2.node_id == c.node_id and c2.mode == "dmr_compare_rerun"), 0)
        ) / total_bytes
        coverage = 0.95 if c.mode == "range_guard_rerun" else 1.0
        new_risk = fault_prior * proxy_norm * coverage
        proxy_candidates.append(CandidateMode(
            node_id=c.node_id, tensor_id=c.tensor_id, mode=c.mode,
            risk_reduction=new_risk, latency_overhead_ms=c.latency_overhead_ms,
            extra_memory_bytes=c.extra_memory_bytes,
            lower_bound=c.lower_bound, upper_bound=c.upper_bound,
            critical_probability=proxy_norm,
            macs_fraction=c.macs_fraction,
            activation_magnitude=c.activation_magnitude,
            taylor_importance=c.taylor_importance,
        ))

    result = optimize_ilp(
        proxy_candidates,
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
    )
    return _summarize("ilp_static", result.selected)


def optimize_ilp_v2(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
    cf_node_ids: set[int] | None = None,
    per_node_rg_coverage: dict[int, float] | None = None,
    rg_coverage_threshold: float = 0.95,
) -> OptimizationResult:
    """Two-tier ILP: precise ILP on CF-confirmed nodes, RG backfill on rest.

    Tier 1: ILP on CF-confirmed nodes only (accurate calibrated CP).
            For nodes with per-node RG coverage >= threshold, exclude DMR.
    Tier 2: Greedy RG backfill on remaining nodes by fault_prior descending.
    """
    if cf_node_ids is None:
        cf_node_ids = {
            c.node_id for c in candidates
            if c.critical_probability is not None and c.critical_probability > 0.05
        }

    tier1_candidates = []
    for c in candidates:
        if c.node_id not in cf_node_ids:
            continue
        if per_node_rg_coverage and c.mode == "dmr_compare_rerun":
            node_cov = per_node_rg_coverage.get(c.node_id, 0)
            if node_cov >= rg_coverage_threshold:
                continue
        tier1_candidates.append(c)

    if tier1_candidates:
        tier1_result = optimize_ilp(
            tier1_candidates,
            latency_budget_ms=latency_budget_ms,
            memory_budget_bytes=memory_budget_bytes,
        )
    else:
        tier1_result = _summarize("ilp_v2", ())

    used_nodes = {c.node_id for c in tier1_result.selected}
    remaining_budget = latency_budget_ms - tier1_result.total_latency_overhead_ms

    total_bytes = sum(
        c.extra_memory_bytes // 2
        for c in candidates
        if c.mode == "dmr_compare_rerun"
    )
    backfill_candidates = [
        c for c in candidates
        if c.node_id not in used_nodes and c.mode == "range_guard_rerun"
    ]
    backfill_candidates.sort(
        key=lambda c: -(
            next((c2.extra_memory_bytes // 2 for c2 in candidates
                  if c2.node_id == c.node_id and c2.mode == "dmr_compare_rerun"), 0)
        ) / total_bytes if total_bytes > 0 else 0
    )

    backfill_selected = []
    for c in backfill_candidates:
        if remaining_budget >= c.latency_overhead_ms:
            backfill_selected.append(c)
            remaining_budget -= c.latency_overhead_ms

    all_selected = tuple(tier1_result.selected) + tuple(backfill_selected)
    return _summarize("ilp_v2", all_selected)


def optimize_ilp_v3(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
    per_node_rg_coverage: dict[int, float] | None = None,
    rg_coverage_threshold: float = 0.95,
) -> OptimizationResult:
    """Coverage-guaranteed ILP: MCKP with Ranger coverage floor constraint.

    **Original contribution.**  Solves the standard risk_reduction-maximising
    MCKP but adds a *coverage floor*: every node that Ranger would protect
    under the same budget must also be protected by ILP-v3.  This guarantees
    ILP-v3 >= Ranger while still optimising mode selection (RG vs DMR) and
    node priority via the ILP objective.

    For nodes with per-node RG coverage >= *rg_coverage_threshold*, DMR
    candidates are excluded (RG already provides near-perfect detection).
    """
    ranger_result = optimize_ranger(
        candidates,
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
    )
    ranger_nodes = {c.node_id for c in ranger_result.selected}

    filtered = []
    for c in candidates:
        if c.mode == "dmr_compare_rerun" and per_node_rg_coverage:
            cov = per_node_rg_coverage.get(c.node_id, 1.0)
            if cov >= rg_coverage_threshold:
                continue
        filtered.append(c)

    if not filtered:
        return _summarize("ilp_v3", ())

    problem = pulp.LpProblem("ilp_v3_coverage_floor", pulp.LpMaximize)
    variables = {
        i: problem.add_variable(f"x_{c.node_id}_{c.mode}_{i}", cat="Binary")
        for i, c in enumerate(filtered)
    }

    max_rr = max(c.risk_reduction for c in filtered)
    scale = 1.0 / max_rr if max_rr > 0 else 1.0
    risk_obj = pulp.lpSum(
        variables[i] * c.risk_reduction * scale for i, c in enumerate(filtered)
    )
    latency_expr = pulp.lpSum(
        variables[i] * c.latency_overhead_ms for i, c in enumerate(filtered)
    )

    problem += risk_obj
    problem += latency_expr <= latency_budget_ms

    for i, c in enumerate(filtered):
        if c.extra_memory_bytes > memory_budget_bytes:
            problem += variables[i] == 0

    for node_id in {c.node_id for c in filtered}:
        node_vars = pulp.lpSum(
            variables[i] for i, c in enumerate(filtered) if c.node_id == node_id
        )
        problem += node_vars <= 1
        if node_id in ranger_nodes:
            problem += node_vars >= 1

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"ILP-v3 failed: {pulp.LpStatus[status]}")

    optimal_risk = float(pulp.value(risk_obj) or 0.0)
    problem += risk_obj >= optimal_risk - 1e-9
    problem.sense = pulp.LpMinimize
    problem.setObjective(latency_expr)
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"ILP-v3 tie-break failed: {pulp.LpStatus[status]}")

    selected = tuple(c for i, c in enumerate(filtered) if pulp.value(variables[i]) > 0.5)
    return _summarize("ilp_v3", selected)


def optimize_uniform_dmr(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
) -> OptimizationResult:
    """Uniform DMR baseline: protect nodes in node-id order, DMR only.

    **Ablation.** Non-selective control: identical to Ranger in ordering
    logic but uses DMR instead of Range Guard.  Isolates the contribution
    of mode selection (DMR vs RG) from vulnerability-based ordering.
    """
    return _pack_ordered(
        [c for c in candidates if c.mode == "dmr_compare_rerun"],
        latency_budget_ms=latency_budget_ms,
        memory_budget_bytes=memory_budget_bytes,
        method="uniform_dmr",
        order_key=lambda c: (c.node_id, c.tensor_id),
    )


def write_plan(plan: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize(method: str, selected: tuple[CandidateMode, ...]) -> OptimizationResult:
    return OptimizationResult(
        method=method,
        selected=selected,
        total_risk_reduction=sum(choice.risk_reduction for choice in selected),
        total_latency_overhead_ms=sum(choice.latency_overhead_ms for choice in selected),
        total_extra_memory_bytes=max((choice.extra_memory_bytes for choice in selected), default=0),
    )


def _pack_ordered(
    candidates: list[CandidateMode],
    *,
    latency_budget_ms: float,
    memory_budget_bytes: int,
    method: str,
    order_key,
) -> OptimizationResult:
    ordered = sorted(candidates, key=order_key) if order_key is not None else candidates
    selected = []
    used_nodes = set()
    used_latency = 0.0
    peak_memory = 0
    for choice in ordered:
        if choice.node_id in used_nodes:
            continue
        if used_latency + choice.latency_overhead_ms > latency_budget_ms:
            continue
        if max(peak_memory, choice.extra_memory_bytes) > memory_budget_bytes:
            continue
        selected.append(choice)
        used_nodes.add(choice.node_id)
        used_latency += choice.latency_overhead_ms
        peak_memory = max(peak_memory, choice.extra_memory_bytes)
    return _summarize(method, tuple(selected))
