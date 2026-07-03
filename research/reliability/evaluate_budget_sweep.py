"""Compare protection-selection algorithms across latency and peak-memory budgets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.reliability.optimizer.plan_optimizer import (
    CandidateMode,
    optimize_ahmadilivani,
    optimize_aspis,
    optimize_bounded_loss_memory_ilp,
    optimize_filr,
    optimize_greedy,
    optimize_ilp,
    optimize_ilp_static,
    optimize_ilp_v2,
    optimize_random_dmr,
    optimize_ranger,
    optimize_ruospo,
    optimize_topk_single_mode,
    optimize_uniform_dmr,
    optimize_vf_greedy,
    write_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep selective protection budgets and compare algorithms.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cost-profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latency-budgets-ms", default="5,10,20,40,80,160,320,640")
    parser.add_argument("--memory-budgets-bytes", default="0,1048576,4194304,8388608")
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--bounded-loss-tolerance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--representative-latency-ms", type=float, default=40.0)
    parser.add_argument("--representative-memory-bytes", type=int, default=8388608)
    parser.add_argument("--platform-profile", default="win_x64_cpu_ref")
    parser.add_argument("--model-id", default="resnet50_eurosat_gpu_prevalidation")
    parser.add_argument("--fault-prior", default="activation_bytes_weighted_single_bit")
    parser.add_argument("--workload-scope")
    parser.add_argument("--output-prefix", default="resnet50")
    parser.add_argument("--title-prefix", default="ResNet50")
    parser.add_argument("--calibration-metadata",
                        help="Calibration metadata JSON for ILP-v2 (provides CF node IDs and per-node RG coverage).")
    args = parser.parse_args()

    candidates = [
        CandidateMode(**item)
        for item in json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    ]
    profile = json.loads(Path(args.cost_profile).read_text(encoding="utf-8"))
    baseline = float(profile["baseline_expected_critical_failure_probability"])

    cf_node_ids = None
    per_node_rg_coverage = None
    if args.calibration_metadata:
        meta = json.loads(Path(args.calibration_metadata).read_text(encoding="utf-8"))
        cf_node_ids = {
            int(nid) for nid, nd in meta.get("per_node", {}).items()
            if nd.get("critical_failures", 0) > 0
        }
        per_node_rg_coverage = {
            int(nid): float(v) for nid, v in meta.get("per_node_rg_coverage", {}).items()
        }

    latencies = [float(value) for value in args.latency_budgets_ms.split(",")]
    memories = [int(value) for value in args.memory_budgets_bytes.split(",")]
    rows = []
    for memory in memories:
        for latency in latencies:
            results = [
                optimize_ilp(candidates, latency_budget_ms=latency, memory_budget_bytes=memory),
                optimize_bounded_loss_memory_ilp(
                    candidates,
                    latency_budget_ms=latency,
                    memory_budget_bytes=memory,
                    risk_loss_tolerance=args.bounded_loss_tolerance,
                ),
                optimize_greedy(candidates, latency_budget_ms=latency, memory_budget_bytes=memory),
                optimize_topk_single_mode(
                    candidates,
                    mode="dmr_compare_rerun",
                    latency_budget_ms=latency,
                    memory_budget_bytes=memory,
                ),
                optimize_filr(candidates, latency_budget_ms=latency, memory_budget_bytes=memory),
                optimize_aspis(candidates, latency_budget_ms=latency, memory_budget_bytes=memory),
                optimize_ruospo(candidates, latency_budget_ms=latency, memory_budget_bytes=memory),
                optimize_ahmadilivani(candidates, latency_budget_ms=latency, memory_budget_bytes=memory),
                optimize_vf_greedy(candidates, latency_budget_ms=latency, memory_budget_bytes=memory),
                optimize_uniform_dmr(candidates, latency_budget_ms=latency, memory_budget_bytes=memory),
                optimize_ilp_static(candidates, latency_budget_ms=latency, memory_budget_bytes=memory),
                optimize_ilp_v2(
                    candidates, latency_budget_ms=latency, memory_budget_bytes=memory,
                    cf_node_ids=cf_node_ids, per_node_rg_coverage=per_node_rg_coverage,
                ),
            ]
            if any(candidate.mode == "range_guard_rerun" for candidate in candidates):
                results.append(
                    optimize_topk_single_mode(
                        candidates,
                        mode="range_guard_rerun",
                        latency_budget_ms=latency,
                        memory_budget_bytes=memory,
                    )
                )
                results.append(
                    optimize_ranger(candidates, latency_budget_ms=latency, memory_budget_bytes=memory),
                )
            for result in results:
                rows.append(_result_row(result, baseline, latency, memory))
            random_results = [
                optimize_random_dmr(
                    candidates,
                    latency_budget_ms=latency,
                    memory_budget_bytes=memory,
                    seed=args.seed + repeat,
                )
                for repeat in range(args.random_repeats)
            ]
            reductions = np.array([item.total_risk_reduction for item in random_results])
            row = _result_row(random_results[0], baseline, latency, memory)
            row["method"] = "random_dmr_mean"
            row["risk_reduction"] = float(reductions.mean())
            row["risk_reduction_std"] = float(reductions.std())
            row["residual_failure_probability"] = baseline - row["risk_reduction"]
            row["mitigation_ratio"] = row["risk_reduction"] / baseline if baseline else 0.0
            rows.append(row)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / f"{args.output_prefix}_budget_sweep.csv", index=False)
    _plot_sweep(table, memories, output_dir / f"{args.output_prefix}_budget_pareto.png", title_prefix=args.title_prefix)

    representative = optimize_ilp(
        candidates,
        latency_budget_ms=args.representative_latency_ms,
        memory_budget_bytes=args.representative_memory_bytes,
    )
    memory_fallback = optimize_bounded_loss_memory_ilp(
        candidates,
        latency_budget_ms=args.representative_latency_ms,
        memory_budget_bytes=args.representative_memory_bytes,
        risk_loss_tolerance=args.bounded_loss_tolerance,
    )
    plan = representative.to_protection_plan(
        model_id=args.model_id,
        platform_profile=args.platform_profile,
        latency_budget_ms=args.representative_latency_ms,
        memory_budget_bytes=args.representative_memory_bytes,
        fault_prior=args.fault_prior,
        workload_scope=args.workload_scope,
    )
    write_plan(plan, output_dir / f"{args.output_prefix}_representative_ilp_plan.json")
    write_plan(
        memory_fallback.to_protection_plan(
            model_id=args.model_id,
            platform_profile=args.platform_profile,
            latency_budget_ms=args.representative_latency_ms,
            memory_budget_bytes=args.representative_memory_bytes,
            fault_prior=args.fault_prior,
            workload_scope=args.workload_scope,
        ),
        output_dir / f"{args.output_prefix}_representative_bounded_loss_memory_plan.json",
    )
    comparisons = table[
        (table["latency_budget_ms"] == args.representative_latency_ms)
        & (table["peak_memory_budget_bytes"] == args.representative_memory_bytes)
    ].sort_values("mitigation_ratio", ascending=False)
    summary = {
        "objective": profile["objective"],
        "baseline_expected_critical_failure_probability": baseline,
        "representative_budget": {
            "latency_overhead_ms": args.representative_latency_ms,
            "peak_extra_memory_bytes": args.representative_memory_bytes,
        },
        "representative_results": comparisons.to_dict(orient="records"),
        "ilp_selected_modes": [asdict(choice) for choice in representative.selected],
        "bounded_loss_tolerance": args.bounded_loss_tolerance,
        "bounded_loss_memory_selected_modes": [asdict(choice) for choice in memory_fallback.selected],
    }
    (output_dir / f"{args.output_prefix}_budget_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["representative_results"], ensure_ascii=False))
    return 0


def _result_row(result, baseline: float, latency: float, memory: int) -> dict:
    return {
        "latency_budget_ms": latency,
        "peak_memory_budget_bytes": memory,
        "method": result.method,
        "selected_count": len(result.selected),
        "risk_reduction": result.total_risk_reduction,
        "risk_reduction_std": 0.0,
        "residual_failure_probability": baseline - result.total_risk_reduction,
        "mitigation_ratio": result.total_risk_reduction / baseline if baseline else 0.0,
        "used_latency_ms": result.total_latency_overhead_ms,
        "used_peak_extra_memory_bytes": result.total_extra_memory_bytes,
    }


def _plot_sweep(table: pd.DataFrame, memories: list[int], output: Path, *, title_prefix: str) -> None:
    labels = {
        "ilp": "Proposed ILP (multi-mode)",
        "greedy": "Greedy benefit/latency",
        "topk_dmr_compare_rerun": "Top-k DMR (risk_reduction)",
        "topk_range_guard_rerun": "Top-k range guard",
        "random_dmr_mean": "Random DMR (mean)",
        "filr": "FILR (Vfmap-ranked DMR) [ISSRE'21]",
        "aspis": "Aspis (Taylor-ranked DMR) [ISSRE'24]",
        "ruospo": "Ruospo (activ-mag DMR) [DDECS'22]",
        "ahmadilivani": "Ahmadilivani (grad+EDAC) [IOLTS'24]",
        "ranger": "Ranger (uniform range guard) [DSN'21]",
        "vf_greedy": "VF-only greedy (ablation)",
        "uniform_dmr": "Uniform DMR (no ranking)",
        "ilp_static": "ILP-static (Taylor proxy)",
        "ilp_v2": "ILP-v2 (two-tier CF+RG)",
    }
    for method in sorted(str(value) for value in table["method"].unique()):
        if method.startswith("bounded_loss_memory_ilp"):
            labels[method] = "Bounded-loss memory ILP"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, memory in zip(axes.ravel(), memories):
        subset = table[table["peak_memory_budget_bytes"] == memory]
        for method, label in labels.items():
            values = subset[subset["method"] == method].sort_values("latency_budget_ms")
            if values.empty:
                continue
            ax.plot(values["latency_budget_ms"], values["mitigation_ratio"], marker="o", label=label)
        ax.set_title(f"{title_prefix}: peak extra memory {memory / (1024 * 1024):.0f} MiB")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Latency overhead budget (ms)")
    axes[1, 1].set_xlabel("Latency overhead budget (ms)")
    axes[0, 0].set_ylabel("Mitigated critical-failure fraction")
    axes[1, 0].set_ylabel("Mitigated critical-failure fraction")
    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
