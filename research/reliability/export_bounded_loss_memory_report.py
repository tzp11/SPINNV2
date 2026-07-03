"""Evaluate bounded-risk-loss memory minimization against the max-risk ILP."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from research.reliability.optimizer.plan_optimizer import (
    CandidateMode,
    optimize_bounded_loss_memory_ilp,
    optimize_ilp,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare max-risk ILP with bounded-loss memory ILP.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="resnet50")
    parser.add_argument("--latency-budgets-ms", default="5,10,20,40,80,160")
    parser.add_argument("--memory-budgets-bytes", default="1048576,4194304,8388608")
    parser.add_argument("--risk-loss-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    candidates = [CandidateMode(**item) for item in json.loads(Path(args.candidates).read_text(encoding="utf-8"))]
    latencies = [float(value) for value in args.latency_budgets_ms.split(",")]
    memories = [int(value) for value in args.memory_budgets_bytes.split(",")]

    rows = []
    for memory in memories:
        for latency in latencies:
            optimum = optimize_ilp(candidates, latency_budget_ms=latency, memory_budget_bytes=memory)
            bounded = optimize_bounded_loss_memory_ilp(
                candidates,
                latency_budget_ms=latency,
                memory_budget_bytes=memory,
                risk_loss_tolerance=args.risk_loss_tolerance,
            )
            risk_retention = (
                bounded.total_risk_reduction / optimum.total_risk_reduction
                if optimum.total_risk_reduction > 0.0
                else 1.0
            )
            rows.append(
                {
                    "latency_budget_ms": latency,
                    "peak_memory_budget_bytes": memory,
                    "ilp_risk_reduction": optimum.total_risk_reduction,
                    "bounded_loss_risk_reduction": bounded.total_risk_reduction,
                    "risk_retention": risk_retention,
                    "risk_loss": 1.0 - risk_retention,
                    "ilp_peak_extra_memory_bytes": optimum.total_extra_memory_bytes,
                    "bounded_loss_peak_extra_memory_bytes": bounded.total_extra_memory_bytes,
                    "peak_memory_saving_bytes": optimum.total_extra_memory_bytes
                    - bounded.total_extra_memory_bytes,
                    "peak_memory_saving_ratio": (
                        (optimum.total_extra_memory_bytes - bounded.total_extra_memory_bytes)
                        / optimum.total_extra_memory_bytes
                        if optimum.total_extra_memory_bytes > 0
                        else 0.0
                    ),
                    "ilp_latency_ms": optimum.total_latency_overhead_ms,
                    "bounded_loss_latency_ms": bounded.total_latency_overhead_ms,
                    "latency_saving_ms": optimum.total_latency_overhead_ms
                    - bounded.total_latency_overhead_ms,
                    "ilp_selected_count": len(optimum.selected),
                    "bounded_loss_selected_count": len(bounded.selected),
                    "bounded_loss_selected_modes": [asdict(choice) for choice in bounded.selected],
                }
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    csv_table = table.drop(columns=["bounded_loss_selected_modes"])
    csv_path = out_dir / f"{args.output_prefix}_bounded_loss_memory_comparison.csv"
    json_path = out_dir / f"{args.output_prefix}_bounded_loss_memory_comparison.json"
    png_path = out_dir / f"{args.output_prefix}_bounded_loss_memory_saving.png"
    csv_table.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "method": "risk-loss-bounded peak-memory minimization",
                "risk_loss_tolerance": args.risk_loss_tolerance,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot(table, memories, png_path)
    summary = {
        "csv": str(csv_path),
        "json": str(json_path),
        "figure": str(png_path),
        "max_peak_memory_saving_ratio": float(table["peak_memory_saving_ratio"].max()),
        "min_risk_retention": float(table["risk_retention"].min()),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _plot(table: pd.DataFrame, memories: list[int], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for memory in memories:
        subset = table[table["peak_memory_budget_bytes"] == memory].sort_values("latency_budget_ms")
        label = f"{memory / (1024 * 1024):.0f} MiB budget"
        axes[0].plot(subset["latency_budget_ms"], subset["peak_memory_saving_ratio"], marker="o", label=label)
        axes[1].plot(subset["latency_budget_ms"], subset["risk_retention"], marker="o", label=label)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Latency overhead budget (ms)")
    axes[0].set_ylabel("Peak extra-memory saving ratio")
    axes[0].grid(alpha=0.25)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Latency overhead budget (ms)")
    axes[1].set_ylabel("Risk-reduction retention")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
