"""Generate vulnerability-concentration tables and figures from screening results."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from research.reliability.profiling.concentration import summarize_concentration


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze concentration of task-level failures among runtime objects.")
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="resnet50")
    parser.add_argument("--title", default="ResNet50 EuroSAT task-failure concentration")
    args = parser.parse_args()

    ranked = pd.read_csv(args.ranked_csv)
    ranked["retained_after_passes"] = ranked["retained_after_passes"].astype(str).str.lower().eq("true")
    scopes = {
        "semantic_candidates": ranked,
        "runtime_protectable": ranked[ranked["retained_after_passes"]].copy(),
    }
    summaries = {}
    curve_frames = []
    for scope, frame in scopes.items():
        summary, curves = summarize_concentration(frame, scope=scope)
        summaries[scope] = asdict(summary)
        curve_frames.append(curves)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.concat(curve_frames, ignore_index=True)
    curves.to_csv(output_dir / f"{args.prefix}_concentration_curves.csv", index=False)
    (output_dir / f"{args.prefix}_concentration_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    protectable = curves[curves["scope"] == "runtime_protectable"]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for ranking, label, color in (
        ("task_risk", "Task-risk ranking", "#1f5a94"),
        ("activation_bytes", "Activation-bytes ranking", "#dc7c24"),
    ):
        series = protectable[protectable["ranking"] == ranking]
        ax.step(
            [0.0, *series["candidate_fraction"]],
            [0.0, *series["cumulative_failure_coverage"]],
            where="post",
            label=label,
            linewidth=2.0,
            color=color,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Uniform/random expectation")
    ax.set_xlabel("Fraction of protectable runtime objects selected")
    ax.set_ylabel("Cumulative critical-failure coverage")
    ax.set_title(args.title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{args.prefix}_concentration_curve.png", dpi=220)
    plt.close(fig)
    print(json.dumps(summaries["runtime_protectable"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
