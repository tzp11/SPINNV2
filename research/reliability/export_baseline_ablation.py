"""Export baseline comparison and risk-score ablation tables from existing reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_table(rows: list[dict[str, Any]], csv_path: Path, json_path: Path, meta: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def _budget_rows(model: str, sweep_csv: Path, representative_latency: float, representative_memory: int) -> list[dict[str, Any]]:
    table = pd.read_csv(sweep_csv)
    subset = table[
        (table["latency_budget_ms"].astype(float) == float(representative_latency))
        & (table["peak_memory_budget_bytes"].astype(int) == int(representative_memory))
    ].copy()
    rows = []
    for row in subset.sort_values("mitigation_ratio", ascending=False).itertuples(index=False):
        rows.append(
            {
                "model": model,
                "comparison_type": "predicted_budget_selection",
                "method": row.method,
                "selected_count": int(row.selected_count),
                "mitigation_ratio": float(row.mitigation_ratio),
                "risk_reduction": float(row.risk_reduction),
                "residual_failure_probability": float(row.residual_failure_probability),
                "used_latency_ms": float(row.used_latency_ms),
                "used_peak_extra_memory_bytes": int(row.used_peak_extra_memory_bytes),
            }
        )
    return rows


def _yolo_runtime_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    activation = _read_json(
        root
        / "artifacts/reports/yolov10n_dior_full_e30_b16/ilp_dmr_activation_prior_comparison/yolov10n_control_dmr_comparison.json"
    )
    stratified = _read_json(
        root
        / "artifacts/reports/yolov10n_dior_full_e30_b16/ilp_dmr_stratified_comparison/yolov10n_control_dmr_comparison.json"
    )
    for sampling, doc in [("activation_prior", activation), ("stratified", stratified)]:
        if not doc:
            continue
        rows.append(
            {
                "model": "yolov10n_dior_full_e30_b16",
                "comparison_type": f"same_event_runtime_{sampling}",
                "method": "no_protection",
                "selected_count": 0,
                "critical_failures": doc["baseline"]["critical_failures"],
                "mitigation_ratio": 0.0,
                "latency_avg_ms": doc["baseline"]["latency_avg_ms"],
                "extra_memory_bytes": 0,
            }
        )
        rows.append(
            {
                "model": "yolov10n_dior_full_e30_b16",
                "comparison_type": f"same_event_runtime_{sampling}",
                "method": "ilp_dmr_runtime_calibrated",
                "selected_count": 40,
                "critical_failures": doc["budget_ilp_dmr"]["critical_failures"],
                "mitigation_ratio": doc["critical_failure_reduction_ratio"],
                "latency_avg_ms": doc["budget_ilp_dmr"]["latency_avg_ms"],
                "extra_memory_bytes": doc["budget_ilp_dmr"]["extra_memory_bytes"],
            }
        )
    control = _read_json(
        root / "artifacts/reports/yolov10n_dior_full_e30_b16/control_dmr_comparison/yolov10n_control_dmr_comparison.json"
    )
    if control:
        rows.append(
            {
                "model": "yolov10n_dior_full_e30_b16",
                "comparison_type": "same_event_runtime_control_path",
                "method": "control_path_dmr",
                "selected_count": 5,
                "critical_failures": control["control_dmr"]["critical_failures"],
                "mitigation_ratio": control["critical_failure_reduction_ratio"],
                "latency_avg_ms": control["control_dmr"]["latency_avg_ms"],
                "extra_memory_bytes": control["control_dmr"]["extra_memory_bytes"],
            }
        )
    return rows


def _ablation_rows(model: str, ranked_csv: Path) -> list[dict[str, Any]]:
    table = pd.read_csv(ranked_csv)
    if table.empty:
        return []
    id_col = "runtime_node_id" if "runtime_node_id" in table.columns else "node_id"
    table = table[table[id_col].notna()].copy()
    if table.empty:
        return []
    activation_total = table["activation_bytes"].sum() if "activation_bytes" in table else 1.0
    variants = {
        "critical_only": (1.0, 0.0, 0.0),
        "severity_only": (0.0, 1.0, 0.0),
        "exposure_only": (0.0, 0.0, 1.0),
        "no_exposure": (0.72, 0.28, 0.0),
        "full_score": (0.65, 0.25, 0.10),
    }
    rows = []
    for name, (cw, sw, ew) in variants.items():
        scored = table.copy()
        if "critical_failure_rate" in scored:
            critical_col = "critical_failure_rate"
        elif "p_critical_failure" in scored:
            critical_col = "p_critical_failure"
        else:
            critical_col = "critical_probability"
        severity_col = "mean_severity" if "mean_severity" in scored else "severity_mean"
        scored["_critical"] = scored[critical_col] if critical_col in scored else 0.0
        scored["_severity"] = scored[severity_col] if severity_col in scored else 0.0
        scored["_exposure"] = scored["activation_bytes"] / activation_total if "activation_bytes" in scored else 0.0
        scored["_score"] = cw * scored["_critical"] + sw * scored["_severity"] + ew * scored["_exposure"]
        scored = scored.sort_values("_score", ascending=False)
        top10 = scored.head(10)
        rows.append(
            {
                "model": model,
                "variant": name,
                "top1_node": int(scored.iloc[0][id_col]),
                "top10_nodes": ",".join(str(int(x)) for x in top10[id_col].tolist()),
                "top10_mean_score": float(top10["_score"].mean()),
                "top10_mean_critical_component": float(top10["_critical"].mean()),
                "top10_activation_bytes": int(top10["activation_bytes"].sum()) if "activation_bytes" in top10 else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="artifacts/reports/paper_assets")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    resnet_baselines = _budget_rows(
        "resnet50_eurosat",
        root / "artifacts/reports/resnet50_windows_avx2_release/feedback2_calibrated/resnet50_budget_sweep.csv",
        20.0,
        4194304,
    )
    yolo_baselines = _budget_rows(
        "yolov10n_dior_full_e30_b16",
        root / "artifacts/reports/yolov10n_dior_full_e30_b16/budget_sweep/yolov10n_dior_full_budget_sweep.csv",
        32.0,
        4194304,
    )
    runtime_yolo = _yolo_runtime_rows(root)
    all_baselines = resnet_baselines + yolo_baselines + runtime_yolo
    _write_table(
        all_baselines,
        out_dir / "baseline_comparison.csv",
        out_dir / "baseline_comparison.json",
        {"source": "existing budget sweeps and same-event runtime comparisons"},
    )

    ablations = []
    ablations += _ablation_rows(
        "resnet50_eurosat",
        root / "artifacts/reports/resnet50_gpu_prevalidation/resnet50_risk_ranked.csv",
    )
    ablations += _ablation_rows(
        "yolov10n_dior_full_e30_b16",
        root / "artifacts/reports/yolov10n_dior_full_e30_b16/stratified/yolov10n_runtime_risk_ranked.csv",
    )
    _write_table(
        ablations,
        out_dir / "risk_score_ablation.csv",
        out_dir / "risk_score_ablation.json",
        {"source": "existing ranked risk tables; scoring variants are post-hoc ablations"},
    )
    print(json.dumps({"baseline_rows": len(all_baselines), "ablation_rows": len(ablations)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
