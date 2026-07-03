"""Aggregate Windows-stage engineering metrics for thesis tables."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def _bench_fields(report: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    if not report:
        return {
            f"{prefix}_avg_ms": None,
            f"{prefix}_p50_ms": None,
            f"{prefix}_p90_ms": None,
            f"{prefix}_p95_ms": None,
            f"{prefix}_p99_ms": None,
            f"{prefix}_max_ms": None,
        }
    return {
        f"{prefix}_avg_ms": report.get("avg_ms"),
        f"{prefix}_p50_ms": report.get("p50_ms"),
        f"{prefix}_p90_ms": report.get("p90_ms"),
        f"{prefix}_p95_ms": report.get("p95_ms"),
        f"{prefix}_p99_ms": report.get("p99_ms"),
        f"{prefix}_max_ms": report.get("max_ms"),
    }


def _spk_memory(debug: dict[str, Any] | None) -> dict[str, Any]:
    if not debug:
        return {}
    memory = debug.get("memory") or {}
    metadata = debug.get("metadata") or {}
    plan = debug.get("protection_plan") or {}
    return {
        "naive_activation_bytes": memory.get("naive_activation_bytes"),
        "planned_activation_bytes": memory.get("planned_activation_bytes"),
        "memory_reduction_ratio": memory.get("memory_reduction_ratio"),
        "scratch_arena_bytes": metadata.get("scratch_arena_bytes"),
        "protection_scratch_bytes": memory.get("protection_scratch_bytes"),
        "protected_node_count": len(plan.get("nodes", [])) if plan else 0,
        "protection_modes": sorted({node.get("mode") for node in plan.get("nodes", [])}) if plan else [],
    }


def _gap_report(root: Path) -> str:
    codegen = root / "compiler/codegen/c_codegen.py"
    text = codegen.read_text(encoding="utf-8") if codegen.exists() else ""
    has_checksum = "verify_checksum" in text and "spkv2_load_memory" in text
    has_runtime_run = "spkv2_run" in text
    mentions_fault = "spkv2_set_fault_event" in text
    mentions_stats = "spkv2_get_reliability_stats" in text
    smoke = _read_json(root / "artifacts/reports/paper_assets/generated_c_protection_smoke.json")
    return "\n".join(
        [
            "# Generated C 保护路径状态",
            "",
            f"- codegen file: `{codegen}`",
            f"- embedded SPK checksum verification: `{has_checksum}`",
            f"- generated wrapper calls `spkv2_run`: `{has_runtime_run}`",
            f"- generated wrapper exposes fault-event control: `{mentions_fault}`",
            f"- generated wrapper exposes reliability stats: `{mentions_stats}`",
            f"- protected generated C smoke status: `{smoke.get('status') if smoke else 'not_run'}`",
            "",
            "## 结论",
            "",
            (
                "Generated C 目前通过嵌入 SPK 并调用 Runtime 执行，因此可继承 SPK 内部的 ProtectionPlan 执行语义。"
                if has_checksum and has_runtime_run
                else "Generated C 当前缺少足够证据证明可继承受保护执行语义。"
            ),
            "",
            "但当前生成接口未直接暴露 fault event 与 reliability stats，因此还不能作为批量注入实验入口。论文中应表述为：Generated C 可用于部署受保护模型，批量故障评估仍以 Runtime/ctypes/CLI 路径完成。",
            "",
            "## 补充说明",
            "",
            (
                f"- ResNet50 protected Generated C smoke 已通过：`{smoke['resnet50']['executable']}`。"
                if smoke
                else "- 尚未记录 protected Generated C smoke。"
            ),
            (
                f"- YOLOv10n protected Generated C smoke 已通过：`{smoke['yolov10n']['executable']}`。"
                if smoke and "yolov10n" in smoke
                else ""
            ),
            "- 先前 ResNet50 的 C1060 来自旧版 codegen 将 94MB SPK 展开为约 603MB C 数组；当前默认外置 SPK 资产模式已修复该问题。",
            "",
        ]
    )


def build_report(root: Path) -> dict[str, Any]:
    resnet_base_spk = root / "artifacts/spk/resnet50_eurosat_gpu_prevalidation_cpu_generic.spk"
    resnet_prot_spk = root / "artifacts/spk/resnet50_eurosat_gpu_prevalidation_avx2_feedback2_ilp_20ms_4mib_runtime_calibrated_val512.spk"
    yolo_base_spk = root / "artifacts/spk/yolov10n_dior_full_e30_b16_cpu_generic.spk"
    yolo_prot_spk = root / "artifacts/spk/yolov10n_dior_full_e30_b16_ilp_dmr_20pct_4mb_runtime_calibrated_cpu_generic.spk"

    resnet_summary = _read_json(
        root / "artifacts/reports/resnet50_windows_avx2_release/feedback2_calibrated/resnet50_final_runtime_summary.json"
    )
    yolo_base_bench = _read_json(root / "artifacts/reports/yolov10n_dior_full_e30_b16/baseline_runtime_bench_50runs.json")
    yolo_prot_bench = _read_json(
        root / "artifacts/reports/yolov10n_dior_full_e30_b16/ilp_dmr_runtime_calibrated_final_spk_bench_50runs.json"
    )
    yolo_activation = _read_json(
        root / "artifacts/reports/yolov10n_dior_full_e30_b16/ilp_dmr_activation_prior_comparison/yolov10n_control_dmr_comparison.json"
    )
    yolo_stratified = _read_json(
        root / "artifacts/reports/yolov10n_dior_full_e30_b16/ilp_dmr_stratified_comparison/yolov10n_control_dmr_comparison.json"
    )

    rows: list[dict[str, Any]] = []
    resnet_row: dict[str, Any] = {
        "model": "resnet50_eurosat",
        "baseline_spk_bytes": _size(resnet_base_spk),
        "protected_spk_bytes": _size(resnet_prot_spk),
        "plan_json_bytes": _size(
            root
            / "artifacts/reports/resnet50_windows_avx2_release/feedback2_calibrated/resnet50_representative_ilp_plan.json"
        ),
    }
    resnet_row.update(_spk_memory(_read_json(resnet_prot_spk.with_suffix(".spk.json"))))
    if resnet_summary:
        clean = resnet_summary["clean_evaluation"]
        fault = resnet_summary["fault_evaluation"]
        budget = resnet_summary["budget_optimization"]
        resnet_row.update(
            {
                "baseline_avg_ms": clean.get("baseline_latency_ms"),
                "protected_avg_ms": clean.get("protected_latency_ms"),
                "latency_overhead_ms": clean.get("latency_overhead_ms"),
                "latency_overhead_ratio": clean.get("latency_overhead_ratio"),
                "clean_false_alarm_rate": clean.get("false_alarm_rate"),
                "detected_faults": fault.get("detected_faults"),
                "recovered_faults": fault.get("recovered_faults"),
                "unprotected_critical_failures": fault.get("unprotected_critical_failures"),
                "protected_critical_failures": fault.get("protected_critical_failures"),
                "observed_reduction_ratio": fault.get("observed_reduction_ratio"),
                "used_peak_extra_memory_bytes": budget.get("used_peak_extra_memory_bytes"),
            }
        )
    rows.append(resnet_row)

    yolo_row: dict[str, Any] = {
        "model": "yolov10n_dior_full_e30_b16",
        "baseline_spk_bytes": _size(yolo_base_spk),
        "protected_spk_bytes": _size(yolo_prot_spk),
        "plan_json_bytes": _size(root / "artifacts/plans/yolov10n_dior_full_e30_b16_ilp_dmr_20pct_4mb_runtime_calibrated.json"),
    }
    yolo_row.update(_spk_memory(_read_json(yolo_prot_spk.with_suffix(".spk.json"))))
    yolo_row.update(_bench_fields(yolo_base_bench, "baseline"))
    yolo_row.update(_bench_fields(yolo_prot_bench, "protected"))
    if yolo_base_bench and yolo_prot_bench:
        yolo_row["latency_overhead_ms"] = yolo_prot_bench.get("avg_ms") - yolo_base_bench.get("avg_ms")
        yolo_row["latency_overhead_ratio"] = yolo_row["latency_overhead_ms"] / yolo_base_bench.get("avg_ms")
    if yolo_activation:
        yolo_row["activation_prior_unprotected_critical_failures"] = yolo_activation["baseline"]["critical_failures"]
        yolo_row["activation_prior_protected_critical_failures"] = yolo_activation["budget_ilp_dmr"]["critical_failures"]
        yolo_row["activation_prior_reduction_ratio"] = yolo_activation["critical_failure_reduction_ratio"]
    if yolo_stratified:
        yolo_row["stratified_unprotected_critical_failures"] = yolo_stratified["baseline"]["critical_failures"]
        yolo_row["stratified_protected_critical_failures"] = yolo_stratified["budget_ilp_dmr"]["critical_failures"]
        yolo_row["stratified_reduction_ratio"] = yolo_stratified["critical_failure_reduction_ratio"]
    rows.append(yolo_row)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "windows_stage_engineering_metrics",
        "notes": [
            "p95/p99 are null when the source benchmark JSON did not record raw timing samples or those quantiles.",
            "Generated C status is a static capability report; batch fault injection still uses Runtime CLI/ctypes.",
        ],
        "rows": rows,
    }


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

    report = build_report(root)
    json_path = out_dir / "engineering_metrics.json"
    csv_path = out_dir / "engineering_metrics.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in report["rows"] for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["rows"])

    gap_path = out_dir / "generated_c_gap_report.md"
    gap_path.write_text(_gap_report(root), encoding="utf-8", newline="\n")

    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "gap_report": str(gap_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
