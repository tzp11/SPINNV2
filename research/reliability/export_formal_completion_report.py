"""Aggregate formal Windows completion experiments into thesis-facing reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sum_dicts(docs: list[dict[str, Any]], key: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    for doc in docs:
        for name, value in doc[key].items():
            if isinstance(value, int):
                totals[name] = totals.get(name, 0) + int(value)
    return totals


def build(root: Path) -> dict[str, Any]:
    formal = root / "artifacts/reports/formal_windows_completion"
    resnet_paths = [
        formal / "resnet50_runtime_faults_10000_seed2026.json",
        formal / "resnet50_runtime_faults_10000_seed2027.json",
        formal / "resnet50_runtime_faults_10000_seed2028.json",
    ]
    resnet_docs = [_read_json(path) for path in resnet_paths]
    resnet_totals = _sum_dicts(resnet_docs, "totals")
    resnet_events = resnet_totals["events"]
    resnet_unprotected = resnet_totals["unprotected_critical_failures"]
    resnet_protected = resnet_totals["protected_critical_failures"]

    yolo_base_summary = _read_json(formal / "yolov10n_runtime_fault_activation_prior_10000_seed2026.summary.json")
    yolo_prot_summary = _read_json(
        formal / "yolov10n_runtime_fault_activation_prior_10000_seed2026_protected.summary.json"
    )
    yolo_guard = _read_json(formal / "yolov10n_task_output_guard_10000_seed2026.json")

    rows = [
        {
            "model": "resnet50_eurosat",
            "experiment": "formal_multi_seed_runtime_faults",
            "events": resnet_events,
            "unprotected_critical_failures": resnet_unprotected,
            "protected_method": "multi_mode_ilp_range_guard_dmr",
            "protected_critical_failures": resnet_protected,
            "critical_failure_reduction": resnet_unprotected - resnet_protected,
            "critical_failure_reduction_ratio": (resnet_unprotected - resnet_protected) / resnet_unprotected,
            "seeds": "2026,2027,2028",
            "notes": "Three real 10000-event runs, not bootstrap substitution.",
        },
        {
            "model": "yolov10n_dior_full_e30_b16",
            "experiment": "formal_activation_prior_runtime_faults",
            "events": yolo_base_summary["observations"],
            "unprotected_critical_failures": yolo_base_summary["critical_failures"],
            "protected_method": "ilp_dmr_runtime_calibrated",
            "protected_critical_failures": yolo_prot_summary["critical_failures"],
            "critical_failure_reduction": yolo_base_summary["critical_failures"] - yolo_prot_summary["critical_failures"],
            "critical_failure_reduction_ratio": (
                (yolo_base_summary["critical_failures"] - yolo_prot_summary["critical_failures"])
                / yolo_base_summary["critical_failures"]
            ),
            "seeds": "2026",
            "notes": "Same seed and candidate sampling; activation-prior 10000 events.",
        },
        {
            "model": "yolov10n_dior_full_e30_b16",
            "experiment": "formal_task_output_guard",
            "events": yolo_guard["totals"]["events"],
            "unprotected_critical_failures": yolo_guard["totals"]["faulted_critical_failures"],
            "protected_method": "task_output_guard_rerun",
            "protected_critical_failures": yolo_guard["totals"]["guarded_critical_failures"],
            "critical_failure_reduction": yolo_guard["totals"]["mitigated_failures"],
            "critical_failure_reduction_ratio": yolo_guard["rates"]["observed_reduction_ratio"],
            "seeds": "replay 2026 baseline events",
            "notes": "Final-output semantic validity checks with rerun recovery under single-transient-fault model.",
        },
    ]
    return {
        "scope": "formal_windows_completion_without_d2000_riscv",
        "rows": rows,
        "source_files": {
            "resnet50": [str(path) for path in resnet_paths],
            "yolov10n_baseline": str(formal / "yolov10n_runtime_fault_activation_prior_10000_seed2026.jsonl"),
            "yolov10n_protected": str(formal / "yolov10n_runtime_fault_activation_prior_10000_seed2026_protected.jsonl"),
            "yolov10n_task_output_guard": str(formal / "yolov10n_task_output_guard_10000_seed2026.json"),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Windows 正式完成实验摘要",
        "",
        "| 模型 | 实验 | 事件数 | 未保护失效 | 方法 | 保护后失效 | 下降比例 | 说明 |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {model} | {experiment} | {events} | {unprotected_critical_failures} | {protected_method} | "
            "{protected_critical_failures} | {ratio:.2%} | {notes} |".format(
                ratio=row["critical_failure_reduction_ratio"], **row
            )
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- ResNet50 结论来自 3 个真实 seed、共 30000 个运行时故障事件。",
            "- YOLOv10n activation-prior 结论来自真实 10000 个运行时故障事件。",
            "- YOLOv10n 的 ILP-DMR 在真实 activation-prior 下收益较保守；task_output_guard 对最终输出异常有更高覆盖，但属于检测输出语义检查机制。",
            "- D2000/RISC-V 未验证，不能写成跨平台实机结论。",
        ]
    )
    return "\n".join(lines) + "\n"


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

    report = build(root)
    json_path = out_dir / "formal_windows_completion_summary.json"
    csv_path = out_dir / "formal_windows_completion_summary.csv"
    md_path = out_dir / "formal_windows_completion_summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(report["rows"])
    md_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
