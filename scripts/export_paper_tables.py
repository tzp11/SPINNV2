#!/usr/bin/env python3
"""Export paper-ready M6 tables from an M6 report JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Input m6_report.json produced by benchmarks/run_m6_models.py.")
    parser.add_argument("--out-dir", default="build/m6_paper_tables", help="Output directory.")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "correctness": correctness_rows(report),
        "memory": memory_rows(report),
        "ops": op_rows(report),
        "freeze": freeze_rows(report),
    }
    for name, rows in tables.items():
        write_csv(out_dir / f"{name}.csv", rows)
        write_markdown(out_dir / f"{name}.md", rows)

    summary = {
        "source_report": str(Path(args.report)),
        "tables": {name: f"{name}.csv" for name in tables},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote paper tables to {out_dir}")
    return 0


def correctness_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model_name, model in report.get("models", {}).items():
        compare = model.get("compare", {})
        runtime = model.get("runtime", {})
        row = {
            "model": model_name,
            "runtime_s": fmt(runtime.get("time_s")),
            "max_abs_error": fmt(compare.get("max_abs_error")),
            "mean_abs_error": fmt(compare.get("mean_abs_error")),
            "top1_equal": compare.get("top1_equal", ""),
            "score_max_abs_error": fmt(compare.get("score_max_abs_error")),
            "score_mean_abs_error": fmt(compare.get("score_mean_abs_error")),
            "top10_max_abs_error": fmt(compare.get("top10_max_abs_error")),
            "top10_same_class_count": compare.get("top10_same_class_count", ""),
        }
        rows.append(row)
    return rows


def memory_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model_name, model in report.get("models", {}).items():
        memory = model.get("memory", {})
        naive = memory.get("naive_activation_bytes")
        planned = memory.get("planned_activation_bytes")
        reduction = memory.get("memory_reduction_ratio")
        rows.append(
            {
                "model": model_name,
                "spk_size_bytes": model.get("spk_size_bytes", ""),
                "naive_activation_bytes": naive if naive is not None else "",
                "planned_activation_bytes": planned if planned is not None else "",
                "memory_reduction_ratio": fmt(reduction),
            }
        )
    return rows


def op_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    all_ops = sorted({op for model in report.get("models", {}).values() for op in model.get("op_counts", {})})
    for model_name, model in report.get("models", {}).items():
        row = {"model": model_name}
        counts = model.get("op_counts", {})
        for op in all_ops:
            row[op] = counts.get(op, 0)
        row["total_nodes"] = sum(counts.values())
        rows.append(row)
    return rows


def freeze_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    models = sorted(report.get("models", {}))
    return [
        {"item": "SIR/SPK version", "status": "frozen_for_m6", "value": "SPK 0.2"},
        {"item": "Runtime API", "status": "frozen_for_m6", "value": "spkv2_load/prepare/run/bind/get_output"},
        {"item": "Target profile", "status": "frozen_for_m6", "value": report.get("target", "cpu_ref")},
        {"item": "Experiment models", "status": "frozen_for_m6", "value": ", ".join(models)},
        {"item": "Benchmark script", "status": "frozen_for_m6", "value": "benchmarks/run_m6_models.py"},
        {"item": "Paper table exporter", "status": "frozen_for_m6", "value": "scripts/export_paper_tables.py"},
        {"item": "Reproducibility checker", "status": "frozen_for_m6", "value": "scripts/check_reproducibility.py"},
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
