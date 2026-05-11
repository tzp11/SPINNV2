from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_m6_table_export_and_repro_check_accept_valid_report(tmp_path: Path):
    report_path = tmp_path / "m6_report.json"
    tables_dir = tmp_path / "tables"
    report_path.write_text(json.dumps(_valid_report()), encoding="utf-8")

    subprocess.run(
        ["python", "scripts/export_paper_tables.py", str(report_path), "--out-dir", str(tables_dir)],
        check=True,
    )
    subprocess.run(["python", "scripts/check_reproducibility.py", str(report_path)], check=True)

    assert (tables_dir / "correctness.csv").exists()
    assert (tables_dir / "memory.md").exists()
    assert "resnet101" in (tables_dir / "correctness.md").read_text(encoding="utf-8")
    assert "YOLO" not in (tables_dir / "freeze.md").read_text(encoding="utf-8")


def test_m6_repro_check_rejects_failed_threshold(tmp_path: Path):
    report = _valid_report()
    report["models"]["yolov10n"]["compare"]["score_max_abs_error"] = 1.0
    report_path = tmp_path / "m6_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = subprocess.run(
        ["python", "scripts/check_reproducibility.py", str(report_path)],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "score max error" in result.stderr


def _valid_report() -> dict:
    return {
        "target": "cpu_ref",
        "models": {
            "resnet101": {
                "op_counts": {"Conv": 104},
                "compile": {"returncode": 0},
                "runtime": {"returncode": 0, "time_s": 1.0},
                "memory": {
                    "naive_activation_bytes": 100,
                    "planned_activation_bytes": 10,
                    "memory_reduction_ratio": 0.9,
                },
                "compare": {
                    "max_abs_error": 0.01,
                    "mean_abs_error": 0.001,
                    "top1_equal": True,
                },
            },
            "yolov10n": {
                "op_counts": {"Conv": 83, "TopK": 2},
                "compile": {"returncode": 0},
                "runtime": {"returncode": 0, "time_s": 1.0},
                "memory": {
                    "naive_activation_bytes": 100,
                    "planned_activation_bytes": 20,
                    "memory_reduction_ratio": 0.8,
                },
                "compare": {
                    "max_abs_error": 1.0,
                    "mean_abs_error": 0.1,
                    "score_max_abs_error": 1e-8,
                    "score_mean_abs_error": 1e-9,
                    "top10_max_abs_error": 1e-3,
                    "top10_mean_abs_error": 1e-4,
                    "top10_same_class_count": 10,
                },
            },
        },
    }
