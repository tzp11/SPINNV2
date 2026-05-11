#!/usr/bin/env python3
"""Run the M6 validation pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/m6_models", help="M6 report/output directory.")
    parser.add_argument("--tables-dir", default="build/m6_paper_tables", help="Paper table output directory.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest and CTest.")
    parser.add_argument("--skip-large-run", action="store_true", help="Compile M6 models but skip runtime execution.")
    args = parser.parse_args()

    commands: list[list[str]] = []
    if not args.skip_tests:
        commands.extend(
            [
                ["cmake", "-S", "runtime", "-B", "build/runtime"],
                ["cmake", "--build", "build/runtime"],
                ["pytest", "tests/compiler", "tests/e2e", "tests/codegen"],
                ["ctest", "--test-dir", "build/runtime"],
                ["python", "tests/e2e/run_e2e.py", "--all-small", "--out-dir", "build/e2e_all_small"],
                ["python", "benchmarks/run_memory.py", "--models", "mnist,lenet,resnet18"],
                ["python", "benchmarks/run_latency.py", "--models", "mnist,lenet"],
                ["python", "tests/codegen/run_codegen_test.py", "--models", "mnist,lenet"],
            ]
        )

    run_m6 = [
        "python",
        "benchmarks/run_m6_models.py",
        "--out-dir",
        args.out_dir,
    ]
    if args.skip_large_run:
        run_m6.append("--skip-run")
    commands.append(run_m6)

    report = str(Path(args.out_dir) / "m6_report.json")
    commands.append(["python", "scripts/export_paper_tables.py", report, "--out-dir", args.tables_dir])
    check = ["python", "scripts/check_reproducibility.py", report]
    if args.skip_large_run:
        check.append("--allow-skip-run")
    commands.append(check)

    for command in commands:
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
