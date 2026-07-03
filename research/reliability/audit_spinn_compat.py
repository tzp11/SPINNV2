"""Audit an exported ONNX graph against the SPINNV2 import/runtime surface."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx

try:
    from compiler.frontend.onnx_importer import LOWERABLE_IMPORT_OPS
except ImportError:
    LOWERABLE_IMPORT_OPS: set[str] = set()
from compiler.ir.types import SUPPORTED_IMPORT_OPS, SUPPORTED_M1_OPS


def audit_model(path: str | Path) -> dict:
    model = onnx.load(path)
    counts = Counter(node.op_type for node in model.graph.node)
    imported_unsupported = sorted(set(counts) - SUPPORTED_IMPORT_OPS - LOWERABLE_IMPORT_OPS)
    runtime_unsupported = sorted(set(counts) - SUPPORTED_M1_OPS - LOWERABLE_IMPORT_OPS)
    return {
        "model": str(path),
        "node_count": sum(counts.values()),
        "op_counts": dict(sorted(counts.items())),
        "unsupported_import_ops": imported_unsupported,
        "lowered_import_ops": sorted(set(counts) & LOWERABLE_IMPORT_OPS),
        "compiler_only_or_unsupported_runtime_ops": runtime_unsupported,
        "import_compatible": not imported_unsupported,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_model(args.model)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["import_compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
