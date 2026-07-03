"""Build ranked runtime-object risk records from JSONL injection observations."""

from __future__ import annotations

import argparse
import json

from research.reliability.profiling.risk_profile import InjectionObservation, build_risk_profile, write_risk_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("observations", help="JSONL records with node/tensor/activation/failure fields.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--critical-weight", type=float, default=0.65)
    parser.add_argument("--severity-weight", type=float, default=0.25)
    parser.add_argument("--exposure-weight", type=float, default=0.10)
    args = parser.parse_args()
    with open(args.observations, encoding="utf-8") as source:
        observations = []
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            observations.append(
                InjectionObservation(
                    node_id=int(item["node_id"]),
                    tensor_id=int(item["tensor_id"]),
                    activation_bytes=int(item["activation_bytes"]),
                    critical_failure=bool(item["critical_failure"]),
                    severity=float(item["severity"]),
                )
            )
    records = build_risk_profile(
        observations,
        critical_weight=args.critical_weight,
        severity_weight=args.severity_weight,
        exposure_weight=args.exposure_weight,
    )
    write_risk_profile(records, args.output)
    print(json.dumps({"nodes": len(records), "highest_risk_node": records[0].node_id, "risk": records[0].risk}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
