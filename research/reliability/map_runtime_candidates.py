"""Map PyTorch semantic screening outputs onto optimized SPINNV2 runtime nodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


ONNX_CANDIDATE_OPS = {"Conv", "Relu", "Gemm"}


def map_candidates(candidate_map: list[dict], onnx_path: str | Path, spk_debug_path: str | Path) -> dict:
    model = onnx.load(str(onnx_path))
    onnx_candidates = [node for node in model.graph.node if node.op_type in ONNX_CANDIDATE_OPS]
    if len(candidate_map) != len(onnx_candidates):
        raise ValueError(
            f"candidate/ONNX output count differs: {len(candidate_map)} screening points vs {len(onnx_candidates)} ONNX nodes"
        )
    spk_debug = json.loads(Path(spk_debug_path).read_text(encoding="utf-8"))
    tensors_by_name = {tensor["name"]: tensor["id"] for tensor in spk_debug["tensors"]}
    producer_by_tensor = {
        tensor_id: node
        for node in spk_debug["nodes"]
        for tensor_id in node["outputs"]
    }
    mappings = []
    for point, node in zip(candidate_map, onnx_candidates):
        output_name = node.output[0]
        runtime_tensor_id = tensors_by_name.get(output_name)
        runtime_node = producer_by_tensor.get(runtime_tensor_id) if runtime_tensor_id is not None else None
        mapping = dict(point)
        mapping.update(
            {
                "onnx_op_type": node.op_type,
                "onnx_node_name": node.name,
                "onnx_output_name": output_name,
                "retained_after_passes": runtime_node is not None,
                "runtime_node_id": runtime_node["id"] if runtime_node else None,
                "runtime_tensor_id": runtime_tensor_id if runtime_node else None,
                "runtime_op_type": runtime_node["op_type"] if runtime_node else None,
                "runtime_attrs": runtime_node["attrs"] if runtime_node else None,
            }
        )
        mappings.append(mapping)
    return {
        "version": 1,
        "mapping_method": "ordered Conv/Relu/Gemm semantic outputs to ONNX outputs, followed by optimized tensor lookup",
        "candidate_count": len(mappings),
        "retained_count": sum(mapping["retained_after_passes"] for mapping in mappings),
        "mappings": mappings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-map", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--spk-debug", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidate_map = json.loads(Path(args.candidate_map).read_text(encoding="utf-8"))
    report = map_candidates(candidate_map, args.onnx, args.spk_debug)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": report["candidate_count"], "retained_count": report["retained_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
