"""Small boundary experiment for persistent ONNX weight bit flips.

This is deliberately a boundary/proxy experiment, not the thesis main fault
model. It flips single bits in ONNX FP32 initializer tensors, runs ONNX Runtime
on fixed input samples, and reports output changes. The result documents why
persistent weight corruption is treated separately from transient runtime
activation faults.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto


def _infer_input_shape(model: onnx.ModelProto) -> list[int]:
    graph_inputs = {item.name for item in model.graph.input}
    initializers = {item.name for item in model.graph.initializer}
    real_inputs = [item for item in model.graph.input if item.name in graph_inputs - initializers]
    if not real_inputs:
        raise ValueError("ONNX model has no non-initializer graph input.")
    shape = []
    for dim in real_inputs[0].type.tensor_type.shape.dim:
        value = dim.dim_value
        if not value:
            raise ValueError("Input shape is dynamic; pass --input-shape.")
        shape.append(int(value))
    return shape


def _parse_shape(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part) for part in value.replace("x", ",").split(",") if part.strip()]


def _session_output(model_path: Path, input_bin: Path, input_shape: list[int]) -> np.ndarray:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    data = np.fromfile(input_bin, dtype=np.float32).reshape(input_shape)
    outputs = sess.run(None, {input_name: data})
    return np.asarray(outputs[0], dtype=np.float32)


def _eligible_initializers(model: onnx.ModelProto) -> list[tuple[int, int]]:
    eligible: list[tuple[int, int]] = []
    for index, tensor in enumerate(model.graph.initializer):
        if tensor.data_type != TensorProto.FLOAT:
            continue
        if tensor.raw_data:
            eligible.append((index, len(tensor.raw_data)))
    if not eligible:
        raise ValueError("No FP32 raw_data initializer was found.")
    return eligible


def _weighted_choice(rng: random.Random, eligible: list[tuple[int, int]]) -> int:
    total = sum(size for _idx, size in eligible)
    pick = rng.randrange(total)
    acc = 0
    for idx, size in eligible:
        acc += size
        if pick < acc:
            return idx
    return eligible[-1][0]


def _flip_initializer_bit(
    model: onnx.ModelProto,
    rng: random.Random,
    bit_index: int | None,
) -> dict[str, Any]:
    eligible = _eligible_initializers(model)
    tensor_index = _weighted_choice(rng, eligible)
    tensor = model.graph.initializer[tensor_index]
    raw = bytearray(tensor.raw_data)
    element_count = len(raw) // 4
    element_index = rng.randrange(element_count)
    bit = rng.randrange(32) if bit_index is None else bit_index
    byte_offset = element_index * 4 + bit // 8
    mask = 1 << (bit % 8)
    before = raw[byte_offset]
    raw[byte_offset] ^= mask
    after = raw[byte_offset]
    tensor.raw_data = bytes(raw)
    return {
        "initializer_name": tensor.name,
        "initializer_index": tensor_index,
        "initializer_raw_bytes": len(raw),
        "element_index": element_index,
        "bit_index": bit,
        "byte_offset": byte_offset,
        "byte_before": before,
        "byte_after": after,
    }


def _summarize_output(baseline: np.ndarray, corrupted: np.ndarray) -> dict[str, Any]:
    diff = np.abs(corrupted - baseline)
    baseline_flat = baseline.reshape(-1)
    corrupted_flat = corrupted.reshape(-1)
    summary: dict[str, Any] = {
        "output_shape": list(baseline.shape),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "num_abs_diff_gt_1e_3": int((diff > 1e-3).sum()),
        "num_abs_diff_gt_1e_2": int((diff > 1e-2).sum()),
        "global_argmax_before": int(np.argmax(baseline_flat)) if baseline_flat.size else -1,
        "global_argmax_after": int(np.argmax(corrupted_flat)) if corrupted_flat.size else -1,
    }
    summary["global_argmax_changed"] = summary["global_argmax_before"] != summary["global_argmax_after"]
    if baseline.ndim == 2 and baseline.shape[0] == 1 and baseline.shape[1] <= 1000:
        summary["classification_top1_before"] = int(np.argmax(baseline[0]))
        summary["classification_top1_after"] = int(np.argmax(corrupted[0]))
        summary["classification_top1_changed"] = (
            summary["classification_top1_before"] != summary["classification_top1_after"]
        )
    return summary


def run_experiment(
    model_path: Path,
    input_bin: Path,
    out_dir: Path,
    trials: int,
    seed: int,
    input_shape: list[int] | None,
    bit_index: int | None,
) -> dict[str, Any]:
    model = onnx.load(str(model_path), load_external_data=True)
    shape = input_shape or _infer_input_shape(model)
    baseline = _session_output(model_path, input_bin, shape)
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="spinnv2_weight_fault_") as tmp:
        tmp_dir = Path(tmp)
        for trial in range(trials):
            corrupted = onnx.load(str(model_path), load_external_data=True)
            fault = _flip_initializer_bit(corrupted, rng, bit_index)
            corrupt_path = tmp_dir / f"{model_path.stem}_weight_fault_{trial:03d}.onnx"
            onnx.save(corrupted, str(corrupt_path))
            corrupted_output = _session_output(corrupt_path, input_bin, shape)
            event = {
                "trial": trial,
                "fault": fault,
                "output": _summarize_output(baseline, corrupted_output),
            }
            events.append(event)

    changed_top1 = sum(1 for item in events if item["output"].get("classification_top1_changed"))
    global_argmax_changed = sum(1 for item in events if item["output"]["global_argmax_changed"])
    max_diffs = [item["output"]["max_abs_diff"] for item in events]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "persistent_weight_fault_boundary_proxy",
        "interpretation": (
            "This experiment documents output sensitivity to persistent ONNX initializer corruption. "
            "It is a boundary experiment and is not used as the thesis main transient activation fault model."
        ),
        "model": str(model_path),
        "input_bin": str(input_bin),
        "input_shape": shape,
        "seed": seed,
        "trials": trials,
        "bit_index": "random" if bit_index is None else bit_index,
        "events": events,
        "summary": {
            "classification_top1_changed": changed_top1,
            "global_argmax_changed": global_argmax_changed,
            "max_abs_diff_max": float(max(max_diffs)) if max_diffs else 0.0,
            "max_abs_diff_mean": float(np.mean(max_diffs)) if max_diffs else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="ONNX model path.")
    parser.add_argument("--input-bin", required=True, help="FP32 input tensor binary.")
    parser.add_argument("--out", required=True, help="Output JSON report.")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--input-shape", default=None, help="Optional shape, e.g. 1,3,224,224.")
    parser.add_argument("--bit-index", type=int, default=None, choices=range(32), help="Fixed bit index; default random.")
    args = parser.parse_args()

    out = Path(args.out)
    report = run_experiment(
        model_path=Path(args.model),
        input_bin=Path(args.input_bin),
        out_dir=out.parent,
        trials=args.trials,
        seed=args.seed,
        input_shape=_parse_shape(args.input_shape),
        bit_index=args.bit_index,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out), "summary": report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
