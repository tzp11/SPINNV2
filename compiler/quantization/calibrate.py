"""Post-training calibration for INT8 quantization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CalibrationResult:
    tensor_id: int
    scale: float
    zero_point: int = 0


@dataclass
class CalibrationProfile:
    activation_scales: dict[int, CalibrationResult] = field(default_factory=dict)
    num_samples: int = 0


def calibrate_activations(
    onnx_path: str | Path,
    calibration_data: list[np.ndarray],
) -> CalibrationProfile:
    """Run FP32 inference on calibration data and collect activation ranges.

    Uses onnxruntime to execute the model. For each intermediate tensor,
    tracks running max(|value|) across all samples. Computes
    scale = max_abs / 127 (symmetric quantization).
    """
    import onnxruntime as ort

    onnx_path = str(onnx_path)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    all_node_names = [n.name for n in sess.get_outputs()]
    try:
        all_node_names += [n for n in sess._outputs_meta if n not in all_node_names]
    except Exception:
        pass

    max_abs: dict[str, float] = {}

    for sample in calibration_data:
        if sample.ndim == 3:
            sample = sample[np.newaxis, ...]
        outputs = sess.run(None, {input_name: sample.astype(np.float32)})
        for name, arr in zip(all_node_names, outputs):
            val = float(np.max(np.abs(arr)))
            if name in max_abs:
                max_abs[name] = max(max_abs[name], val)
            else:
                max_abs[name] = val

    profile = CalibrationProfile(num_samples=len(calibration_data))
    for tid_str, abs_max in max_abs.items():
        try:
            tid = int(tid_str)
        except ValueError:
            continue
        scale = abs_max / 127.0 if abs_max > 0 else 1.0
        profile.activation_scales[tid] = CalibrationResult(
            tensor_id=tid, scale=scale
        )

    return profile


def calibrate_from_graph(
    graph,
    calibration_data: list[np.ndarray],
    onnx_path: str | Path,
    method: str = "minmax",
) -> CalibrationProfile:
    """Simplified calibration that assigns per-tensor scale to Conv input tensors.

    For each Conv node, we need the activation scale of its input tensor.
    This uses onnxruntime with intermediate output capture.

    Parameters
    ----------
    method : ``"minmax"`` (default) or ``"entropy"`` (KL-divergence threshold).
    """
    if method == "entropy":
        from .kl_calibrate import calibrate_from_graph_kl
        return calibrate_from_graph_kl(graph, calibration_data, onnx_path)

    import onnx
    import onnxruntime as ort

    onnx_path = str(onnx_path)
    model = onnx.load(onnx_path)

    conv_input_names: set[str] = set()
    name_to_tid: dict[str, int] = {}
    for tensor in graph.tensors:
        name_to_tid[tensor.name] = tensor.id
    for node in graph.nodes:
        if node.op_type == "Conv" and node.inputs:
            input_tensor = graph.tensors[node.inputs[0]]
            conv_input_names.add(input_tensor.name)

    for name in conv_input_names:
        try:
            model.graph.output.append(
                onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
            )
        except Exception:
            pass

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = f.name
        onnx.save(model, tmp_path)

    try:
        sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_names = [o.name for o in sess.get_outputs()]

        max_abs: dict[str, float] = {}
        for sample in calibration_data:
            if sample.ndim == 3:
                sample = sample[np.newaxis, ...]
            outputs = sess.run(output_names, {input_name: sample.astype(np.float32)})
            for name, arr in zip(output_names, outputs):
                val = float(np.max(np.abs(arr)))
                if name in max_abs:
                    max_abs[name] = max(max_abs[name], val)
                else:
                    max_abs[name] = val
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    profile = CalibrationProfile(num_samples=len(calibration_data))
    for name, abs_max in max_abs.items():
        if name in name_to_tid:
            tid = name_to_tid[name]
            scale = abs_max / 127.0 if abs_max > 0 else 1.0
            profile.activation_scales[tid] = CalibrationResult(
                tensor_id=tid, scale=scale
            )

    return profile


def save_calibration(profile: CalibrationProfile, path: str | Path) -> None:
    data = {
        "num_samples": profile.num_samples,
        "activation_scales": {
            str(tid): {"tensor_id": r.tensor_id, "scale": r.scale, "zero_point": r.zero_point}
            for tid, r in profile.activation_scales.items()
        },
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path: str | Path) -> CalibrationProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    profile = CalibrationProfile(num_samples=data["num_samples"])
    for tid_str, entry in data["activation_scales"].items():
        tid = int(tid_str)
        profile.activation_scales[tid] = CalibrationResult(
            tensor_id=entry["tensor_id"],
            scale=entry["scale"],
            zero_point=entry.get("zero_point", 0),
        )
    return profile
