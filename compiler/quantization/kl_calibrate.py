"""KL-divergence (entropy) calibration for INT8 quantization.

Finds the optimal clipping threshold per activation tensor by minimizing
KL divergence between the original FP32 distribution and its quantized
approximation.  This produces tighter scales than min-max for tensors with
long-tailed distributions (common after ReLU).

Reference: TensorRT calibration algorithm, ncnn ``ncnn2table.cpp``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .calibrate import CalibrationProfile, CalibrationResult

NUM_BINS_DEFAULT = 2048
NUM_QUANT_BINS = 128


def _find_kl_threshold(
    hist: np.ndarray,
    num_bins: int,
    num_quant_bins: int = NUM_QUANT_BINS,
) -> int:
    """Return the bin index that minimizes KL(reference || quantized) + clipping cost.

    The objective balances two terms:
    - KL divergence: quantization error within the selected threshold range
    - Clipping cost: fraction of probability mass beyond the threshold

    Parameters
    ----------
    hist : 1-D int/float array of length *num_bins* (absolute-value histogram).
    num_bins : number of bins in *hist*.
    num_quant_bins : target number of INT8 buckets (128 for symmetric int8).

    Returns
    -------
    Optimal threshold bin index in [num_quant_bins, num_bins].
    """
    assert len(hist) == num_bins

    total = float(hist.sum())
    if total == 0:
        return num_bins

    best_score = float("inf")
    best_t = num_bins

    for t in range(num_quant_bins, num_bins + 1):
        ref = hist[:t].copy().astype(np.float64)

        ref_sum = ref.sum()
        if ref_sum == 0:
            continue

        bin_width = t / num_quant_bins
        quantized = np.zeros(t, dtype=np.float64)
        for q in range(num_quant_bins):
            lo = int(q * bin_width)
            hi = min(int((q + 1) * bin_width), t)
            if lo >= hi:
                continue
            bucket_sum = ref[lo:hi].sum()
            nonzero_count = np.count_nonzero(ref[lo:hi])
            if nonzero_count > 0:
                avg = bucket_sum / nonzero_count
                for i in range(lo, hi):
                    if ref[i] != 0:
                        quantized[i] = avg

        q_sum = quantized.sum()
        if q_sum == 0:
            continue

        ref_norm = ref / ref_sum
        q_norm = quantized / q_sum

        kl = 0.0
        for i in range(t):
            if ref_norm[i] > 0 and q_norm[i] > 0:
                kl += ref_norm[i] * np.log(ref_norm[i] / q_norm[i])

        clip_fraction = 1.0 - ref_sum / total
        score = kl + clip_fraction

        if score < best_score:
            best_score = score
            best_t = t

    return best_t


def collect_histograms(
    graph,
    calibration_data: list[np.ndarray],
    onnx_path: str | Path,
    num_bins: int = NUM_BINS_DEFAULT,
) -> dict[str, tuple[np.ndarray, float]]:
    """Run ORT inference and collect per-tensor absolute-value histograms.

    Returns ``{tensor_name: (histogram, max_abs)}`` for every Conv input tensor.
    """
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
                if name not in conv_input_names and name not in name_to_tid:
                    continue
                val = float(np.max(np.abs(arr)))
                if name in max_abs:
                    max_abs[name] = max(max_abs[name], val)
                else:
                    max_abs[name] = val

        histograms: dict[str, np.ndarray] = {
            name: np.zeros(num_bins, dtype=np.float64) for name in max_abs
        }

        for sample in calibration_data:
            if sample.ndim == 3:
                sample = sample[np.newaxis, ...]
            outputs = sess.run(output_names, {input_name: sample.astype(np.float32)})
            for name, arr in zip(output_names, outputs):
                if name not in histograms:
                    continue
                abs_vals = np.abs(arr.ravel()).astype(np.float64)
                ma = max_abs[name]
                if ma <= 0:
                    continue
                indices = np.minimum(
                    (abs_vals / ma * num_bins).astype(np.int64),
                    num_bins - 1,
                )
                np.add.at(histograms[name], indices, 1)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    result: dict[str, tuple[np.ndarray, float]] = {}
    for name in histograms:
        result[name] = (histograms[name], max_abs[name])
    return result


def calibrate_from_graph_kl(
    graph,
    calibration_data: list[np.ndarray],
    onnx_path: str | Path,
    num_bins: int = NUM_BINS_DEFAULT,
) -> CalibrationProfile:
    """KL-divergence calibration: find optimal clipping threshold per tensor."""
    histograms = collect_histograms(graph, calibration_data, onnx_path, num_bins)

    name_to_tid: dict[str, int] = {}
    for tensor in graph.tensors:
        name_to_tid[tensor.name] = tensor.id

    profile = CalibrationProfile(num_samples=len(calibration_data))

    for name, (hist, ma) in histograms.items():
        if name not in name_to_tid:
            continue
        tid = name_to_tid[name]

        if ma <= 0:
            scale = 1.0
        else:
            best_t = _find_kl_threshold(hist, num_bins)
            threshold = (best_t / num_bins) * ma
            scale = threshold / 127.0 if threshold > 0 else 1.0

        profile.activation_scales[tid] = CalibrationResult(
            tensor_id=tid, scale=scale
        )

    return profile
