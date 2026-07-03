"""Unit tests for KL-divergence (entropy) calibration."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.quantization.kl_calibrate import (
    _find_kl_threshold,
    NUM_BINS_DEFAULT,
    NUM_QUANT_BINS,
)


def test_kl_threshold_uniform():
    """Uniform distribution: threshold should cover most of the range.

    With uniform data, clipping any portion is costly. The algorithm should
    select a high threshold (covering most bins) even though quantization
    error is higher, because clipping cost dominates.
    """
    num_bins = 2048
    hist = np.ones(num_bins, dtype=np.float64) * 100
    t = _find_kl_threshold(hist, num_bins)
    assert t >= num_bins * 0.5, f"Uniform dist threshold {t} too low (expected >= {num_bins * 0.5})"


def test_kl_threshold_long_tail():
    """Exponential (long-tail) distribution: threshold should clip the tail.

    Most mass is in early bins; the long tail should be clipped to improve
    quantization precision in the high-mass region.
    """
    num_bins = 2048
    bins = np.arange(num_bins, dtype=np.float64)
    hist = np.exp(-bins / 200.0) * 10000
    hist = hist.astype(np.float64)
    t = _find_kl_threshold(hist, num_bins)
    assert t < num_bins, f"Long-tail threshold {t} should be < {num_bins}"
    assert t > NUM_QUANT_BINS + 10, f"Threshold {t} should be well above {NUM_QUANT_BINS}"


def test_kl_threshold_better_than_minmax():
    """For a distribution with sparse outliers, KL should clip the tail.

    Exponential core (99.5% mass in first 30% of bins) with sparse
    outliers scattered in the remaining 70%.  KL should find a threshold
    well below num_bins, giving a tighter scale than min-max.
    """
    num_bins = 2048
    hist = np.zeros(num_bins, dtype=np.float64)
    bins = np.arange(num_bins, dtype=np.float64)
    hist[:600] = np.exp(-bins[:600] / 100.0) * 10000
    rng = np.random.RandomState(42)
    outlier_positions = rng.randint(600, num_bins, size=50)
    for pos in outlier_positions:
        hist[pos] += rng.uniform(1, 20)

    t = _find_kl_threshold(hist, num_bins)
    assert t < num_bins * 0.8, (
        f"KL threshold {t} should be < {num_bins * 0.8} (should clip sparse outliers)"
    )
    assert t > NUM_QUANT_BINS + 10, (
        f"KL threshold {t} should cover the core distribution (> {NUM_QUANT_BINS + 10})"
    )


def test_kl_threshold_all_zeros():
    """All-zero histogram: should return num_bins (no crash)."""
    num_bins = 2048
    hist = np.zeros(num_bins, dtype=np.float64)
    t = _find_kl_threshold(hist, num_bins)
    assert t == num_bins


def test_kl_threshold_single_bin():
    """All values in one bin: threshold should be >= num_quant_bins."""
    num_bins = 2048
    hist = np.zeros(num_bins, dtype=np.float64)
    hist[0] = 10000
    t = _find_kl_threshold(hist, num_bins)
    assert t >= NUM_QUANT_BINS


def test_entropy_calibration_integration():
    """Integration: calibrate_from_graph with method='entropy' on a synthetic Conv."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    from compiler.frontend.onnx_importer import import_onnx
    from compiler.passes.manager import DEFAULT_PIPELINE, run_pass_pipeline
    from compiler.target.profile import load_target_profile
    from compiler.quantization.calibrate import calibrate_from_graph

    np.random.seed(42)
    C_in, C_out, H, W = 8, 16, 8, 8
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, C_in, H, W])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    w_init = numpy_helper.from_array(
        np.random.randn(C_out, C_in, 3, 3).astype(np.float32) * 0.1, name="W"
    )
    b_init = numpy_helper.from_array(np.zeros(C_out, dtype=np.float32), name="B")
    conv = helper.make_node(
        "Conv", ["X", "W", "B"], ["Y"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1],
    )
    graph_def = helper.make_graph([conv], "test", [x_info], [y_info], initializer=[w_init, b_init])
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
        onnx.save(model, onnx_path)

    try:
        profile = load_target_profile("cpu_generic")
        graph = import_onnx(onnx_path)
        run_pass_pipeline(graph, pipeline=list(DEFAULT_PIPELINE), enabled=True, profile=profile)

        cal_data = [np.random.randn(1, C_in, H, W).astype(np.float32) * 0.1 for _ in range(5)]

        minmax_result = calibrate_from_graph(graph, cal_data, onnx_path, method="minmax")
        entropy_result = calibrate_from_graph(graph, cal_data, onnx_path, method="entropy")

        assert len(entropy_result.activation_scales) > 0, "Entropy calibration should produce scales"
        assert entropy_result.num_samples == 5

        for tid, cal_res in entropy_result.activation_scales.items():
            assert cal_res.scale > 0, f"Scale for tensor {tid} should be positive"
            assert cal_res.zero_point == 0, "Symmetric quantization: zero_point should be 0"

        for tid in minmax_result.activation_scales:
            assert tid in entropy_result.activation_scales, (
                f"Entropy should cover same tensors as minmax (missing {tid})"
            )
    finally:
        Path(onnx_path).unlink(missing_ok=True)
