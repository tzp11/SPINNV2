"""Unit tests for --optimization-level CLI and pipeline_for_level()."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.passes.manager import (
    DEFAULT_PIPELINE,
    OPTIMIZATION_LEVELS,
    pipeline_for_level,
)


def test_level_0_empty():
    assert pipeline_for_level(0) == []


def test_level_1_basic():
    result = pipeline_for_level(1)
    assert len(result) == 4
    assert "EliminateIdentityDropout" in result
    assert "ConstantFold" in result
    assert "FuseConvBatchNorm" in result
    assert "EliminateDead" in result
    assert "FuseConvRelu" not in result
    assert "FuseConvAdd" not in result


def test_level_2_full():
    assert pipeline_for_level(2) == list(DEFAULT_PIPELINE)


def test_invalid_level_raises():
    with pytest.raises(ValueError, match="unknown optimization level"):
        pipeline_for_level(3)
    with pytest.raises(ValueError, match="unknown optimization level"):
        pipeline_for_level(-1)


def test_pipeline_for_level_returns_copy():
    p1 = pipeline_for_level(2)
    p2 = pipeline_for_level(2)
    assert p1 is not p2
    p1.append("extra")
    assert len(pipeline_for_level(2)) == len(DEFAULT_PIPELINE)


def _make_single_conv_onnx(tmp_dir: Path) -> Path:
    np.random.seed(42)
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    w_init = numpy_helper.from_array(
        np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1, name="W"
    )
    b_init = numpy_helper.from_array(np.zeros(16, dtype=np.float32), name="B")
    conv = helper.make_node(
        "Conv", ["X", "W", "B"], ["Y"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph([conv], "test", [x_info], [y_info], initializer=[w_init, b_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_dir / "conv.onnx"
    onnx.save(model, str(path))
    return path


def test_cli_compile_O0(tmp_path):
    onnx_path = _make_single_conv_onnx(tmp_path)
    spk_path = tmp_path / "out.spk"
    stats_path = tmp_path / "stats.json"
    result = subprocess.run(
        [sys.executable, "-m", "spinnv2.compiler", "compile",
         str(onnx_path), "-o", str(spk_path),
         "--target", "cpu_generic",
         "-O", "0",
         "--pass-stats-json", str(stats_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert spk_path.exists()
    import json
    stats = json.loads(stats_path.read_text())
    assert stats == [], "O0 should produce no pass stats"


def test_cli_compile_O1(tmp_path):
    onnx_path = _make_single_conv_onnx(tmp_path)
    spk_path = tmp_path / "out.spk"
    stats_path = tmp_path / "stats.json"
    result = subprocess.run(
        [sys.executable, "-m", "spinnv2.compiler", "compile",
         str(onnx_path), "-o", str(spk_path),
         "--target", "cpu_generic",
         "-O", "1",
         "--pass-stats-json", str(stats_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    import json
    stats = json.loads(stats_path.read_text())
    pass_names = [s["name"] for s in stats]
    assert "EliminateIdentityDropout" in pass_names
    assert "ConstantFold" in pass_names
    assert "FuseConvBatchNorm" in pass_names
    assert "EliminateDead" in pass_names
    assert "FuseConvRelu" not in pass_names
    assert "FuseConvAdd" not in pass_names
