"""Tests for FP16 mixed-precision weight storage."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from tests.e2e.test_m1_e2e import _run_ort, _write_tiny_cnn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BUILD_DIR = _PROJECT_ROOT / "build"
_RUNNER = _BUILD_DIR / "spkv2_run"


def _compile_spk(model_path: Path, spk_path: Path, fp16: bool = False) -> None:
    cmd = [
        sys.executable, "-m", "compiler.cli", "compile",
        str(model_path), "-o", str(spk_path), "--target", "cpu_generic",
    ]
    if fp16:
        cmd.append("--fp16-weights")
    subprocess.run(cmd, check=True, capture_output=True)


def _run_spk(spk_path: Path, input_path: Path, output_path: Path) -> np.ndarray:
    subprocess.run(
        [str(_RUNNER), str(spk_path), str(input_path), str(output_path)],
        check=True, capture_output=True,
    )
    return np.frombuffer(output_path.read_bytes(), dtype=np.float32)


def test_fp16_compiler_converts_weights():
    """FP16 pass converts weight tensors to half precision."""
    from compiler.frontend.onnx_importer import import_onnx
    from compiler.ir import types
    from compiler.passes.convert_fp16 import convert_weights_fp16

    model_path = _BUILD_DIR / "codegen_test" / "tiny_cnn.onnx"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tiny_cnn(model_path)

    graph = import_onnx(model_path)
    fp32_weights = [t for t in graph.tensors if t.role == types.ROLE_WEIGHT and t.dtype == types.DTYPE_FP32]
    multi_dim_count = sum(1 for t in fp32_weights if len(t.shape) >= 2)

    count = convert_weights_fp16(graph)
    assert count == multi_dim_count

    for t in graph.tensors:
        if t.role == types.ROLE_WEIGHT and len(t.shape) >= 2:
            assert t.dtype == types.DTYPE_FP16
            assert t.size_bytes == t.elem_count * 2
        elif t.role == types.ROLE_WEIGHT and len(t.shape) < 2:
            assert t.dtype == types.DTYPE_FP32


def test_fp16_spk_size_reduction():
    """FP16 SPK should be significantly smaller than FP32."""
    model_path = _BUILD_DIR / "codegen_test" / "tiny_cnn.onnx"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tiny_cnn(model_path)

    fp32_spk = _BUILD_DIR / "fp16_test_fp32.spk"
    fp16_spk = _BUILD_DIR / "fp16_test_fp16.spk"
    _compile_spk(model_path, fp32_spk, fp16=False)
    _compile_spk(model_path, fp16_spk, fp16=True)

    fp32_size = fp32_spk.stat().st_size
    fp16_size = fp16_spk.stat().st_size
    assert fp16_size < fp32_size


def test_fp16_inference_matches_fp32():
    """FP16 inference output should closely match FP32."""
    model_path = _BUILD_DIR / "codegen_test" / "tiny_cnn.onnx"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tiny_cnn(model_path)

    fp32_spk = _BUILD_DIR / "fp16_test_fp32.spk"
    fp16_spk = _BUILD_DIR / "fp16_test_fp16.spk"
    _compile_spk(model_path, fp32_spk, fp16=False)
    _compile_spk(model_path, fp16_spk, fp16=True)

    x = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_path = _BUILD_DIR / "fp16_test_input.bin"
    input_path.write_bytes(x.tobytes())

    out32 = _run_spk(fp32_spk, input_path, _BUILD_DIR / "fp16_test_out32.bin")
    out16 = _run_spk(fp16_spk, input_path, _BUILD_DIR / "fp16_test_out16.bin")

    cos = np.dot(out16, out32) / (np.linalg.norm(out16) * np.linalg.norm(out32) + 1e-10)
    assert cos > 0.9999, f"cosine similarity too low: {cos}"
    np.testing.assert_allclose(out16, out32, rtol=1e-2, atol=1e-3)


def test_fp16_inference_matches_ort():
    """FP16 inference should match ORT reference within tolerance."""
    model_path = _BUILD_DIR / "codegen_test" / "tiny_cnn.onnx"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tiny_cnn(model_path)

    fp16_spk = _BUILD_DIR / "fp16_test_fp16.spk"
    _compile_spk(model_path, fp16_spk, fp16=True)

    x = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_path = _BUILD_DIR / "fp16_test_input.bin"
    input_path.write_bytes(x.tobytes())

    ort_out = _run_ort(model_path, x).reshape(-1)
    spk_out = _run_spk(fp16_spk, input_path, _BUILD_DIR / "fp16_test_ort.bin")

    cos = np.dot(spk_out, ort_out) / (np.linalg.norm(spk_out) * np.linalg.norm(ort_out) + 1e-10)
    assert cos > 0.9999, f"cosine similarity vs ORT too low: {cos}"


def test_fp16_biases_stay_fp32():
    """Bias tensors (1-D) should remain FP32 after FP16 conversion."""
    from compiler.frontend.onnx_importer import import_onnx
    from compiler.ir import types
    from compiler.passes.convert_fp16 import convert_weights_fp16

    model_path = _BUILD_DIR / "codegen_test" / "tiny_cnn.onnx"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tiny_cnn(model_path)

    graph = import_onnx(model_path)
    convert_weights_fp16(graph)

    for t in graph.tensors:
        if t.role == types.ROLE_WEIGHT and len(t.shape) == 1:
            assert t.dtype == types.DTYPE_FP32, f"bias {t.name} was incorrectly converted to FP16"
