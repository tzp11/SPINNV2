"""Tests for non-Apple ARM optimization and big.LITTLE scheduling APIs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from tests.e2e.test_m1_e2e import _run_ort, _write_tiny_cnn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BUILD_DIR = _PROJECT_ROOT / "build"
_RUNNER = _BUILD_DIR / "spkv2_run"


def _compile_spk(model_path: Path, spk_path: Path, target: str = "cpu_generic") -> None:
    subprocess.run(
        [sys.executable, "-m", "compiler.cli", "compile",
         str(model_path), "-o", str(spk_path), "--target", target],
        check=True, capture_output=True,
    )


def _run_spk(spk_path: Path, input_path: Path, output_path: Path,
             env_extra: dict | None = None) -> np.ndarray:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    subprocess.run(
        [str(_RUNNER), str(spk_path), str(input_path), str(output_path)],
        check=True, capture_output=True, env=env,
    )
    return np.frombuffer(output_path.read_bytes(), dtype=np.float32)


def test_arm_neon_target_listed():
    """cpu_arm_neon target should be available."""
    from compiler.cli import list_targets
    targets = list_targets()
    assert "cpu_arm_neon" in targets


def test_arm_neon_target_compiles():
    """Compile with cpu_arm_neon target should succeed."""
    model_path = _BUILD_DIR / "codegen_test" / "tiny_cnn.onnx"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tiny_cnn(model_path)

    spk_path = _BUILD_DIR / "arm_test.spk"
    _compile_spk(model_path, spk_path, target="cpu_arm_neon")
    assert spk_path.exists()

    meta = json.loads(spk_path.with_suffix(".spk.json").read_text())
    assert len(meta["nodes"]) > 0


def test_arm_neon_inference_matches_ort():
    """ARM NEON target inference should match ORT."""
    model_path = _BUILD_DIR / "codegen_test" / "tiny_cnn.onnx"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tiny_cnn(model_path)

    spk_path = _BUILD_DIR / "arm_test.spk"
    _compile_spk(model_path, spk_path, target="cpu_arm_neon")

    x = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_path = _BUILD_DIR / "arm_test_input.bin"
    input_path.write_bytes(x.tobytes())

    spk_out = _run_spk(spk_path, input_path, _BUILD_DIR / "arm_test_out.bin")
    ort_out = _run_ort(model_path, x).reshape(-1)

    cos = np.dot(spk_out, ort_out) / (np.linalg.norm(spk_out) * np.linalg.norm(ort_out) + 1e-10)
    assert cos > 0.9999, f"cosine similarity too low: {cos}"


def test_gemm_tiling_env_vars():
    """Runtime GEMM tiling via env vars should not affect correctness."""
    model_path = _BUILD_DIR / "codegen_test" / "tiny_cnn.onnx"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tiny_cnn(model_path)

    spk_path = _BUILD_DIR / "arm_test.spk"
    _compile_spk(model_path, spk_path, target="cpu_arm_neon")

    x = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_path = _BUILD_DIR / "arm_test_input.bin"
    input_path.write_bytes(x.tobytes())

    ort_out = _run_ort(model_path, x).reshape(-1)

    for kc, nc in [(64, 64), (128, 128), (256, 512)]:
        out = _run_spk(spk_path, input_path, _BUILD_DIR / "arm_test_tuned.bin",
                       env_extra={"SPKV2_GEMM_KC": str(kc), "SPKV2_GEMM_NC": str(nc)})
        cos = np.dot(out, ort_out) / (np.linalg.norm(out) * np.linalg.norm(ort_out) + 1e-10)
        assert cos > 0.9999, f"KC={kc} NC={nc}: cosine {cos}"


def test_auto_tune_env_vars():
    """Auto-tune via L1/L2 cache size env vars should not affect correctness."""
    model_path = _BUILD_DIR / "codegen_test" / "tiny_cnn.onnx"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tiny_cnn(model_path)

    spk_path = _BUILD_DIR / "arm_test.spk"
    _compile_spk(model_path, spk_path, target="cpu_arm_neon")

    x = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_path = _BUILD_DIR / "arm_test_input.bin"
    input_path.write_bytes(x.tobytes())

    ort_out = _run_ort(model_path, x).reshape(-1)

    for l1, l2, label in [(32768, 262144, "A55"), (65536, 524288, "A76")]:
        out = _run_spk(spk_path, input_path, _BUILD_DIR / "arm_test_auto.bin",
                       env_extra={"SPKV2_L1_CACHE": str(l1), "SPKV2_L2_CACHE": str(l2)})
        cos = np.dot(out, ort_out) / (np.linalg.norm(out) * np.linalg.norm(ort_out) + 1e-10)
        assert cos > 0.9999, f"{label} (L1={l1} L2={l2}): cosine {cos}"


def test_arm_neon_winograd_in_conv_list():
    """cpu_arm_neon profile should include winograd_f43 in Conv kernel list."""
    from compiler.target.profile import load_target_profile
    profile = load_target_profile("cpu_arm_neon")
    conv_kernels = profile["ops"]["Conv"]
    assert "simd_winograd_f43" in conv_kernels
    assert "simd_bnns_fp32" not in conv_kernels
