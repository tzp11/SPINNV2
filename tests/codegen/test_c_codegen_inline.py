from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from tests.e2e.test_m1_e2e import _run_ort, _write_tiny_cnn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TEST_DIR = _PROJECT_ROOT / "build" / "codegen_inline_test_pytest"


def _compile_spk(model_path: Path, spk_path: Path) -> None:
    subprocess.run(
        [
            sys.executable, "-m", "spinnv2.compiler",
            "compile", str(model_path), "-o", str(spk_path),
            "--target", "cpu_generic",
            "--external-inputs", "--external-outputs",
        ],
        check=True,
    )


def _build_and_run(gen_dir: Path, build_dir: Path, input_path: Path,
                   output_path: Path, extra_args: list[str] | None = None) -> None:
    subprocess.run(["cmake", "-S", str(gen_dir), "-B", str(build_dir),
                     "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release"], check=True)
    cmd = [str(build_dir / "tiny_main_test"), str(input_path), str(output_path)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def test_inline_codegen_embedded_weights():
    test_dir = _TEST_DIR / "embedded"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    model_path = test_dir / "tiny_cnn.onnx"
    spk_path = test_dir / "tiny_cnn.spk"
    gen_dir = test_dir / "generated"
    build_dir = test_dir / "build"
    input_path = test_dir / "input.bin"
    output_path = test_dir / "output.bin"

    _write_tiny_cnn(model_path)
    _compile_spk(model_path, spk_path)

    subprocess.run(
        [
            sys.executable, "-m", "spinnv2.compiler",
            "codegen", str(spk_path),
            "--out-dir", str(gen_dir),
            "--name", "tiny",
            "--runtime-dir", "runtime",
            "--inline",
        ],
        check=True,
    )

    src = (gen_dir / "tiny.c").read_text(encoding="utf-8")
    assert "g_tiny_w" in src or "g_tiny_loaded_w" not in src
    assert "int tiny_init(void)" in src

    x = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_path.write_bytes(np.ascontiguousarray(x).tobytes())

    _build_and_run(gen_dir, build_dir, input_path, output_path)

    actual = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    expected = _run_ort(model_path, x).reshape(-1)
    np.testing.assert_allclose(actual, expected, rtol=0.05, atol=0.02)


def test_inline_codegen_external_weights():
    test_dir = _TEST_DIR / "external"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    model_path = test_dir / "tiny_cnn.onnx"
    spk_path = test_dir / "tiny_cnn.spk"
    gen_dir = test_dir / "generated"
    build_dir = test_dir / "build"
    input_path = test_dir / "input.bin"
    output_path = test_dir / "output.bin"

    _write_tiny_cnn(model_path)
    _compile_spk(model_path, spk_path)

    subprocess.run(
        [
            sys.executable, "-m", "spinnv2.compiler",
            "codegen", str(spk_path),
            "--out-dir", str(gen_dir),
            "--name", "tiny",
            "--runtime-dir", "runtime",
            "--inline",
            "--external-weights",
        ],
        check=True,
    )

    src = (gen_dir / "tiny.c").read_text(encoding="utf-8")
    assert "g_tiny_loaded_w" in src
    assert "int tiny_init(const char *weights_path)" in src
    assert (gen_dir / "tiny_weights.bin").exists()

    x = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_path.write_bytes(np.ascontiguousarray(x).tobytes())

    _build_and_run(gen_dir, build_dir, input_path, output_path,
                   extra_args=[str(gen_dir / "tiny_weights.bin")])

    actual = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    expected = _run_ort(model_path, x).reshape(-1)
    np.testing.assert_allclose(actual, expected, rtol=0.05, atol=0.02)
