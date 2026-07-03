from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from tests.e2e.test_m1_e2e import _run_ort, _write_tiny_cnn

# Build inside the project tree so enterprise security (云壳) does not block
# binaries compiled in /tmp.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TEST_DIR = _PROJECT_ROOT / "build" / "codegen_test"


def test_generated_c_model_builds_runs_and_rejects_bad_checksum():
    test_dir = _TEST_DIR
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    model_path = test_dir / "tiny_cnn.onnx"
    spk_path = test_dir / "tiny_cnn.spk"
    gen_dir = test_dir / "generated"
    build_dir = test_dir / "generated_build"
    input_path = test_dir / "input.bin"
    output_path = test_dir / "output.bin"

    _write_tiny_cnn(model_path)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "spinnv2.compiler",
            "compile",
            str(model_path),
            "-o",
            str(spk_path),
            "--target",
            "cpu_generic",
            "--external-inputs",
            "--external-outputs",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "spinnv2.compiler",
            "codegen",
            str(spk_path),
            "--out-dir",
            str(gen_dir),
            "--name",
            "tiny",
            "--runtime-dir",
            "runtime",
        ],
        check=True,
    )
    assert (gen_dir / "tiny.c").exists()
    assert (gen_dir / "tiny.h").exists()
    assert "g_tiny_activation_arena" in (gen_dir / "tiny.c").read_text(encoding="utf-8")
    assert "spkv2_prepare_with_scratch" in (gen_dir / "tiny.c").read_text(encoding="utf-8")

    subprocess.run(["cmake", "-S", str(gen_dir), "-B", str(build_dir)], check=True)
    subprocess.run(["cmake", "--build", str(build_dir)], check=True)

    x = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_path.write_bytes(np.ascontiguousarray(x).tobytes())
    subprocess.run(
        [str(build_dir / "tiny_main_test"), str(input_path), str(output_path)],
        check=True,
    )

    actual = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    expected = _run_ort(model_path, x).reshape(-1)
    np.testing.assert_allclose(actual, expected, rtol=0.05, atol=0.02)

    checksum_test = gen_dir / "checksum_test.c"
    checksum_test.write_text(
        """
#include "tiny.h"

int main(void) {
    unsigned char damaged[1] = {0};
    return tiny_verify_checksum(damaged, sizeof(damaged)) == 0 ? 1 : 0;
}
""".lstrip(),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "cc",
            "-I",
            str(gen_dir),
            "-I",
            "runtime/include",
            str(checksum_test),
            str(gen_dir / "tiny.c"),
            str(build_dir / "spkv2_runtime_build" / "libspkv2_runtime.a"),
            "-lm",
        ] + (["-framework", "Accelerate"] if platform.system() == "Darwin" else []) + [
            "-o",
            str(build_dir / "checksum_test"),
        ],
        check=True,
    )
    subprocess.run([str(build_dir / "checksum_test")], check=True)
