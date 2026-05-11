from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from compiler.frontend.onnx_importer import import_onnx


def test_m1_tiny_cnn_e2e(tmp_path: Path):
    model_path = tmp_path / "tiny_cnn.onnx"
    spk_path = tmp_path / "tiny_cnn.spk"
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"

    _write_tiny_cnn(model_path)

    graph = import_onnx(model_path)
    assert [node.op_type for node in graph.nodes] == ["Conv", "Relu", "MaxPool", "Flatten", "Gemm", "Softmax"]
    subprocess.run(
        ["python", "-m", "spinnv2.compiler", "compile", str(model_path), "-o", str(spk_path)],
        check=True,
    )

    x = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_path.write_bytes(np.ascontiguousarray(x).tobytes())

    runner = Path("build/runtime/spkv2_run")
    subprocess.run(["cmake", "-S", "runtime", "-B", "build/runtime"], check=True)
    subprocess.run(["cmake", "--build", "build/runtime"], check=True)

    subprocess.run([str(runner), str(spk_path), str(input_path), str(output_path)], check=True)

    actual = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    expected = _run_ort(model_path, x).reshape(-1)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def _run_ort(model_path: Path, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output = session.run(None, {"input": x})[0]
    return output.astype(np.float32)


def _write_tiny_cnn(path: Path) -> None:
    conv_w = (np.arange(18, dtype=np.float32).reshape(2, 1, 3, 3) - 8.0) / 16.0
    conv_b = np.array([0.1, -0.2], dtype=np.float32)
    gemm_w = (np.arange(24, dtype=np.float32).reshape(8, 3) - 12.0) / 20.0
    gemm_b = np.array([0.05, -0.03, 0.01], dtype=np.float32)

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "conv_w", "conv_b"],
            ["conv_out"],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            kernel_shape=[3, 3],
        ),
        helper.make_node("Relu", ["conv_out"], ["relu_out"]),
        helper.make_node(
            "MaxPool",
            ["relu_out"],
            ["pool_out"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        ),
        helper.make_node("Flatten", ["pool_out"], ["flat_out"], axis=1),
        helper.make_node("Gemm", ["flat_out", "gemm_w", "gemm_b"], ["gemm_out"], alpha=1.0, beta=1.0),
        helper.make_node("Softmax", ["gemm_out"], ["output"], axis=1),
    ]

    graph = helper.make_graph(
        nodes,
        "tiny_cnn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
        [
            numpy_helper.from_array(conv_w, "conv_w"),
            numpy_helper.from_array(conv_b, "conv_b"),
            numpy_helper.from_array(gemm_w, "gemm_w"),
            numpy_helper.from_array(gemm_b, "gemm_b"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
