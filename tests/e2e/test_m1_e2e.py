from __future__ import annotations

import subprocess
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from compiler.frontend.onnx_importer import import_onnx


def test_m1_tiny_cnn_e2e(tmp_path: Path):
    model_path = tmp_path / "tiny_cnn.onnx"
    spk_path = tmp_path / "tiny_cnn.spk"
    csv_path = tmp_path / "memory_plan.csv"
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"

    _write_tiny_cnn(model_path)

    graph = import_onnx(model_path)
    assert [node.op_type for node in graph.nodes] == ["Conv", "Relu", "MaxPool", "Flatten", "Gemm", "Softmax"]
    subprocess.run(
        [
            "python",
            "-m",
            "spinnv2.compiler",
            "compile",
            str(model_path),
            "-o",
            str(spk_path),
            "--memory-plan-csv",
            str(csv_path),
        ],
        check=True,
    )
    debug = json.loads(spk_path.with_suffix(".spk.json").read_text(encoding="utf-8"))
    assert debug["memory"]["planned_activation_bytes"] < debug["memory"]["naive_activation_bytes"]
    assert "pass_stats" in debug["metadata"]
    assert csv_path.exists()

    x = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_path.write_bytes(np.ascontiguousarray(x).tobytes())

    runner = Path("build/runtime/spkv2_run")
    subprocess.run(["cmake", "-S", "runtime", "-B", "build/runtime"], check=True)
    subprocess.run(["cmake", "--build", "build/runtime"], check=True)

    subprocess.run([str(runner), str(spk_path), str(input_path), str(output_path)], check=True)

    actual = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    expected = _run_ort(model_path, x).reshape(-1)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    bad_spk_path = tmp_path / "tiny_cnn_bad_checksum.spk"
    bad_spk = bytearray(spk_path.read_bytes())
    bad_spk[100] ^= 0x1
    bad_spk_path.write_bytes(bad_spk)
    bad_run = subprocess.run([str(runner), str(bad_spk_path), str(input_path), str(output_path)])
    assert bad_run.returncode != 0


def test_m3_conv_bn_relu_fusion_e2e(tmp_path: Path):
    model_path = tmp_path / "conv_bn_relu.onnx"
    spk_path = tmp_path / "conv_bn_relu.spk"
    pass_stats_path = tmp_path / "pass_stats.json"
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"

    _write_conv_bn_relu(model_path)

    subprocess.run(
        [
            "python",
            "-m",
            "spinnv2.compiler",
            "compile",
            str(model_path),
            "-o",
            str(spk_path),
            "--pass-stats-json",
            str(pass_stats_path),
        ],
        check=True,
    )
    debug = json.loads(spk_path.with_suffix(".spk.json").read_text(encoding="utf-8"))
    assert [node["op_type"] for node in debug["nodes"]] == ["Conv"]
    assert debug["nodes"][0]["attrs"]["fused_activation"] == "Relu"
    pass_stats = json.loads(pass_stats_path.read_text(encoding="utf-8"))
    assert any(result["name"] == "FuseConvBatchNorm" and result["changed"] == 1 for result in pass_stats)
    assert any(result["name"] == "FuseConvRelu" and result["changed"] == 1 for result in pass_stats)

    x = np.array([[[[-2.0, -0.5], [0.25, 2.0]]]], dtype=np.float32)
    input_path.write_bytes(np.ascontiguousarray(x).tobytes())

    runner = Path("build/runtime/spkv2_run")
    subprocess.run(["cmake", "-S", "runtime", "-B", "build/runtime"], check=True)
    subprocess.run(["cmake", "--build", "build/runtime"], check=True)
    subprocess.run([str(runner), str(spk_path), str(input_path), str(output_path)], check=True)

    actual = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    expected = _run_ort(model_path, x).reshape(-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_m4_cpu_generic_kernel_spec_e2e(tmp_path: Path):
    model_path = tmp_path / "tiny_cnn.onnx"
    spk_path = tmp_path / "tiny_cnn_cpu_generic.spk"
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"

    _write_tiny_cnn(model_path)
    subprocess.run(
        [
            "python",
            "-m",
            "spinnv2.compiler",
            "compile",
            str(model_path),
            "-o",
            str(spk_path),
            "--target",
            "cpu_generic",
        ],
        check=True,
    )
    debug = json.loads(spk_path.with_suffix(".spk.json").read_text(encoding="utf-8"))
    kernel_specs = debug["metadata"]["kernel_specs"]
    assert any(spec["op_type"] == "Conv" and spec["kernel_kind"] == "im2col_gemm" for spec in kernel_specs)
    assert any(spec["op_type"] == "Gemm" and spec["kernel_kind"] == "direct" for spec in kernel_specs)
    assert debug["metadata"]["scratch_arena_bytes"] > 0

    x = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_path.write_bytes(np.ascontiguousarray(x).tobytes())

    runner = Path("build/runtime/spkv2_run")
    subprocess.run(["cmake", "-S", "runtime", "-B", "build/runtime"], check=True)
    subprocess.run(["cmake", "--build", "build/runtime"], check=True)
    subprocess.run([str(runner), str(spk_path), str(input_path), str(output_path)], check=True)

    actual = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    expected = _run_ort(model_path, x).reshape(-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_runtime_uses_graph_output_order(tmp_path: Path):
    model_path = tmp_path / "multi_output_order.onnx"
    spk_path = tmp_path / "multi_output_order.spk"
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"

    _write_multi_output_order(model_path)
    subprocess.run(
        [
            "python",
            "-m",
            "spinnv2.compiler",
            "compile",
            str(model_path),
            "-o",
            str(spk_path),
        ],
        check=True,
    )
    debug = json.loads(spk_path.with_suffix(".spk.json").read_text(encoding="utf-8"))
    graph_outputs = debug["outputs"]
    assert graph_outputs[0] > graph_outputs[1]

    x = np.array([[-2.0, -0.5, 0.25, 2.0]], dtype=np.float32)
    input_path.write_bytes(np.ascontiguousarray(x).tobytes())

    runner = Path("build/runtime/spkv2_run")
    subprocess.run(["cmake", "-S", "runtime", "-B", "build/runtime"], check=True)
    subprocess.run(["cmake", "--build", "build/runtime"], check=True)
    subprocess.run([str(runner), str(spk_path), str(input_path), str(output_path)], check=True)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    expected_first = session.run(None, {"input": x})[0].astype(np.float32).reshape(-1)
    actual = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    np.testing.assert_allclose(actual, expected_first, rtol=1e-6, atol=1e-6)


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


def _write_multi_output_order(path: Path) -> None:
    nodes = [
        helper.make_node("Relu", ["input"], ["early_output"]),
        helper.make_node("Sigmoid", ["early_output"], ["late_output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "multi_output_order",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [
            helper.make_tensor_value_info("late_output", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("early_output", TensorProto.FLOAT, [1, 4]),
        ],
        [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _write_conv_bn_relu(path: Path) -> None:
    conv_w = np.array([[[[1.5]]], [[[-2.0]]]], dtype=np.float32)
    conv_b = np.array([0.25, -0.5], dtype=np.float32)
    scale = np.array([1.25, 0.75], dtype=np.float32)
    bn_bias = np.array([0.1, -0.2], dtype=np.float32)
    mean = np.array([0.5, -0.25], dtype=np.float32)
    var = np.array([0.75, 1.5], dtype=np.float32)

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "conv_w", "conv_b"],
            ["conv_out"],
            kernel_shape=[1, 1],
        ),
        helper.make_node(
            "BatchNormalization",
            ["conv_out", "scale", "bn_bias", "mean", "var"],
            ["bn_out"],
            epsilon=1e-5,
        ),
        helper.make_node("Relu", ["bn_out"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes,
        "conv_bn_relu",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2, 2])],
        [
            numpy_helper.from_array(conv_w, "conv_w"),
            numpy_helper.from_array(conv_b, "conv_b"),
            numpy_helper.from_array(scale, "scale"),
            numpy_helper.from_array(bn_bias, "bn_bias"),
            numpy_helper.from_array(mean, "mean"),
            numpy_helper.from_array(var, "var"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
