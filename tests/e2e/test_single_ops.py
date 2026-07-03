"""Per-op accuracy tests: build a minimal ONNX model for each op configuration,
compile → SPK → runtime, and compare output against ONNX Runtime."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper


_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _ROOT / "build" / "runtime" / "spkv2_run"


def _ensure_runtime():
    subprocess.run(
        ["cmake", "-S", str(_ROOT / "runtime"), "-B", str(_ROOT / "build" / "runtime"),
         "-DCMAKE_BUILD_TYPE=Release"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["cmake", "--build", str(_ROOT / "build" / "runtime"), "-j4"],
        check=True, capture_output=True,
    )


def _compile_and_run(model_path: Path, spk_path: Path, x: np.ndarray,
                     target: str = "cpu_generic") -> np.ndarray:
    subprocess.run(
        [sys.executable, "-m", "spinnv2.compiler", "compile",
         str(model_path), "-o", str(spk_path), "--target", target],
        check=True, capture_output=True,
    )
    input_path = spk_path.with_suffix(".input.bin")
    output_path = spk_path.with_suffix(".output.bin")
    input_path.write_bytes(np.ascontiguousarray(x).tobytes())
    subprocess.run(
        [str(_RUNNER), str(spk_path), str(input_path), str(output_path)],
        check=True,
    )
    return np.frombuffer(output_path.read_bytes(), dtype=np.float32)


def _run_ort(model_path: Path, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session.run(None, {"input": x})[0].astype(np.float32).reshape(-1)


def _save(path, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, path.stem, inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _weight(shape, scale=0.05, offset=0.0):
    return (np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) * scale + offset).astype(np.float32)


@pytest.fixture(scope="module", autouse=True)
def build_runtime():
    _ensure_runtime()


# ── Conv variants ──

CONV_CASES = [
    ("conv_3x3_s1_p1", dict(C_in=4, C_out=8, kH=3, kW=3, sH=1, sW=1, pH=1, pW=1, H=8, W=8, group=1)),
    ("conv_1x1", dict(C_in=8, C_out=4, kH=1, kW=1, sH=1, sW=1, pH=0, pW=0, H=8, W=8, group=1)),
    ("conv_3x3_s2", dict(C_in=4, C_out=8, kH=3, kW=3, sH=2, sW=2, pH=1, pW=1, H=16, W=16, group=1)),
    ("conv_5x5", dict(C_in=2, C_out=4, kH=5, kW=5, sH=1, sW=1, pH=2, pW=2, H=8, W=8, group=1)),
    ("conv_depthwise", dict(C_in=6, C_out=6, kH=3, kW=3, sH=1, sW=1, pH=1, pW=1, H=8, W=8, group=6)),
    ("conv_grouped", dict(C_in=8, C_out=8, kH=3, kW=3, sH=1, sW=1, pH=1, pW=1, H=8, W=8, group=2)),
    ("conv_no_bias", dict(C_in=4, C_out=4, kH=3, kW=3, sH=1, sW=1, pH=1, pW=1, H=8, W=8, group=1, no_bias=True)),
]


@pytest.mark.parametrize("name,cfg", CONV_CASES, ids=[c[0] for c in CONV_CASES])
def test_conv(tmp_path, name, cfg):
    model_path = tmp_path / f"{name}.onnx"
    spk_path = tmp_path / f"{name}.spk"
    no_bias = cfg.pop("no_bias", False)
    C_in, C_out = cfg["C_in"], cfg["C_out"]
    kH, kW = cfg["kH"], cfg["kW"]
    H, W = cfg["H"], cfg["W"]
    sH, sW = cfg["sH"], cfg["sW"]
    pH, pW = cfg["pH"], cfg["pW"]
    group = cfg["group"]
    outH = (H + 2 * pH - kH) // sH + 1
    outW = (W + 2 * pW - kW) // sW + 1
    w = _weight((C_out, C_in // group, kH, kW), 0.02, -0.1)
    inits = [numpy_helper.from_array(w, "w")]
    inputs_list = ["input", "w"]
    if not no_bias:
        b = np.linspace(-0.05, 0.05, num=C_out, dtype=np.float32)
        inits.append(numpy_helper.from_array(b, "b"))
        inputs_list.append("b")
    _save(
        model_path,
        [helper.make_node("Conv", inputs_list, ["output"],
                          kernel_shape=[kH, kW], strides=[sH, sW],
                          pads=[pH, pW, pH, pW], group=group)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C_in, H, W])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C_out, outH, outW])],
        inits,
    )
    x = np.linspace(-1, 1, num=C_in * H * W, dtype=np.float32).reshape(1, C_in, H, W)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


# ── Gemm ──

GEMM_CASES = [
    ("gemm_basic", dict(M=1, K=8, N=4, alpha=1.0, beta=1.0, has_bias=True)),
    ("gemm_no_bias", dict(M=1, K=8, N=4, alpha=1.0, beta=1.0, has_bias=False)),
    ("gemm_alpha_beta", dict(M=1, K=4, N=3, alpha=0.5, beta=2.0, has_bias=True)),
]


@pytest.mark.parametrize("name,cfg", GEMM_CASES, ids=[c[0] for c in GEMM_CASES])
def test_gemm(tmp_path, name, cfg):
    model_path = tmp_path / f"{name}.onnx"
    spk_path = tmp_path / f"{name}.spk"
    M, K, N = cfg["M"], cfg["K"], cfg["N"]
    w = _weight((K, N), 0.05, -0.2)
    inits = [numpy_helper.from_array(w, "w")]
    inputs_list = ["input", "w"]
    kwargs = dict(alpha=cfg["alpha"], beta=cfg["beta"])
    if cfg["has_bias"]:
        b = np.linspace(-0.1, 0.1, num=N, dtype=np.float32)
        inits.append(numpy_helper.from_array(b, "b"))
        inputs_list.append("b")
    _save(
        model_path,
        [helper.make_node("Gemm", inputs_list, ["output"], **kwargs)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [M, K])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [M, N])],
        inits,
    )
    x = np.linspace(-1, 1, num=M * K, dtype=np.float32).reshape(M, K)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


# ── Elementwise binary ──

BINARY_CASES = [
    ("add", "Add"),
    ("sub", "Sub"),
    ("mul", "Mul"),
    ("div", "Div"),
]


@pytest.mark.parametrize("name,op", BINARY_CASES, ids=[c[0] for c in BINARY_CASES])
def test_binary(tmp_path, name, op):
    model_path = tmp_path / f"{name}.onnx"
    spk_path = tmp_path / f"{name}.spk"
    b = np.array([[0.5, -0.3, 0.7, -0.1]], dtype=np.float32)
    if op == "Div":
        b = np.array([[2.0, -1.5, 0.5, 3.0]], dtype=np.float32)
    _save(
        model_path,
        [helper.make_node(op, ["input", "b"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(b, "b")],
    )
    x = np.array([[-1.0, -0.25, 0.5, 1.0]], dtype=np.float32)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


# ── Activations ──

ACTIVATION_CASES = [
    ("relu", "Relu", {}),
    ("sigmoid", "Sigmoid", {}),
    ("leakyrelu_01", "LeakyRelu", dict(alpha=0.1)),
    ("leakyrelu_03", "LeakyRelu", dict(alpha=0.3)),
]


@pytest.mark.parametrize("name,op,kwargs", ACTIVATION_CASES, ids=[c[0] for c in ACTIVATION_CASES])
def test_activation(tmp_path, name, op, kwargs):
    model_path = tmp_path / f"{name}.onnx"
    spk_path = tmp_path / f"{name}.spk"
    _save(
        model_path,
        [helper.make_node(op, ["input"], ["output"], **kwargs)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])],
        [],
    )
    x = np.linspace(-2.0, 2.0, num=8, dtype=np.float32).reshape(1, 8)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


# ── Clip ──

def test_clip(tmp_path):
    model_path = tmp_path / "clip.onnx"
    spk_path = tmp_path / "clip.spk"
    min_val = np.array(-0.5, dtype=np.float32)
    max_val = np.array(0.5, dtype=np.float32)
    _save(
        model_path,
        [helper.make_node("Clip", ["input", "clip_min", "clip_max"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])],
        [numpy_helper.from_array(min_val, "clip_min"),
         numpy_helper.from_array(max_val, "clip_max")],
    )
    x = np.linspace(-2.0, 2.0, num=8, dtype=np.float32).reshape(1, 8)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


# ── Pooling ──

POOL_CASES = [
    ("maxpool_2x2_s2", "MaxPool", dict(kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0]),
     [1, 4, 4, 4]),
    ("maxpool_3x3_s1_p1", "MaxPool", dict(kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]),
     [1, 4, 8, 8]),
    ("avgpool_3x3_s2_p1", "AveragePool", dict(kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]),
     [1, 4, 4, 4]),
    ("avgpool_2x2_s2", "AveragePool", dict(kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0]),
     [1, 4, 4, 4]),
]


@pytest.mark.parametrize("name,op,kwargs,out_shape", POOL_CASES, ids=[c[0] for c in POOL_CASES])
def test_pool(tmp_path, name, op, kwargs, out_shape):
    model_path = tmp_path / f"{name}.onnx"
    spk_path = tmp_path / f"{name}.spk"
    _save(
        model_path,
        [helper.make_node(op, ["input"], ["output"], **kwargs)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shape)],
        [],
    )
    x = np.linspace(-1, 1, num=4 * 8 * 8, dtype=np.float32).reshape(1, 4, 8, 8)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_global_avgpool(tmp_path):
    model_path = tmp_path / "global_avgpool.onnx"
    spk_path = tmp_path / "global_avgpool.spk"
    _save(
        model_path,
        [helper.make_node("GlobalAveragePool", ["input"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 1, 1])],
        [],
    )
    x = np.linspace(-1, 1, num=4 * 8 * 8, dtype=np.float32).reshape(1, 4, 8, 8)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


# ── Softmax ──

def test_softmax(tmp_path):
    model_path = tmp_path / "softmax.onnx"
    spk_path = tmp_path / "softmax.spk"
    _save(
        model_path,
        [helper.make_node("Softmax", ["input"], ["output"], axis=1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])],
        [],
    )
    x = np.linspace(-2.0, 2.0, num=8, dtype=np.float32).reshape(1, 8)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


# ── Transpose ──

def test_transpose(tmp_path):
    model_path = tmp_path / "transpose.onnx"
    spk_path = tmp_path / "transpose.spk"
    _save(
        model_path,
        [helper.make_node("Transpose", ["input"], ["output"], perm=[0, 2, 3, 1])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 2, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 3, 4])],
        [],
    )
    x = np.linspace(-1, 1, num=24, dtype=np.float32).reshape(1, 4, 2, 3)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# ── Flatten ──

def test_flatten(tmp_path):
    model_path = tmp_path / "flatten.onnx"
    spk_path = tmp_path / "flatten.spk"
    _save(
        model_path,
        [helper.make_node("Flatten", ["input"], ["output"], axis=1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 2, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 24])],
        [],
    )
    x = np.linspace(-1, 1, num=24, dtype=np.float32).reshape(1, 4, 2, 3)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# ── Concat ──

def test_concat(tmp_path):
    model_path = tmp_path / "concat.onnx"
    spk_path = tmp_path / "concat.spk"
    b = np.ones((1, 4), dtype=np.float32) * 0.5
    _save(
        model_path,
        [
            helper.make_node("Relu", ["input"], ["relu_out"]),
            helper.make_node("Concat", ["relu_out", "const_b"], ["output"], axis=1),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])],
        [numpy_helper.from_array(b, "const_b")],
    )
    x = np.array([[-1.0, -0.5, 0.5, 1.0]], dtype=np.float32)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# ── Conv + Activation fusion ──

CONV_FUSED_CASES = [
    ("conv_relu_fused", "Relu"),
    ("conv_sigmoid_fused", "Sigmoid"),
]


@pytest.mark.parametrize("name,act", CONV_FUSED_CASES, ids=[c[0] for c in CONV_FUSED_CASES])
def test_conv_fused_activation(tmp_path, name, act):
    model_path = tmp_path / f"{name}.onnx"
    spk_path = tmp_path / f"{name}.spk"
    w = _weight((4, 2, 3, 3), 0.02, -0.1)
    b = np.linspace(-0.05, 0.05, num=4, dtype=np.float32)
    nodes = [
        helper.make_node("Conv", ["input", "w", "b"], ["conv_out"],
                          kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node(act, ["conv_out"], ["output"]),
    ]
    _save(
        model_path, nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
    )
    x = np.linspace(-1, 1, num=2 * 8 * 8, dtype=np.float32).reshape(1, 2, 8, 8)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


# ── Conv + Add residual fusion ──

def test_conv_add_residual(tmp_path):
    model_path = tmp_path / "conv_add_res.onnx"
    spk_path = tmp_path / "conv_add_res.spk"
    w = _weight((4, 4, 3, 3), 0.01, -0.05)
    b = np.linspace(-0.02, 0.02, num=4, dtype=np.float32)
    nodes = [
        helper.make_node("Relu", ["input"], ["relu_out"]),
        helper.make_node("Conv", ["relu_out", "w", "b"], ["conv_out"],
                          kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Add", ["conv_out", "relu_out"], ["output"]),
    ]
    _save(
        model_path, nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
    )
    x = np.linspace(-1, 1, num=4 * 8 * 8, dtype=np.float32).reshape(1, 4, 8, 8)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


# ── Reshape ──

def test_reshape(tmp_path):
    model_path = tmp_path / "reshape.onnx"
    spk_path = tmp_path / "reshape.spk"
    shape = np.array([1, 2, 12], dtype=np.int64)
    _save(
        model_path,
        [helper.make_node("Reshape", ["input", "shape"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 2, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 12])],
        [numpy_helper.from_array(shape, "shape")],
    )
    x = np.linspace(-1, 1, num=24, dtype=np.float32).reshape(1, 4, 2, 3)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# ── Unsqueeze ──

def test_unsqueeze(tmp_path):
    model_path = tmp_path / "unsqueeze.onnx"
    spk_path = tmp_path / "unsqueeze.spk"
    axes = np.array([0, 3], dtype=np.int64)
    _save(
        model_path,
        [helper.make_node("Unsqueeze", ["input", "axes"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 3, 1])],
        [numpy_helper.from_array(axes, "axes")],
    )
    x = np.linspace(-1, 1, num=12, dtype=np.float32).reshape(4, 3)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# ── Cast ──

def test_cast(tmp_path):
    model_path = tmp_path / "cast.onnx"
    spk_path = tmp_path / "cast.spk"
    _save(
        model_path,
        [helper.make_node("Cast", ["input"], ["output"], to=TensorProto.FLOAT)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])],
        [],
    )
    x = np.linspace(-2, 2, num=8, dtype=np.float32).reshape(1, 8)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# ── Slice ──

def test_slice(tmp_path):
    model_path = tmp_path / "slice.onnx"
    spk_path = tmp_path / "slice.spk"
    starts = np.array([1], dtype=np.int64)
    ends = np.array([3], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    _save(
        model_path,
        [helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4])],
        [numpy_helper.from_array(starts, "starts"),
         numpy_helper.from_array(ends, "ends"),
         numpy_helper.from_array(axes, "axes")],
    )
    x = np.linspace(-1, 1, num=16, dtype=np.float32).reshape(1, 4, 4)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# ── Split ──

def test_split(tmp_path):
    model_path = tmp_path / "split.onnx"
    spk_path = tmp_path / "split.spk"
    split_sizes = np.array([2, 2], dtype=np.int64)
    _save(
        model_path,
        [helper.make_node("Split", ["input", "split_sizes"], ["out0", "out1"], axis=1),
         helper.make_node("Add", ["out0", "out1"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 3])],
        [numpy_helper.from_array(split_sizes, "split_sizes")],
    )
    x = np.linspace(-1, 1, num=12, dtype=np.float32).reshape(1, 4, 3)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


# ── MatMul ──

MATMUL_CASES = [
    ("matmul_2d", [1, 8], [8, 4], [1, 4]),
    ("matmul_3d", [2, 3, 8], [2, 8, 4], [2, 3, 4]),
]


@pytest.mark.parametrize("name,in_shape,w_shape,out_shape", MATMUL_CASES,
                         ids=[c[0] for c in MATMUL_CASES])
def test_matmul(tmp_path, name, in_shape, w_shape, out_shape):
    model_path = tmp_path / f"{name}.onnx"
    spk_path = tmp_path / f"{name}.spk"
    w = _weight(w_shape, 0.05, -0.2)
    _save(
        model_path,
        [helper.make_node("MatMul", ["input", "w"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, in_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shape)],
        [numpy_helper.from_array(w, "w")],
    )
    x = np.linspace(-1, 1, num=int(np.prod(in_shape)), dtype=np.float32).reshape(in_shape)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


# ── ReduceMean ──

def test_reduce_mean(tmp_path):
    model_path = tmp_path / "reduce_mean.onnx"
    spk_path = tmp_path / "reduce_mean.spk"
    _save(
        model_path,
        [helper.make_node("ReduceMean", ["input"], ["output"], axes=[2, 3], keepdims=1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 1, 1])],
        [],
    )
    x = np.linspace(-1, 1, num=4 * 8 * 8, dtype=np.float32).reshape(1, 4, 8, 8)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


# ── ReduceMax ──

def test_reduce_max(tmp_path):
    model_path = tmp_path / "reduce_max.onnx"
    spk_path = tmp_path / "reduce_max.spk"
    _save(
        model_path,
        [helper.make_node("ReduceMax", ["input"], ["output"], axes=[1], keepdims=1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 3])],
        [],
    )
    x = np.linspace(-2, 2, num=12, dtype=np.float32).reshape(1, 4, 3)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


# ── Resize ──

def test_resize(tmp_path):
    model_path = tmp_path / "resize.onnx"
    spk_path = tmp_path / "resize.spk"
    roi = np.array([], dtype=np.float32)
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 2, 16, 16], dtype=np.int64)
    _save(
        model_path,
        [helper.make_node("Resize", ["input", "roi", "scales", "sizes"], ["output"],
                          mode="nearest")],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 16, 16])],
        [numpy_helper.from_array(roi, "roi"),
         numpy_helper.from_array(scales, "scales"),
         numpy_helper.from_array(sizes, "sizes")],
    )
    x = np.linspace(-1, 1, num=2 * 8 * 8, dtype=np.float32).reshape(1, 2, 8, 8)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


# ── Gather ──

def test_gather(tmp_path):
    model_path = tmp_path / "gather.onnx"
    spk_path = tmp_path / "gather.spk"
    indices = np.array([0, 2], dtype=np.int64)
    _save(
        model_path,
        [helper.make_node("Gather", ["input", "indices"], ["output"], axis=1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 3])],
        [numpy_helper.from_array(indices, "indices")],
    )
    x = np.linspace(-1, 1, num=12, dtype=np.float32).reshape(1, 4, 3)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# ── GatherElements ──

def test_gather_elements(tmp_path):
    model_path = tmp_path / "gather_elements.onnx"
    spk_path = tmp_path / "gather_elements.spk"
    indices = np.array([[1, 0, 1, 0]], dtype=np.int64)
    _save(
        model_path,
        [helper.make_node("GatherElements", ["input", "indices"], ["output"], axis=0)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(indices, "indices")],
    )
    x = np.linspace(-1, 1, num=8, dtype=np.float32).reshape(2, 4)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# ── Tile ──

def test_tile(tmp_path):
    model_path = tmp_path / "tile.onnx"
    spk_path = tmp_path / "tile.spk"
    repeats = np.array([1, 2, 1], dtype=np.int64)
    _save(
        model_path,
        [helper.make_node("Tile", ["input", "repeats"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 6, 4])],
        [numpy_helper.from_array(repeats, "repeats")],
    )
    x = np.linspace(-1, 1, num=12, dtype=np.float32).reshape(1, 3, 4)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# ── Mod ──

def test_mod(tmp_path):
    model_path = tmp_path / "mod.onnx"
    spk_path = tmp_path / "mod.spk"
    b = np.array([[3.0, 2.0, 4.0, 5.0]], dtype=np.float32)
    _save(
        model_path,
        [helper.make_node("Mod", ["input", "b"], ["output"], fmod=1)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(b, "b")],
    )
    x = np.array([[7.0, 5.0, 9.0, 4.0]], dtype=np.float32)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


# ── TopK ──

def test_topk(tmp_path):
    model_path = tmp_path / "topk.onnx"
    spk_path = tmp_path / "topk.spk"
    k = np.array([3], dtype=np.int64)
    _save(
        model_path,
        [helper.make_node("TopK", ["input", "k"], ["values", "indices_out"], axis=-1),
         helper.make_node("Cast", ["values"], ["output"], to=TensorProto.FLOAT)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
        [numpy_helper.from_array(k, "k")],
    )
    x = np.array([[0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4]], dtype=np.float32)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


# ── Winograd (3×3 s1 p1 with large spatial) ──

def test_winograd_f43(tmp_path):
    model_path = tmp_path / "winograd_f43.onnx"
    spk_path = tmp_path / "winograd_f43.spk"
    w = _weight((16, 8, 3, 3), 0.01, -0.08)
    b = np.linspace(-0.02, 0.02, num=16, dtype=np.float32)
    _save(
        model_path,
        [helper.make_node("Conv", ["input", "w", "b"], ["conv_out"],
                          kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
         helper.make_node("Relu", ["conv_out"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 16, 16])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
    )
    x = np.linspace(-1, 1, num=8 * 16 * 16, dtype=np.float32).reshape(1, 8, 16, 16)
    actual = _compile_and_run(model_path, spk_path, x)
    expected = _run_ort(model_path, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
