"""Generated fixed-shape ONNX models for integration and M6 validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TOY_MODELS = ("add", "gemm_softmax", "conv_relu", "conv_bn_relu")
SMALL_MODELS = ("mnist", "lenet")
MEDIUM_MODELS = ("resnet18", "mobilenetv2")
DETECTION_MODELS = ("yolo_tiny_prenms",)
ALL_MODELS = TOY_MODELS + SMALL_MODELS + MEDIUM_MODELS + DETECTION_MODELS


def write_model(name: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    builders = {
        "add": _write_add,
        "gemm_softmax": _write_gemm_softmax,
        "conv_relu": _write_conv_relu,
        "conv_bn_relu": _write_conv_bn_relu,
        "mnist": _write_mnist,
        "mnist_cnn": _write_mnist,
        "lenet": _write_lenet,
        "resnet18": _write_resnet18_like,
        "mobilenetv2": _write_mobilenetv2_like,
        "yolo_tiny_prenms": _write_yolo_tiny_prenms,
    }
    if name not in builders:
        raise ValueError(f"unknown generated model: {name}")
    builders[name](path)


def make_input(name: str) -> np.ndarray:
    shapes = {
        "add": (1, 4),
        "gemm_softmax": (1, 4),
        "conv_relu": (1, 1, 4, 4),
        "conv_bn_relu": (1, 1, 2, 2),
        "mnist": (1, 1, 8, 8),
        "mnist_cnn": (1, 1, 8, 8),
        "lenet": (1, 1, 16, 16),
        "resnet18": (1, 3, 16, 16),
        "mobilenetv2": (1, 3, 16, 16),
        "yolo_tiny_prenms": (1, 3, 16, 16),
    }
    if name not in shapes:
        raise ValueError(f"unknown generated model: {name}")
    shape = shapes[name]
    return np.linspace(-1.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)


def output_tolerance(name: str) -> tuple[float, float]:
    if name in {"gemm_softmax", "mnist", "mnist_cnn", "lenet"}:
        return 1e-3, 1e-5
    if name in {"resnet18", "mobilenetv2", "yolo_tiny_prenms"}:
        return 1e-3, 1e-5
    return 1e-4, 1e-5


def _save(path: Path, nodes, inputs, outputs, initializers) -> None:
    graph = helper.make_graph(nodes, path.stem, inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _weight(shape: tuple[int, ...], scale: float = 0.05, offset: float = 0.0) -> np.ndarray:
    return (np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) * scale + offset).astype(np.float32)


def _write_add(path: Path) -> None:
    bias = np.array([[0.25, -0.5, 0.75, -1.0]], dtype=np.float32)
    _save(
        path,
        [helper.make_node("Add", ["input", "bias"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(bias, "bias")],
    )


def _write_gemm_softmax(path: Path) -> None:
    w = _weight((4, 3), 0.07, -0.2)
    b = np.array([0.05, -0.03, 0.01], dtype=np.float32)
    _save(
        path,
        [
            helper.make_node("Gemm", ["input", "w", "b"], ["gemm"], alpha=1.0, beta=1.0),
            helper.make_node("Softmax", ["gemm"], ["output"], axis=1),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
    )


def _write_conv_relu(path: Path) -> None:
    w = (_weight((2, 1, 3, 3), 0.05, -0.4)).astype(np.float32)
    b = np.array([0.1, -0.2], dtype=np.float32)
    _save(
        path,
        [
            helper.make_node(
                "Conv",
                ["input", "w", "b"],
                ["conv"],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
                kernel_shape=[3, 3],
            ),
            helper.make_node("Relu", ["conv"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4, 4])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
    )


def _write_conv_bn_relu(path: Path) -> None:
    conv_w = np.array([[[[1.5]]], [[[-2.0]]]], dtype=np.float32)
    conv_b = np.array([0.25, -0.5], dtype=np.float32)
    scale = np.array([1.25, 0.75], dtype=np.float32)
    bn_bias = np.array([0.1, -0.2], dtype=np.float32)
    mean = np.array([0.5, -0.25], dtype=np.float32)
    var = np.array([0.75, 1.5], dtype=np.float32)
    _save(
        path,
        [
            helper.make_node("Conv", ["input", "conv_w", "conv_b"], ["conv"], kernel_shape=[1, 1]),
            helper.make_node(
                "BatchNormalization",
                ["conv", "scale", "bn_bias", "mean", "var"],
                ["bn"],
                epsilon=1e-5,
            ),
            helper.make_node("Relu", ["bn"], ["output"]),
        ],
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


def _write_mnist(path: Path) -> None:
    conv_w = _weight((2, 1, 3, 3), 0.03, -0.2)
    conv_b = np.array([0.05, -0.04], dtype=np.float32)
    gemm_w = _weight((32, 10), 0.01, -0.1)
    gemm_b = np.linspace(-0.05, 0.05, num=10, dtype=np.float32)
    _save(
        path,
        [
            helper.make_node("Conv", ["input", "conv_w", "conv_b"], ["conv"], pads=[1, 1, 1, 1], kernel_shape=[3, 3]),
            helper.make_node("Relu", ["conv"], ["relu"]),
            helper.make_node("MaxPool", ["relu"], ["pool"], kernel_shape=[2, 2], strides=[2, 2]),
            helper.make_node("Flatten", ["pool"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "gemm_w", "gemm_b"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["output"], axis=1),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
        [
            numpy_helper.from_array(conv_w, "conv_w"),
            numpy_helper.from_array(conv_b, "conv_b"),
            numpy_helper.from_array(gemm_w, "gemm_w"),
            numpy_helper.from_array(gemm_b, "gemm_b"),
        ],
    )


def _write_lenet(path: Path) -> None:
    conv1_w = _weight((4, 1, 3, 3), 0.02, -0.15)
    conv1_b = np.linspace(-0.02, 0.02, num=4, dtype=np.float32)
    conv2_w = _weight((8, 4, 3, 3), 0.01, -0.12)
    conv2_b = np.linspace(-0.03, 0.03, num=8, dtype=np.float32)
    gemm_w = _weight((128, 10), 0.005, -0.07)
    gemm_b = np.linspace(-0.05, 0.05, num=10, dtype=np.float32)
    _save(
        path,
        [
            helper.make_node("Conv", ["input", "conv1_w", "conv1_b"], ["conv1"], pads=[1, 1, 1, 1], kernel_shape=[3, 3]),
            helper.make_node("Relu", ["conv1"], ["relu1"]),
            helper.make_node("MaxPool", ["relu1"], ["pool1"], kernel_shape=[2, 2], strides=[2, 2]),
            helper.make_node("Conv", ["pool1", "conv2_w", "conv2_b"], ["conv2"], pads=[1, 1, 1, 1], kernel_shape=[3, 3]),
            helper.make_node("Relu", ["conv2"], ["relu2"]),
            helper.make_node("MaxPool", ["relu2"], ["pool2"], kernel_shape=[2, 2], strides=[2, 2]),
            helper.make_node("Flatten", ["pool2"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "gemm_w", "gemm_b"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["output"], axis=1),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
        [
            numpy_helper.from_array(conv1_w, "conv1_w"),
            numpy_helper.from_array(conv1_b, "conv1_b"),
            numpy_helper.from_array(conv2_w, "conv2_w"),
            numpy_helper.from_array(conv2_b, "conv2_b"),
            numpy_helper.from_array(gemm_w, "gemm_w"),
            numpy_helper.from_array(gemm_b, "gemm_b"),
        ],
    )


def _write_resnet18_like(path: Path) -> None:
    w1 = _weight((4, 3, 3, 3), 0.01, -0.12)
    b1 = np.linspace(-0.02, 0.02, num=4, dtype=np.float32)
    w2 = _weight((4, 4, 3, 3), 0.008, -0.08)
    b2 = np.linspace(-0.01, 0.01, num=4, dtype=np.float32)
    gemm_w = _weight((1024, 5), 0.002, -0.04)
    gemm_b = np.linspace(-0.02, 0.02, num=5, dtype=np.float32)
    _save(
        path,
        [
            helper.make_node("Conv", ["input", "w1", "b1"], ["conv1"], pads=[1, 1, 1, 1], kernel_shape=[3, 3]),
            helper.make_node("Relu", ["conv1"], ["relu1"]),
            helper.make_node("Conv", ["relu1", "w2", "b2"], ["conv2"], pads=[1, 1, 1, 1], kernel_shape=[3, 3]),
            helper.make_node("Add", ["conv2", "relu1"], ["residual"]),
            helper.make_node("Relu", ["residual"], ["relu2"]),
            helper.make_node("Flatten", ["relu2"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "gemm_w", "gemm_b"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5])],
        [
            numpy_helper.from_array(w1, "w1"),
            numpy_helper.from_array(b1, "b1"),
            numpy_helper.from_array(w2, "w2"),
            numpy_helper.from_array(b2, "b2"),
            numpy_helper.from_array(gemm_w, "gemm_w"),
            numpy_helper.from_array(gemm_b, "gemm_b"),
        ],
    )


def _write_mobilenetv2_like(path: Path) -> None:
    pw1 = _weight((6, 3, 1, 1), 0.02, -0.08)
    pw1_b = np.linspace(-0.02, 0.02, num=6, dtype=np.float32)
    dw = _weight((6, 1, 3, 3), 0.015, -0.06)
    pw2 = _weight((4, 6, 1, 1), 0.02, -0.04)
    pw2_b = np.linspace(-0.01, 0.01, num=4, dtype=np.float32)
    gemm_w = _weight((1024, 5), 0.002, -0.04)
    gemm_b = np.linspace(-0.02, 0.02, num=5, dtype=np.float32)
    _save(
        path,
        [
            helper.make_node("Conv", ["input", "pw1", "pw1_b"], ["expand"], kernel_shape=[1, 1]),
            helper.make_node("Relu", ["expand"], ["expand_relu"]),
            helper.make_node("Conv", ["expand_relu", "dw"], ["dw_out"], pads=[1, 1, 1, 1], group=6, kernel_shape=[3, 3]),
            helper.make_node("Relu", ["dw_out"], ["dw_relu"]),
            helper.make_node("Conv", ["dw_relu", "pw2", "pw2_b"], ["project"], kernel_shape=[1, 1]),
            helper.make_node("Relu", ["project"], ["project_relu"]),
            helper.make_node("Flatten", ["project_relu"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "gemm_w", "gemm_b"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5])],
        [
            numpy_helper.from_array(pw1, "pw1"),
            numpy_helper.from_array(pw1_b, "pw1_b"),
            numpy_helper.from_array(dw, "dw"),
            numpy_helper.from_array(pw2, "pw2"),
            numpy_helper.from_array(pw2_b, "pw2_b"),
            numpy_helper.from_array(gemm_w, "gemm_w"),
            numpy_helper.from_array(gemm_b, "gemm_b"),
        ],
    )


def _write_yolo_tiny_prenms(path: Path) -> None:
    w = _weight((4, 3, 3, 3), 0.01, -0.1)
    b = np.linspace(-0.02, 0.02, num=4, dtype=np.float32)
    _save(
        path,
        [
            helper.make_node("Conv", ["input", "w", "b"], ["conv"], pads=[1, 1, 1, 1], kernel_shape=[3, 3]),
            helper.make_node("Sigmoid", ["conv"], ["score"]),
            helper.make_node("Mul", ["conv", "score"], ["act"]),
            helper.make_node("Concat", ["act", "score"], ["concat"], axis=1),
            helper.make_node("Transpose", ["concat"], ["output"], perm=[0, 2, 3, 1]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 16, 8])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
    )
