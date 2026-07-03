"""Per-channel symmetric weight quantization for Conv."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from compiler.ir.graph import Graph
from compiler.ir import types
from compiler.quantization.calibrate import CalibrationProfile


@dataclass
class QuantParam:
    tensor_id: int
    scheme: str
    quant_axis: int
    num_channels: int
    scales: list[float]
    zero_points: list[int] = field(default_factory=list)


def quantize_conv_weights(graph: Graph) -> list[QuantParam]:
    """Quantize all Conv weight tensors from FP32 to INT8 in-place.

    Per-channel symmetric: scale[c] = max(|W[c,:]|) / 127
    W_q[c,:] = clamp(round(W[c,:] / scale[c]), -128, 127)

    Returns QuantParam for each quantized weight tensor.
    Does NOT touch non-Conv weights or activation tensors.
    """
    quant_params: list[QuantParam] = []
    quantized_tids: set[int] = set()

    for node in graph.nodes:
        if node.op_type != "Conv":
            continue
        if len(node.inputs) < 2:
            continue

        group = int(node.attrs.get("group", 1))
        if group != 1:
            continue

        w_tid = node.inputs[1]
        if w_tid in quantized_tids:
            continue

        w_tensor = graph.tensors[w_tid]
        if w_tensor.dtype != types.DTYPE_FP32:
            continue
        if w_tensor.data is None:
            continue

        w_fp32 = np.frombuffer(w_tensor.data, dtype=np.float32).copy().reshape(w_tensor.shape)
        out_channels = w_fp32.shape[0]

        w_flat = w_fp32.reshape(out_channels, -1)
        channel_max = np.max(np.abs(w_flat), axis=1)
        channel_max = np.maximum(channel_max, 1e-8)
        scales = channel_max / 127.0

        w_q = np.zeros_like(w_flat, dtype=np.int8)
        for c in range(out_channels):
            w_scaled = np.round(w_flat[c] / scales[c])
            w_q[c] = np.clip(w_scaled, -128, 127).astype(np.int8)

        w_tensor.data = w_q.reshape(w_tensor.shape).tobytes()
        w_tensor.dtype = types.DTYPE_INT8

        qp = QuantParam(
            tensor_id=w_tid,
            scheme="per_channel",
            quant_axis=0,
            num_channels=out_channels,
            scales=scales.tolist(),
            zero_points=[0] * out_channels,
        )
        quant_params.append(qp)
        quantized_tids.add(w_tid)

    return quant_params


def build_activation_quant_params(
    graph: Graph,
    calibration: CalibrationProfile,
) -> list[QuantParam]:
    """Build per-tensor symmetric quant params for Conv input activation tensors.

    Does NOT modify tensor data -- activations are quantized on-the-fly at runtime.
    """
    quant_params: list[QuantParam] = []
    seen_tids: set[int] = set()

    for node in graph.nodes:
        if node.op_type != "Conv":
            continue
        if not node.inputs:
            continue

        x_tid = node.inputs[0]
        if x_tid in seen_tids:
            continue
        seen_tids.add(x_tid)

        if x_tid in calibration.activation_scales:
            cal = calibration.activation_scales[x_tid]
            qp = QuantParam(
                tensor_id=x_tid,
                scheme="per_tensor",
                quant_axis=0,
                num_channels=1,
                scales=[cal.scale],
                zero_points=[cal.zero_point],
            )
            quant_params.append(qp)

    return quant_params
