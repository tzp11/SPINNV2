# SIR Specification

SIR, Satellite Inference IR, is the compiler-side intermediate representation used by SPINNV2.

## Scope

SIR is not a complete ONNX clone. It is a static-shape deployment IR for satellite-oriented inference workloads.

## Graph

A SIR graph contains:

| Field | Meaning |
|---|---|
| `model_name` | Human-readable model name |
| `opset_source` | Source model opset metadata |
| `inputs` | Graph input tensor IDs |
| `outputs` | Graph output tensor IDs |
| `tensors` | Tensor table |
| `nodes` | Topologically sorted node table |
| `metadata` | Compiler and target metadata |

## Tensor

Each tensor contains:

| Field | Meaning |
|---|---|
| `id` | Stable tensor ID |
| `name` | Debug name |
| `dtype` | Storage dtype, initially `fp32` |
| `shape` | Static shape |
| `layout` | `NCHW`, `NHWC`, packed layouts, or target-specific layouts |
| `role` | `input`, `output`, `weight`, `activation`, or `constant` |
| `size_bytes` | Static byte size |
| `producer` | Producer node ID, or null |
| `consumers` | Consumer node IDs |
| `quant` | Optional quantization descriptor |
| `memory` | Filled by the memory planner |

## Node

Each node contains:

| Field | Meaning |
|---|---|
| `id` | Stable node ID |
| `op_type` | SIR op type |
| `inputs` | Input tensor IDs |
| `outputs` | Output tensor IDs |
| `attrs` | Structured SPINNV2 attributes |
| `execution_index` | Topological execution order |
| `backend_hint` | Filled by backend binding |
| `kernel_spec` | Filled by KernelSpec selection |
| `flags` | Fused, skipped, debug, or other deployment flags |

## First Supported Op Set

M1 requires:

```text
Conv
Relu
MaxPool
Flatten
Gemm
Softmax
```

Optional M1 additions:

```text
Add
Mul
Reshape
Transpose
```

## M6 Large-Model Op Set

M6 extends the fixed-shape fp32 op set for ResNet101 and YOLOv10n:

```text
Add
Cast
Concat
Conv
Div
Flatten
GatherElements
Gemm
MatMul
MaxPool
Mod
Mul
ReduceMax
ReduceMean
Relu
Reshape
Resize
Sigmoid
Softmax
Split
Sub
Tile
TopK
Transpose
Unsqueeze
```

SPINNV2 still treats this as a deliberately bounded deployment op set, not full
ONNX coverage.

## Rejection Rule

Unsupported ONNX operators must fail at compile time. Runtime must never interpret unknown ONNX operators.
