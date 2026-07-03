"""Shared SIR constants."""

DTYPE_FP32 = "fp32"
DTYPE_INT8 = "int8"
DTYPE_FP16 = "fp16"

ROLE_INPUT = "input"
ROLE_OUTPUT = "output"
ROLE_WEIGHT = "weight"
ROLE_ACTIVATION = "activation"
ROLE_CONSTANT = "constant"

LAYOUT_NCHW = "NCHW"

SUPPORTED_M1_OPS = {
    "Add",
    "AveragePool",
    "Cast",
    "Clip",
    "Concat",
    "Conv",
    "Div",
    "Flatten",
    "Gather",
    "GatherElements",
    "Gemm",
    "GlobalAveragePool",
    "LeakyRelu",
    "MatMul",
    "MaxPool",
    "Mod",
    "Mul",
    "ReduceMax",
    "ReduceMean",
    "Relu",
    "Reshape",
    "Resize",
    "Sigmoid",
    "Slice",
    "Softmax",
    "Split",
    "Sub",
    "Tile",
    "TopK",
    "Transpose",
    "Unsqueeze",
}

COMPILER_ONLY_OPS = {
    "BatchNormalization",
    "Dropout",
    "Identity",
}

SUPPORTED_IMPORT_OPS = SUPPORTED_M1_OPS | COMPILER_ONLY_OPS
