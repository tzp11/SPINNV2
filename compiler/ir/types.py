"""Shared SIR constants."""

DTYPE_FP32 = "fp32"

ROLE_INPUT = "input"
ROLE_OUTPUT = "output"
ROLE_WEIGHT = "weight"
ROLE_ACTIVATION = "activation"
ROLE_CONSTANT = "constant"

LAYOUT_NCHW = "NCHW"

SUPPORTED_M1_OPS = {
    "Add",
    "Conv",
    "Flatten",
    "Gemm",
    "MaxPool",
    "Relu",
    "Softmax",
}

COMPILER_ONLY_OPS = {
    "BatchNormalization",
    "Dropout",
    "Identity",
}

SUPPORTED_IMPORT_OPS = SUPPORTED_M1_OPS | COMPILER_ONLY_OPS
