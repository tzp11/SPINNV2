#include "reference_kernels.h"

#ifdef __AVX2__
extern int kernel_conv_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
extern int kernel_gemm_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
extern int kernel_matmul_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
extern int kernel_add_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
extern int kernel_mul_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
extern int kernel_sub_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
extern int kernel_div_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
extern int kernel_relu_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
extern int kernel_sigmoid_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
extern int kernel_transpose_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
extern int kernel_reduce_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
extern int kernel_softmax_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);
#endif

typedef struct {
    uint16_t op_type;
    uint16_t backend;
    uint16_t kernel_kind;
    NodeKernelFn fn;
} KernelRegistryEntry;

static const KernelRegistryEntry REGISTRY[] = {
    /* ── SIMD optimised entries (higher priority, matched first) ── */
#ifdef __AVX2__
    {SPKV2_OP_CONV,    SPKV2_BACKEND_SIMD, SPKV2_KERNEL_IM2COL_GEMM, kernel_conv_simd},
    {SPKV2_OP_CONV,    SPKV2_BACKEND_SIMD, SPKV2_KERNEL_POINTWISE_1X1, kernel_conv_simd},
    {SPKV2_OP_CONV,    SPKV2_BACKEND_SIMD, SPKV2_KERNEL_DEPTHWISE_DIRECT, kernel_conv_simd},
    {SPKV2_OP_CONV,    SPKV2_BACKEND_SIMD, SPKV2_KERNEL_WINOGRAD_3X3S1, kernel_conv_simd},
    {SPKV2_OP_CONV,    SPKV2_BACKEND_SIMD, SPKV2_KERNEL_CONV3X3S2_DIRECT, kernel_conv_simd},
    {SPKV2_OP_GEMM,    SPKV2_BACKEND_SIMD, SPKV2_KERNEL_DIRECT,      kernel_gemm_simd},
    {SPKV2_OP_MATMUL,  SPKV2_BACKEND_SIMD, SPKV2_KERNEL_DIRECT,      kernel_matmul_simd},
    {SPKV2_OP_ADD,     SPKV2_BACKEND_SIMD, SPKV2_KERNEL_REFERENCE,   kernel_add_simd},
    {SPKV2_OP_MUL,     SPKV2_BACKEND_SIMD, SPKV2_KERNEL_REFERENCE,   kernel_mul_simd},
    {SPKV2_OP_SUB,     SPKV2_BACKEND_SIMD, SPKV2_KERNEL_REFERENCE,   kernel_sub_simd},
    {SPKV2_OP_DIV,     SPKV2_BACKEND_SIMD, SPKV2_KERNEL_REFERENCE,   kernel_div_simd},
    {SPKV2_OP_RELU,    SPKV2_BACKEND_SIMD, SPKV2_KERNEL_REFERENCE,   kernel_relu_simd},
    {SPKV2_OP_SIGMOID, SPKV2_BACKEND_SIMD, SPKV2_KERNEL_REFERENCE,   kernel_sigmoid_simd},
    {SPKV2_OP_TRANSPOSE, SPKV2_BACKEND_SIMD, SPKV2_KERNEL_REFERENCE, kernel_transpose_simd},
    {SPKV2_OP_REDUCEMAX, SPKV2_BACKEND_SIMD, SPKV2_KERNEL_REFERENCE, kernel_reduce_simd},
    {SPKV2_OP_REDUCEMEAN, SPKV2_BACKEND_SIMD, SPKV2_KERNEL_REFERENCE, kernel_reduce_simd},
    {SPKV2_OP_SOFTMAX, SPKV2_BACKEND_SIMD, SPKV2_KERNEL_REFERENCE, kernel_softmax_simd},
#endif
    /* ── Reference entries ── */
    {SPKV2_OP_ADD, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_add},
    {SPKV2_OP_CAST, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_copy},
    {SPKV2_OP_CONCAT, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_concat},
    {SPKV2_OP_CONV, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_conv},
    {SPKV2_OP_CONV, SPKV2_BACKEND_CPU, SPKV2_KERNEL_IM2COL_GEMM, kernel_conv_im2col},
    {SPKV2_OP_DIV, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_binary},
    {SPKV2_OP_FLATTEN, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_flatten},
    {SPKV2_OP_GATHERELEMENTS, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_gather_elements},
    {SPKV2_OP_GEMM, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_gemm},
    {SPKV2_OP_GEMM, SPKV2_BACKEND_CPU, SPKV2_KERNEL_DIRECT, kernel_gemm_cpu_direct},
    {SPKV2_OP_MATMUL, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_matmul},
    {SPKV2_OP_MAXPOOL, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_maxpool},
    {SPKV2_OP_MOD, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_binary},
    {SPKV2_OP_MUL, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_binary},
    {SPKV2_OP_REDUCEMAX, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_reduce},
    {SPKV2_OP_REDUCEMEAN, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_reduce},
    {SPKV2_OP_RELU, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_relu},
    {SPKV2_OP_RESHAPE, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_copy},
    {SPKV2_OP_RESIZE, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_resize},
    {SPKV2_OP_SIGMOID, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_sigmoid},
    {SPKV2_OP_SOFTMAX, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_softmax},
    {SPKV2_OP_SPLIT, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_split},
    {SPKV2_OP_SUB, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_binary},
    {SPKV2_OP_TILE, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_tile},
    {SPKV2_OP_TOPK, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_topk},
    {SPKV2_OP_TRANSPOSE, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_transpose},
    {SPKV2_OP_UNSQUEEZE, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_unsqueeze},
};

static const Spkv2KernelSpecRecord *kernel_spec_by_id(const Spkv2Context *ctx, uint32_t id) {
    if (id == 0xFFFFFFFFu || id >= ctx->kernel_spec_count) return NULL;
    return &ctx->kernel_spec_records[id];
}

static NodeKernelFn find_kernel(uint16_t op_type, const Spkv2KernelSpecRecord *spec) {
    size_t count = sizeof(REGISTRY) / sizeof(REGISTRY[0]);
    for (size_t i = 0; i < count; i++) {
        if (REGISTRY[i].op_type == op_type &&
            REGISTRY[i].backend == spec->backend &&
            REGISTRY[i].kernel_kind == spec->kernel_kind) {
            return REGISTRY[i].fn;
        }
    }
    return NULL;
}

static const Spkv2KernelSpecRecord fallback_ref_spec = {
    0xFFFFFFFFu,
    0xFFFFFFFFu,
    SPKV2_KERNEL_REFERENCE,
    SPKV2_BACKEND_REF,
    SPKV2_DTYPE_FP32,
    1,
    1,
    0,
    0,
    0,
    0xFFFFFFFFu,
    1,
};

static const Spkv2KernelSpecRecord *selected_spec(const Spkv2Context *ctx, const Spkv2NodeRecord *node) {
    const Spkv2KernelSpecRecord *spec = kernel_spec_by_id(ctx, node->kernel_spec_id);
    if (!spec || spec->node_id != node->id) {
        return &fallback_ref_spec;
    }
    return spec;
}

static int execute_with_spec(Spkv2Context *ctx, const Spkv2NodeRecord *node, const Spkv2KernelSpecRecord *spec) {
    NodeKernelFn fn = find_kernel(node->op_type, spec);
    if (!fn) return -99;
    if (spec->scratch_bytes > ctx->scratch_size) return -14;
    void *scratch = spec->scratch_bytes > 0 ? ctx->scratch + spec->scratch_offset : NULL;
    if (spec->scratch_bytes > 0 && spec->scratch_offset > ctx->scratch_size - spec->scratch_bytes) return -14;
    return fn(ctx, node, scratch);
}

int spkv2_execute_node(Spkv2Context *ctx, const Spkv2NodeRecord *node) {
    const Spkv2KernelSpecRecord *spec = selected_spec(ctx, node);
    int rc = execute_with_spec(ctx, node, spec);
    if (rc != -99) return rc;
    const Spkv2KernelSpecRecord *fallback = kernel_spec_by_id(ctx, spec->fallback_kernel_spec_id);
    if (fallback) return execute_with_spec(ctx, node, fallback);

    Spkv2KernelSpecRecord ref_spec = fallback_ref_spec;
    ref_spec.node_id = node->id;
    switch (node->op_type) {
    case SPKV2_OP_ADD:
    case SPKV2_OP_CAST:
    case SPKV2_OP_CONCAT:
    case SPKV2_OP_CONV:
    case SPKV2_OP_DIV:
    case SPKV2_OP_FLATTEN:
    case SPKV2_OP_GATHERELEMENTS:
    case SPKV2_OP_GEMM:
    case SPKV2_OP_MATMUL:
    case SPKV2_OP_MAXPOOL:
    case SPKV2_OP_MOD:
    case SPKV2_OP_MUL:
    case SPKV2_OP_REDUCEMAX:
    case SPKV2_OP_REDUCEMEAN:
    case SPKV2_OP_RELU:
    case SPKV2_OP_RESHAPE:
    case SPKV2_OP_RESIZE:
    case SPKV2_OP_SIGMOID:
    case SPKV2_OP_SOFTMAX:
    case SPKV2_OP_SPLIT:
    case SPKV2_OP_SUB:
    case SPKV2_OP_TILE:
    case SPKV2_OP_TOPK:
    case SPKV2_OP_TRANSPOSE:
    case SPKV2_OP_UNSQUEEZE:
        return execute_with_spec(ctx, node, &ref_spec);
    default:
        return -99;
    }
}
