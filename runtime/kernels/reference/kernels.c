#include "context.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

typedef int (*NodeKernelFn)(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch);

typedef struct {
    uint16_t op_type;
    uint16_t backend;
    uint16_t kernel_kind;
    NodeKernelFn fn;
} KernelRegistryEntry;

static size_t elem_count(const Spkv2TensorRecord *record) {
    size_t count = 1;
    for (uint16_t i = 0; i < record->rank; i++) {
        count *= record->shape[i];
    }
    return count;
}

static int get_attr(const Spkv2Context *ctx, const Spkv2NodeRecord *node, Spkv2AttrRecord *attr) {
    if (node->attr_offset > ctx->attrs_size ||
        node->attr_size > ctx->attrs_size - node->attr_offset ||
        node->attr_size < sizeof(Spkv2AttrRecord)) {
        return -10;
    }
    memcpy(attr, ctx->attrs + node->attr_offset, sizeof(Spkv2AttrRecord));
    return 0;
}

static int kernel_add(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = elem_count(y_rec);
    size_t bn = elem_count(b_rec);
    size_t an = elem_count(a_rec);
    for (size_t i = 0; i < n; i++) {
        y[i] = a[(an == 1) ? 0 : i] + b[(bn == 1) ? 0 : i];
    }
    return 0;
}

static int kernel_relu(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = elem_count(x_rec);
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
    return 0;
}

static int kernel_flatten(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    memcpy(ctx->tensors[node->outputs[0]].data, ctx->tensors[node->inputs[0]].data, x_rec->size_bytes);
    return 0;
}

static int kernel_gemm(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = get_attr(ctx, node, &attr);
    if (rc != 0) return rc;

    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *c = node->input_count > 2 ? (const float *)ctx->tensors[node->inputs[2]].data : NULL;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int a_rows = attr.trans_a ? (int)a_rec->shape[1] : (int)a_rec->shape[0];
    int a_cols = attr.trans_a ? (int)a_rec->shape[0] : (int)a_rec->shape[1];
    int b_cols = attr.trans_b ? (int)b_rec->shape[0] : (int)b_rec->shape[1];

    for (int m = 0; m < a_rows; m++) {
        for (int n = 0; n < b_cols; n++) {
            float sum = 0.0f;
            for (int k = 0; k < a_cols; k++) {
                float av = attr.trans_a ? a[k * a_rows + m] : a[m * a_cols + k];
                float bv = attr.trans_b ? b[n * a_cols + k] : b[k * b_cols + n];
                sum += av * bv;
            }
            float bias = c ? c[n] : 0.0f;
            y[m * b_cols + n] = attr.alpha * sum + bias;
        }
    }
    return 0;
}

static int kernel_gemm_cpu_direct(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    if (attr.trans_a || attr.trans_b) {
        return kernel_gemm(ctx, node, scratch);
    }

    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *c = node->input_count > 2 ? (const float *)ctx->tensors[node->inputs[2]].data : NULL;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int rows = (int)a_rec->shape[0];
    int inner = (int)a_rec->shape[1];
    int cols = (int)b_rec->shape[1];
    for (int m = 0; m < rows; m++) {
        const float *a_row = a + (size_t)m * inner;
        float *y_row = y + (size_t)m * cols;
        for (int n = 0; n < cols; n++) {
            y_row[n] = c ? c[n] : 0.0f;
        }
        for (int k = 0; k < inner; k++) {
            float av = attr.alpha * a_row[k];
            const float *b_row = b + (size_t)k * cols;
            for (int n = 0; n < cols; n++) {
                y_row[n] += av * b_row[n];
            }
        }
    }
    return 0;
}

static int kernel_softmax(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int axis = attr.axis < 0 ? (int)x_rec->rank + attr.axis : attr.axis;
    if (axis < 0 || axis >= (int)x_rec->rank) return -11;

    size_t outer = 1, inner = 1;
    size_t dim = x_rec->shape[axis];
    for (int i = 0; i < axis; i++) outer *= x_rec->shape[i];
    for (uint16_t i = (uint16_t)axis + 1; i < x_rec->rank; i++) inner *= x_rec->shape[i];

    for (size_t o = 0; o < outer; o++) {
        for (size_t in = 0; in < inner; in++) {
            float maxv = -INFINITY;
            for (size_t d = 0; d < dim; d++) {
                float v = x[(o * dim + d) * inner + in];
                if (v > maxv) maxv = v;
            }
            float sum = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                float e = expf(x[(o * dim + d) * inner + in] - maxv);
                y[(o * dim + d) * inner + in] = e;
                sum += e;
            }
            for (size_t d = 0; d < dim; d++) {
                y[(o * dim + d) * inner + in] /= sum;
            }
        }
    }
    return 0;
}

static int kernel_conv(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    if (attr.group != 1) return -12;

    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *w_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *w = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *bias = node->input_count > 2 ? (const float *)ctx->tensors[node->inputs[2]].data : NULL;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int N = (int)x_rec->shape[0];
    int C = (int)x_rec->shape[1];
    int H = (int)x_rec->shape[2];
    int W = (int)x_rec->shape[3];
    int M = (int)w_rec->shape[0];
    int kH = (int)w_rec->shape[2];
    int kW = (int)w_rec->shape[3];
    int outH = (int)y_rec->shape[2];
    int outW = (int)y_rec->shape[3];

    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float sum = bias ? bias[m] : 0.0f;
                    for (int c = 0; c < C; c++) {
                        for (int kh = 0; kh < kH; kh++) {
                            int ih = oh * attr.strides[0] + kh * attr.dilations[0] - attr.pads[0];
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < kW; kw++) {
                                int iw = ow * attr.strides[1] + kw * attr.dilations[1] - attr.pads[1];
                                if (iw < 0 || iw >= W) continue;
                                size_t xi = ((size_t)n * C * H * W) + ((size_t)c * H * W) + ((size_t)ih * W) + iw;
                                size_t wi = ((size_t)m * C * kH * kW) + ((size_t)c * kH * kW) + ((size_t)kh * kW) + kw;
                                sum += x[xi] * w[wi];
                            }
                        }
                    }
                    if (attr.fused_activation == 1 && sum < 0.0f) {
                        sum = 0.0f;
                    }
                    y[((size_t)n * M * outH * outW) + ((size_t)m * outH * outW) + ((size_t)oh * outW) + ow] = sum;
                }
            }
        }
    }
    return 0;
}

static int kernel_conv_im2col(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    Spkv2AttrRecord attr;
    int rc = get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    if (attr.group != 1) return -12;
    if (!scratch && node->scratch_bytes > 0) return -13;

    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *w_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *w = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *bias = node->input_count > 2 ? (const float *)ctx->tensors[node->inputs[2]].data : NULL;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    float *patch = (float *)scratch;

    int N = (int)x_rec->shape[0];
    int C = (int)x_rec->shape[1];
    int H = (int)x_rec->shape[2];
    int W = (int)x_rec->shape[3];
    int M = (int)w_rec->shape[0];
    int kH = (int)w_rec->shape[2];
    int kW = (int)w_rec->shape[3];
    int outH = (int)y_rec->shape[2];
    int outW = (int)y_rec->shape[3];
    int patch_len = C * kH * kW;

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int pi = 0;
                for (int c = 0; c < C; c++) {
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = oh * attr.strides[0] + kh * attr.dilations[0] - attr.pads[0];
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = ow * attr.strides[1] + kw * attr.dilations[1] - attr.pads[1];
                            float value = 0.0f;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                value = x[((size_t)n * C * H * W) + ((size_t)c * H * W) + ((size_t)ih * W) + iw];
                            }
                            patch[pi++] = value;
                        }
                    }
                }
                for (int m = 0; m < M; m++) {
                    const float *w_row = w + (size_t)m * patch_len;
                    float sum = bias ? bias[m] : 0.0f;
                    for (int k = 0; k < patch_len; k++) {
                        sum += patch[k] * w_row[k];
                    }
                    if (attr.fused_activation == 1 && sum < 0.0f) {
                        sum = 0.0f;
                    }
                    y[((size_t)n * M * outH * outW) + ((size_t)m * outH * outW) + ((size_t)oh * outW) + ow] = sum;
                }
            }
        }
    }
    return 0;
}

static int kernel_maxpool(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int N = (int)x_rec->shape[0];
    int C = (int)x_rec->shape[1];
    int H = (int)x_rec->shape[2];
    int W = (int)x_rec->shape[3];
    int outH = (int)y_rec->shape[2];
    int outW = (int)y_rec->shape[3];

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float maxv = -INFINITY;
                    for (int kh = 0; kh < attr.kernel_shape[0]; kh++) {
                        int ih = oh * attr.strides[0] + kh - attr.pads[0];
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < attr.kernel_shape[1]; kw++) {
                            int iw = ow * attr.strides[1] + kw - attr.pads[1];
                            if (iw < 0 || iw >= W) continue;
                            float v = x[((size_t)n * C * H * W) + ((size_t)c * H * W) + ((size_t)ih * W) + iw];
                            if (v > maxv) maxv = v;
                        }
                    }
                    y[((size_t)n * C * outH * outW) + ((size_t)c * outH * outW) + ((size_t)oh * outW) + ow] = maxv;
                }
            }
        }
    }
    return 0;
}

static const KernelRegistryEntry REGISTRY[] = {
    {SPKV2_OP_ADD, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_add},
    {SPKV2_OP_CONV, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_conv},
    {SPKV2_OP_CONV, SPKV2_BACKEND_CPU, SPKV2_KERNEL_IM2COL_GEMM, kernel_conv_im2col},
    {SPKV2_OP_FLATTEN, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_flatten},
    {SPKV2_OP_GEMM, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_gemm},
    {SPKV2_OP_GEMM, SPKV2_BACKEND_CPU, SPKV2_KERNEL_DIRECT, kernel_gemm_cpu_direct},
    {SPKV2_OP_MAXPOOL, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_maxpool},
    {SPKV2_OP_RELU, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_relu},
    {SPKV2_OP_SOFTMAX, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, kernel_softmax},
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
    void *scratch = spec->scratch_bytes > 0 ? ctx->owned_scratch + spec->scratch_offset : NULL;
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
    case SPKV2_OP_CONV:
    case SPKV2_OP_FLATTEN:
    case SPKV2_OP_GEMM:
    case SPKV2_OP_MAXPOOL:
    case SPKV2_OP_RELU:
    case SPKV2_OP_SOFTMAX:
        return execute_with_spec(ctx, node, &ref_spec);
    default:
        return -99;
    }
}
