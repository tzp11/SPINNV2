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

static int normalize_axis(int axis, uint16_t rank) {
    if (axis < 0) axis += (int)rank;
    return axis;
}

static size_t linear_to_coords(size_t linear, const Spkv2TensorRecord *record, uint32_t coords[8]) {
    (void)linear;
    for (int i = (int)record->rank - 1; i >= 0; i--) {
        uint32_t dim = record->shape[i];
        coords[i] = dim == 0 ? 0 : (uint32_t)(linear % dim);
        linear = dim == 0 ? 0 : linear / dim;
    }
    return linear;
}

static size_t coords_to_linear(const Spkv2TensorRecord *record, const uint32_t coords[8]) {
    size_t index = 0;
    for (uint16_t i = 0; i < record->rank; i++) {
        index = index * record->shape[i] + coords[i];
    }
    return index;
}

static size_t broadcast_index(const Spkv2TensorRecord *in_rec, const Spkv2TensorRecord *out_rec, size_t out_index) {
    uint32_t out_coords[8] = {0};
    uint32_t in_coords[8] = {0};
    linear_to_coords(out_index, out_rec, out_coords);
    int rank_delta = (int)out_rec->rank - (int)in_rec->rank;
    for (uint16_t i = 0; i < in_rec->rank; i++) {
        uint32_t coord = out_coords[i + rank_delta];
        in_coords[i] = in_rec->shape[i] == 1 ? 0 : coord;
    }
    return coords_to_linear(in_rec, in_coords);
}

static int tensor_axis_values(
    const Spkv2Context *ctx,
    const Spkv2NodeRecord *node,
    const Spkv2AttrRecord *attr,
    uint16_t rank,
    int axes[8],
    int *axis_count) {
    *axis_count = 0;
    if (node->input_count > 1) {
        const Spkv2TensorRecord *axes_rec = ctx->tensors[node->inputs[1]].record;
        const float *axes_data = (const float *)ctx->tensors[node->inputs[1]].data;
        size_t count = elem_count(axes_rec);
        if (count > 8) return -11;
        for (size_t i = 0; i < count; i++) {
            axes[*axis_count] = normalize_axis((int)axes_data[i], rank);
            (*axis_count)++;
        }
        return 0;
    }
    if (attr->extra_count > 0) {
        if (attr->extra_count > 8) return -11;
        for (int i = 0; i < attr->extra_count; i++) {
            axes[*axis_count] = normalize_axis(attr->extra[i], rank);
            (*axis_count)++;
        }
        return 0;
    }
    for (uint16_t i = 0; i < rank; i++) {
        axes[*axis_count] = (int)i;
        (*axis_count)++;
    }
    return 0;
}

static int axis_in_set(int axis, const int axes[8], int axis_count) {
    for (int i = 0; i < axis_count; i++) {
        if (axes[i] == axis) return 1;
    }
    return 0;
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
    for (size_t i = 0; i < n; i++) {
        y[i] = a[broadcast_index(a_rec, y_rec, i)] + b[broadcast_index(b_rec, y_rec, i)];
    }
    return 0;
}

static int kernel_binary(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = elem_count(y_rec);
    for (size_t i = 0; i < n; i++) {
        float av = a[broadcast_index(a_rec, y_rec, i)];
        float bv = b[broadcast_index(b_rec, y_rec, i)];
        switch (node->op_type) {
        case SPKV2_OP_MUL:
            y[i] = av * bv;
            break;
        case SPKV2_OP_SUB:
            y[i] = av - bv;
            break;
        case SPKV2_OP_DIV:
            y[i] = av / bv;
            break;
        case SPKV2_OP_MOD:
            y[i] = (float)((int)av % (int)bv);
            break;
        default:
            return -99;
        }
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

static int kernel_sigmoid(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = elem_count(x_rec);
    for (size_t i = 0; i < n; i++) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
    return 0;
}

static int kernel_flatten(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    memcpy(ctx->tensors[node->outputs[0]].data, ctx->tensors[node->inputs[0]].data, x_rec->size_bytes);
    return 0;
}

static int kernel_copy(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    memcpy(ctx->tensors[node->outputs[0]].data, ctx->tensors[node->inputs[0]].data, y_rec->size_bytes);
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
    int group = attr.group > 0 ? attr.group : 1;
    if (C % group != 0 || M % group != 0) return -12;
    int C_per_group = C / group;
    int M_per_group = M / group;
    int kH = (int)w_rec->shape[2];
    int kW = (int)w_rec->shape[3];
    int outH = (int)y_rec->shape[2];
    int outW = (int)y_rec->shape[3];

    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            int g = m / M_per_group;
            int c_begin = g * C_per_group;
            int c_end = c_begin + C_per_group;
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float sum = bias ? bias[m] : 0.0f;
                    for (int c = c_begin; c < c_end; c++) {
                        int wc = c - c_begin;
                        for (int kh = 0; kh < kH; kh++) {
                            int ih = oh * attr.strides[0] + kh * attr.dilations[0] - attr.pads[0];
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < kW; kw++) {
                                int iw = ow * attr.strides[1] + kw * attr.dilations[1] - attr.pads[1];
                                if (iw < 0 || iw >= W) continue;
                                size_t xi = ((size_t)n * C * H * W) + ((size_t)c * H * W) + ((size_t)ih * W) + iw;
                                size_t wi = ((size_t)m * C_per_group * kH * kW) + ((size_t)wc * kH * kW) + ((size_t)kh * kW) + kw;
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
    if (attr.group != 1) return -99;
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

static int kernel_transpose(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = elem_count(y_rec);
    for (size_t i = 0; i < n; i++) {
        uint32_t y_coords[8] = {0};
        uint32_t x_coords[8] = {0};
        linear_to_coords(i, y_rec, y_coords);
        for (uint16_t d = 0; d < x_rec->rank; d++) {
            int src_axis = attr.extra_count > 0 ? attr.extra[d] : (int)d;
            x_coords[src_axis] = y_coords[d];
        }
        y[i] = x[coords_to_linear(x_rec, x_coords)];
    }
    return 0;
}

static int kernel_concat(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    int axis = normalize_axis(attr.axis, y_rec->rank);
    if (axis < 0 || axis >= (int)y_rec->rank) return -11;
    size_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= y_rec->shape[i];
    for (uint16_t i = (uint16_t)axis + 1; i < y_rec->rank; i++) inner *= y_rec->shape[i];
    size_t y_axis = y_rec->shape[axis];
    for (size_t o = 0; o < outer; o++) {
        size_t axis_offset = 0;
        for (uint16_t in_id = 0; in_id < node->input_count; in_id++) {
            const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[in_id]].record;
            const float *x = (const float *)ctx->tensors[node->inputs[in_id]].data;
            size_t x_axis = x_rec->shape[axis];
            size_t x_block = x_axis * inner;
            memcpy(y + (o * y_axis + axis_offset) * inner, x + o * x_block, x_block * sizeof(float));
            axis_offset += x_axis;
        }
    }
    return 0;
}

static int kernel_split(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    int axis = normalize_axis(attr.axis, x_rec->rank);
    if (axis < 0 || axis >= (int)x_rec->rank) return -11;
    size_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= x_rec->shape[i];
    for (uint16_t i = (uint16_t)axis + 1; i < x_rec->rank; i++) inner *= x_rec->shape[i];
    size_t x_axis = x_rec->shape[axis];
    for (size_t o = 0; o < outer; o++) {
        size_t axis_offset = 0;
        for (uint16_t out_id = 0; out_id < node->output_count; out_id++) {
            const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[out_id]].record;
            float *y = (float *)ctx->tensors[node->outputs[out_id]].data;
            size_t y_axis = y_rec->shape[axis];
            size_t y_block = y_axis * inner;
            memcpy(y + o * y_block, x + (o * x_axis + axis_offset) * inner, y_block * sizeof(float));
            axis_offset += y_axis;
        }
    }
    return 0;
}

static int kernel_reduce(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    int axes[8] = {0};
    int axis_count = 0;
    rc = tensor_axis_values(ctx, node, &attr, x_rec->rank, axes, &axis_count);
    if (rc != 0) return rc;
    size_t x_count = elem_count(x_rec);
    size_t y_count = elem_count(y_rec);
    for (size_t yi = 0; yi < y_count; yi++) {
        y[yi] = node->op_type == SPKV2_OP_REDUCEMAX ? -INFINITY : 0.0f;
    }
    float denom = 1.0f;
    for (int i = 0; i < axis_count; i++) denom *= (float)x_rec->shape[axes[i]];
    for (size_t xi = 0; xi < x_count; xi++) {
        uint32_t x_coords[8] = {0};
        uint32_t y_coords[8] = {0};
        linear_to_coords(xi, x_rec, x_coords);
        uint16_t yd = 0;
        for (uint16_t xd = 0; xd < x_rec->rank; xd++) {
            if (axis_in_set((int)xd, axes, axis_count)) {
                if (attr.keepdims && yd < y_rec->rank) y_coords[yd++] = 0;
            } else if (yd < y_rec->rank) {
                y_coords[yd++] = x_coords[xd];
            }
        }
        size_t yi = coords_to_linear(y_rec, y_coords);
        if (node->op_type == SPKV2_OP_REDUCEMAX) {
            if (x[xi] > y[yi]) y[yi] = x[xi];
        } else {
            y[yi] += x[xi] / denom;
        }
    }
    return 0;
}

static int kernel_matmul(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    if (a_rec->rank < 2 || b_rec->rank < 2 || y_rec->rank < 2) return -11;
    size_t M = y_rec->shape[y_rec->rank - 2];
    size_t N = y_rec->shape[y_rec->rank - 1];
    size_t K = a_rec->shape[a_rec->rank - 1];
    size_t batch = elem_count(y_rec) / (M * N);
    for (size_t bi = 0; bi < batch; bi++) {
        uint32_t y_batch_coords[8] = {0};
        size_t tmp = bi;
        for (int d = (int)y_rec->rank - 3; d >= 0; d--) {
            y_batch_coords[d] = (uint32_t)(tmp % y_rec->shape[d]);
            tmp /= y_rec->shape[d];
        }
        size_t a_batch = 0;
        for (uint16_t d = 0; d + 2 < a_rec->rank; d++) {
            int yd = (int)y_rec->rank - 2 - ((int)a_rec->rank - 2) + d;
            uint32_t coord = yd >= 0 ? y_batch_coords[yd] : 0;
            a_batch = a_batch * a_rec->shape[d] + (a_rec->shape[d] == 1 ? 0 : coord);
        }
        size_t b_batch = 0;
        for (uint16_t d = 0; d + 2 < b_rec->rank; d++) {
            int yd = (int)y_rec->rank - 2 - ((int)b_rec->rank - 2) + d;
            uint32_t coord = yd >= 0 ? y_batch_coords[yd] : 0;
            b_batch = b_batch * b_rec->shape[d] + (b_rec->shape[d] == 1 ? 0 : coord);
        }
        const float *a_base = a + a_batch * M * K;
        const float *b_base = b + b_batch * K * N;
        float *y_base = y + bi * M * N;
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    sum += a_base[m * K + k] * b_base[k * N + n];
                }
                y_base[m * N + n] = sum;
            }
        }
    }
    return 0;
}

static int kernel_resize(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    if (x_rec->rank != 4 || y_rec->rank != 4) return -11;
    int N = (int)y_rec->shape[0], C = (int)y_rec->shape[1], outH = (int)y_rec->shape[2], outW = (int)y_rec->shape[3];
    int H = (int)x_rec->shape[2], W = (int)x_rec->shape[3];
    float scale_h = (float)outH / (float)H;
    float scale_w = (float)outW / (float)W;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < outH; oh++) {
                int ih = (int)floorf((float)oh / scale_h);
                if (ih >= H) ih = H - 1;
                for (int ow = 0; ow < outW; ow++) {
                    int iw = (int)floorf((float)ow / scale_w);
                    if (iw >= W) iw = W - 1;
                    y[((size_t)n * C * outH * outW) + ((size_t)c * outH * outW) + ((size_t)oh * outW) + ow] =
                        x[((size_t)n * C * H * W) + ((size_t)c * H * W) + ((size_t)ih * W) + iw];
                }
            }
        }
    }
    return 0;
}

static int kernel_tile(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = elem_count(y_rec);
    for (size_t i = 0; i < n; i++) {
        uint32_t y_coords[8] = {0};
        uint32_t x_coords[8] = {0};
        linear_to_coords(i, y_rec, y_coords);
        for (uint16_t d = 0; d < x_rec->rank; d++) {
            x_coords[d] = y_coords[d] % x_rec->shape[d];
        }
        y[i] = x[coords_to_linear(x_rec, x_coords)];
    }
    return 0;
}

static int kernel_unsqueeze(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    int axes[8] = {0};
    int axis_count = 0;
    Spkv2AttrRecord attr;
    int rc = get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    rc = tensor_axis_values(ctx, node, &attr, y_rec->rank, axes, &axis_count);
    if (rc != 0) return rc;
    size_t n = elem_count(y_rec);
    for (size_t i = 0; i < n; i++) {
        uint32_t y_coords[8] = {0};
        uint32_t x_coords[8] = {0};
        linear_to_coords(i, y_rec, y_coords);
        uint16_t xd = 0;
        for (uint16_t yd = 0; yd < y_rec->rank; yd++) {
            if (!axis_in_set((int)yd, axes, axis_count) && xd < x_rec->rank) {
                x_coords[xd++] = y_coords[yd];
            }
        }
        y[i] = x[coords_to_linear(x_rec, x_coords)];
    }
    return 0;
}

static int kernel_gather_elements(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *data_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *idx_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *data = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *idx = (const float *)ctx->tensors[node->inputs[1]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    int axis = normalize_axis(attr.axis, data_rec->rank);
    if (axis < 0 || axis >= (int)data_rec->rank) return -11;
    size_t n = elem_count(y_rec);
    for (size_t i = 0; i < n; i++) {
        uint32_t coords[8] = {0};
        linear_to_coords(i, idx_rec, coords);
        int gather_index = (int)idx[i];
        if (gather_index < 0) gather_index += (int)data_rec->shape[axis];
        coords[axis] = (uint32_t)gather_index;
        y[i] = data[coords_to_linear(data_rec, coords)];
    }
    return 0;
}

static int kernel_topk(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *k_rec = ctx->tensors[node->inputs[1]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *k_data = (const float *)ctx->tensors[node->inputs[1]].data;
    float *values = (float *)ctx->tensors[node->outputs[0]].data;
    float *indices = (float *)ctx->tensors[node->outputs[1]].data;
    (void)k_rec;
    int axis = normalize_axis(attr.axis, x_rec->rank);
    if (axis < 0 || axis >= (int)x_rec->rank) return -11;
    size_t k = (size_t)k_data[0];
    size_t dim = x_rec->shape[axis];
    size_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= x_rec->shape[i];
    for (uint16_t i = (uint16_t)axis + 1; i < x_rec->rank; i++) inner *= x_rec->shape[i];
    for (size_t o = 0; o < outer; o++) {
        for (size_t in = 0; in < inner; in++) {
            for (size_t out_i = 0; out_i < k; out_i++) {
                float best_value = attr.largest ? -INFINITY : INFINITY;
                size_t best_index = 0;
                for (size_t d = 0; d < dim; d++) {
                    float v = x[(o * dim + d) * inner + in];
                    int already = 0;
                    for (size_t prev = 0; prev < out_i; prev++) {
                        if ((size_t)indices[(o * k + prev) * inner + in] == d) {
                            already = 1;
                            break;
                        }
                    }
                    if (already) continue;
                    if ((attr.largest && v > best_value) || (!attr.largest && v < best_value)) {
                        best_value = v;
                        best_index = d;
                    }
                }
                values[(o * k + out_i) * inner + in] = best_value;
                indices[(o * k + out_i) * inner + in] = (float)best_index;
            }
        }
    }
    return 0;
}

static const KernelRegistryEntry REGISTRY[] = {
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
