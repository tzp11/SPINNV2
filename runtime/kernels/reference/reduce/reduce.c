#include "reference_kernels.h"

#include <math.h>

int kernel_reduce(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    int axes[8] = {0};
    int axis_count = 0;
    rc = spkv2_kernel_tensor_axis_values(ctx, node, &attr, x_rec->rank, axes, &axis_count);
    if (rc != 0) return rc;
    size_t x_count = spkv2_kernel_elem_count(x_rec);
    size_t y_count = spkv2_kernel_elem_count(y_rec);
    for (size_t yi = 0; yi < y_count; yi++) {
        y[yi] = node->op_type == SPKV2_OP_REDUCEMAX ? -INFINITY : 0.0f;
    }
    float denom = 1.0f;
    for (int i = 0; i < axis_count; i++) denom *= (float)x_rec->shape[axes[i]];
    for (size_t xi = 0; xi < x_count; xi++) {
        uint32_t x_coords[8] = {0};
        uint32_t y_coords[8] = {0};
        spkv2_kernel_linear_to_coords(xi, x_rec, x_coords);
        uint16_t yd = 0;
        for (uint16_t xd = 0; xd < x_rec->rank; xd++) {
            if (spkv2_kernel_axis_in_set((int)xd, axes, axis_count)) {
                if (attr.keepdims && yd < y_rec->rank) y_coords[yd++] = 0;
            } else if (yd < y_rec->rank) {
                y_coords[yd++] = x_coords[xd];
            }
        }
        size_t yi = spkv2_kernel_coords_to_linear(y_rec, y_coords);
        if (node->op_type == SPKV2_OP_REDUCEMAX) {
            if (x[xi] > y[yi]) y[yi] = x[xi];
        } else {
            y[yi] += x[xi] / denom;
        }
    }
    return 0;
}

