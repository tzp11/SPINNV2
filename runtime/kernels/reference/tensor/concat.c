#include "reference_kernels.h"

#include <string.h>

int kernel_concat(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    int axis = spkv2_kernel_normalize_axis(attr.axis, y_rec->rank);
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

