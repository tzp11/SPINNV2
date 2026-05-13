#include "reference_kernels.h"

#include <string.h>

int kernel_split(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    int axis = spkv2_kernel_normalize_axis(attr.axis, x_rec->rank);
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

