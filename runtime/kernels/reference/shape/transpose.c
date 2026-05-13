#include "reference_kernels.h"

int kernel_transpose(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = spkv2_kernel_elem_count(y_rec);
    for (size_t i = 0; i < n; i++) {
        uint32_t y_coords[8] = {0};
        uint32_t x_coords[8] = {0};
        spkv2_kernel_linear_to_coords(i, y_rec, y_coords);
        for (uint16_t d = 0; d < x_rec->rank; d++) {
            int src_axis = attr.extra_count > 0 ? attr.extra[d] : (int)d;
            x_coords[src_axis] = y_coords[d];
        }
        y[i] = x[spkv2_kernel_coords_to_linear(x_rec, x_coords)];
    }
    return 0;
}

