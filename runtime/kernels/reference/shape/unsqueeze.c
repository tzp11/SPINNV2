#include "reference_kernels.h"

int kernel_unsqueeze(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    int axes[8] = {0};
    int axis_count = 0;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    rc = spkv2_kernel_tensor_axis_values(ctx, node, &attr, y_rec->rank, axes, &axis_count);
    if (rc != 0) return rc;
    size_t n = spkv2_kernel_elem_count(y_rec);
    for (size_t i = 0; i < n; i++) {
        uint32_t y_coords[8] = {0};
        uint32_t x_coords[8] = {0};
        spkv2_kernel_linear_to_coords(i, y_rec, y_coords);
        uint16_t xd = 0;
        for (uint16_t yd = 0; yd < y_rec->rank; yd++) {
            if (!spkv2_kernel_axis_in_set((int)yd, axes, axis_count) && xd < x_rec->rank) {
                x_coords[xd++] = y_coords[yd];
            }
        }
        y[i] = x[spkv2_kernel_coords_to_linear(x_rec, x_coords)];
    }
    return 0;
}

