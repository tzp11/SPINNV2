#include "reference_kernels.h"

int kernel_gather_elements(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *data_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *idx_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *data = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *idx = (const float *)ctx->tensors[node->inputs[1]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    int axis = spkv2_kernel_normalize_axis(attr.axis, data_rec->rank);
    if (axis < 0 || axis >= (int)data_rec->rank) return -11;
    size_t n = spkv2_kernel_elem_count(y_rec);
    for (size_t i = 0; i < n; i++) {
        uint32_t coords[8] = {0};
        spkv2_kernel_linear_to_coords(i, idx_rec, coords);
        int gather_index = (int)idx[i];
        if (gather_index < 0) gather_index += (int)data_rec->shape[axis];
        coords[axis] = (uint32_t)gather_index;
        y[i] = data[spkv2_kernel_coords_to_linear(data_rec, coords)];
    }
    return 0;
}

