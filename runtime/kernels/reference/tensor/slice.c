#include "reference_kernels.h"
#include <string.h>

int kernel_slice(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    /* Read starts, ends from input tensors (they are constants/weights). */
    const float *starts_f = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *ends_f = (const float *)ctx->tensors[node->inputs[2]].data;

    int64_t start = (int64_t)starts_f[0];
    int64_t end   = (int64_t)ends_f[0];

    /* axes (optional, input 3) */
    int axis = 0;
    if (node->input_count > 3 && node->inputs[3] != 0xFFFF) {
        const float *axes_f = (const float *)ctx->tensors[node->inputs[3]].data;
        axis = (int)axes_f[0];
    }

    /* steps (optional, input 4) */
    int step = 1;
    if (node->input_count > 4 && node->inputs[4] != 0xFFFF) {
        const float *steps_f = (const float *)ctx->tensors[node->inputs[4]].data;
        step = (int)steps_f[0];
    }

    int rank = (int)x_rec->rank;
    if (axis < 0) axis += rank;

    /* Compute outer, slice_dim, inner sizes */
    size_t outer = 1;
    for (int i = 0; i < axis; i++) outer *= x_rec->shape[i];

    int dim_size = (int)x_rec->shape[axis];
    if (start < 0) start += dim_size;
    if (end < 0)   end   += dim_size;
    if (end > dim_size) end = dim_size;
    if (start < 0) start = 0;
    int out_dim = (int)((end - start + step - 1) / step);

    size_t inner = 1;
    for (int i = axis + 1; i < rank; i++) inner *= x_rec->shape[i];

    for (size_t o = 0; o < outer; o++) {
        for (int s = 0, si = (int)start; s < out_dim; s++, si += step) {
            memcpy(y + (o * (size_t)out_dim + (size_t)s) * inner,
                   x + (o * (size_t)dim_size + (size_t)si) * inner,
                   inner * sizeof(float));
        }
    }
    return 0;
}
