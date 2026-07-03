#include "reference_kernels.h"
#include <string.h>

int kernel_gather(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;

    const Spkv2TensorRecord *data_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *idx_rec  = ctx->tensors[node->inputs[1]].record;
    const float *data      = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *indices_f = (const float *)ctx->tensors[node->inputs[1]].data;
    float *y               = (float *)ctx->tensors[node->outputs[0]].data;

    int axis = attr.axis;
    int rank = (int)data_rec->rank;
    if (axis < 0) axis += rank;

    size_t outer = 1;
    for (int i = 0; i < axis; i++) outer *= data_rec->shape[i];
    int dim_size = (int)data_rec->shape[axis];
    size_t inner = 1;
    for (int i = axis + 1; i < rank; i++) inner *= data_rec->shape[i];

    size_t num_indices = 1;
    for (int i = 0; i < (int)idx_rec->rank; i++) num_indices *= idx_rec->shape[i];

    size_t yi = 0;
    for (size_t o = 0; o < outer; o++) {
        for (size_t idx = 0; idx < num_indices; idx++) {
            int ix = (int)indices_f[idx];
            if (ix < 0) ix += dim_size;
            memcpy(y + yi, data + (o * (size_t)dim_size + (size_t)ix) * inner,
                   inner * sizeof(float));
            yi += inner;
        }
    }
    return 0;
}
