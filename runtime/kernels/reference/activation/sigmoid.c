#include "reference_kernels.h"

#include <math.h>

int kernel_sigmoid(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = spkv2_kernel_elem_count(x_rec);
    for (size_t i = 0; i < n; i++) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
    return 0;
}

