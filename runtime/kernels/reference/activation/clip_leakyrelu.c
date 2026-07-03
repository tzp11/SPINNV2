#include "reference_kernels.h"

#include <math.h>

int kernel_clip(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    float min_val = -INFINITY;
    float max_val = INFINITY;
    if (node->input_count >= 2)
        min_val = *(const float *)ctx->tensors[node->inputs[1]].data;
    if (node->input_count >= 3)
        max_val = *(const float *)ctx->tensors[node->inputs[2]].data;

    size_t n = 1;
    for (int d = 0; d < x_rec->rank; d++)
        n *= x_rec->shape[d];

    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
        y[i] = v;
    }
    return 0;
}

int kernel_leakyrelu(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    float alpha = attr.alpha;

    size_t n = 1;
    for (int d = 0; d < x_rec->rank; d++)
        n *= x_rec->shape[d];

    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v >= 0.0f ? v : v * alpha;
    }
    return 0;
}
