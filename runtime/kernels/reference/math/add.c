#include "reference_kernels.h"

int kernel_add(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = spkv2_kernel_elem_count(y_rec);
    for (size_t i = 0; i < n; i++) {
        float sum = a[spkv2_kernel_broadcast_index(a_rec, y_rec, i)] + b[spkv2_kernel_broadcast_index(b_rec, y_rec, i)];
        y[i] = spkv2_kernel_apply_fused_activation_scalar(sum, attr.fused_activation);
    }
    return 0;
}

