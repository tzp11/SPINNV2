#include "reference_kernels.h"

#include <string.h>

int kernel_flatten(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    void *dst = ctx->tensors[node->outputs[0]].data;
    const void *src = ctx->tensors[node->inputs[0]].data;
    if (dst != src)
        memcpy(dst, src, x_rec->size_bytes);
    return 0;
}

