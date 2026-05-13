#include "reference_kernels.h"

#include <string.h>

int kernel_flatten(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    memcpy(ctx->tensors[node->outputs[0]].data, ctx->tensors[node->inputs[0]].data, x_rec->size_bytes);
    return 0;
}

