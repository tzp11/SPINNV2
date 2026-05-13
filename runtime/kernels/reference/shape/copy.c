#include "reference_kernels.h"

#include <string.h>

int kernel_copy(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    memcpy(ctx->tensors[node->outputs[0]].data, ctx->tensors[node->inputs[0]].data, y_rec->size_bytes);
    return 0;
}

