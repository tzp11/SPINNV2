#include "reference_kernels.h"

int kernel_binary(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = spkv2_kernel_elem_count(y_rec);
    for (size_t i = 0; i < n; i++) {
        float av = a[spkv2_kernel_broadcast_index(a_rec, y_rec, i)];
        float bv = b[spkv2_kernel_broadcast_index(b_rec, y_rec, i)];
        switch (node->op_type) {
        case SPKV2_OP_MUL:
            y[i] = av * bv;
            break;
        case SPKV2_OP_SUB:
            y[i] = av - bv;
            break;
        case SPKV2_OP_DIV:
            y[i] = av / bv;
            break;
        case SPKV2_OP_MOD:
            y[i] = (float)((int)av % (int)bv);
            break;
        default:
            return -99;
        }
    }
    return 0;
}

