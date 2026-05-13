#include "reference_kernels.h"

int kernel_gemm(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;

    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *c = node->input_count > 2 ? (const float *)ctx->tensors[node->inputs[2]].data : NULL;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int a_rows = attr.trans_a ? (int)a_rec->shape[1] : (int)a_rec->shape[0];
    int a_cols = attr.trans_a ? (int)a_rec->shape[0] : (int)a_rec->shape[1];
    int b_cols = attr.trans_b ? (int)b_rec->shape[0] : (int)b_rec->shape[1];

    for (int m = 0; m < a_rows; m++) {
        for (int n = 0; n < b_cols; n++) {
            float sum = 0.0f;
            for (int k = 0; k < a_cols; k++) {
                float av = attr.trans_a ? a[k * a_rows + m] : a[m * a_cols + k];
                float bv = attr.trans_b ? b[n * a_cols + k] : b[k * b_cols + n];
                sum += av * bv;
            }
            float bias = c ? c[n] : 0.0f;
            y[m * b_cols + n] = attr.alpha * sum + bias;
        }
    }
    return 0;
}

