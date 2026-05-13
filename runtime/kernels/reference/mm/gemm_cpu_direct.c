#include "reference_kernels.h"

int kernel_gemm_cpu_direct(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    if (attr.trans_a || attr.trans_b) {
        return kernel_gemm(ctx, node, scratch);
    }

    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *c = node->input_count > 2 ? (const float *)ctx->tensors[node->inputs[2]].data : NULL;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int rows = (int)a_rec->shape[0];
    int inner = (int)a_rec->shape[1];
    int cols = (int)b_rec->shape[1];
    for (int m = 0; m < rows; m++) {
        const float *a_row = a + (size_t)m * inner;
        float *y_row = y + (size_t)m * cols;
        for (int n = 0; n < cols; n++) {
            y_row[n] = c ? c[n] : 0.0f;
        }
        for (int k = 0; k < inner; k++) {
            float av = attr.alpha * a_row[k];
            const float *b_row = b + (size_t)k * cols;
            for (int n = 0; n < cols; n++) {
                y_row[n] += av * b_row[n];
            }
        }
    }
    return 0;
}

