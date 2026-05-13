#include "reference_kernels.h"

#include <math.h>

int kernel_softmax(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int axis = attr.axis < 0 ? (int)x_rec->rank + attr.axis : attr.axis;
    if (axis < 0 || axis >= (int)x_rec->rank) return -11;

    size_t outer = 1, inner = 1;
    size_t dim = x_rec->shape[axis];
    for (int i = 0; i < axis; i++) outer *= x_rec->shape[i];
    for (uint16_t i = (uint16_t)axis + 1; i < x_rec->rank; i++) inner *= x_rec->shape[i];

    for (size_t o = 0; o < outer; o++) {
        for (size_t in = 0; in < inner; in++) {
            float maxv = -INFINITY;
            for (size_t d = 0; d < dim; d++) {
                float v = x[(o * dim + d) * inner + in];
                if (v > maxv) maxv = v;
            }
            float sum = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                float e = expf(x[(o * dim + d) * inner + in] - maxv);
                y[(o * dim + d) * inner + in] = e;
                sum += e;
            }
            for (size_t d = 0; d < dim; d++) {
                y[(o * dim + d) * inner + in] /= sum;
            }
        }
    }
    return 0;
}

