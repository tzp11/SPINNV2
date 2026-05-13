#include "reference_kernels.h"

#include <math.h>

int kernel_maxpool(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = spkv2_kernel_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int N = (int)x_rec->shape[0];
    int C = (int)x_rec->shape[1];
    int H = (int)x_rec->shape[2];
    int W = (int)x_rec->shape[3];
    int outH = (int)y_rec->shape[2];
    int outW = (int)y_rec->shape[3];

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float maxv = -INFINITY;
                    for (int kh = 0; kh < attr.kernel_shape[0]; kh++) {
                        int ih = oh * attr.strides[0] + kh - attr.pads[0];
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < attr.kernel_shape[1]; kw++) {
                            int iw = ow * attr.strides[1] + kw - attr.pads[1];
                            if (iw < 0 || iw >= W) continue;
                            float v = x[((size_t)n * C * H * W) + ((size_t)c * H * W) + ((size_t)ih * W) + iw];
                            if (v > maxv) maxv = v;
                        }
                    }
                    y[((size_t)n * C * outH * outW) + ((size_t)c * outH * outW) + ((size_t)oh * outW) + ow] = maxv;
                }
            }
        }
    }
    return 0;
}

