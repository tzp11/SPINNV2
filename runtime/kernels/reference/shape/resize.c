#include "reference_kernels.h"

#include <math.h>

int kernel_resize(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    if (x_rec->rank != 4 || y_rec->rank != 4) return -11;
    int N = (int)y_rec->shape[0], C = (int)y_rec->shape[1], outH = (int)y_rec->shape[2], outW = (int)y_rec->shape[3];
    int H = (int)x_rec->shape[2], W = (int)x_rec->shape[3];
    float scale_h = (float)outH / (float)H;
    float scale_w = (float)outW / (float)W;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < outH; oh++) {
                int ih = (int)floorf((float)oh / scale_h);
                if (ih >= H) ih = H - 1;
                for (int ow = 0; ow < outW; ow++) {
                    int iw = (int)floorf((float)ow / scale_w);
                    if (iw >= W) iw = W - 1;
                    y[((size_t)n * C * outH * outW) + ((size_t)c * outH * outW) + ((size_t)oh * outW) + ow] =
                        x[((size_t)n * C * H * W) + ((size_t)c * H * W) + ((size_t)ih * W) + iw];
                }
            }
        }
    }
    return 0;
}

