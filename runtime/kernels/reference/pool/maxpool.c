#include "reference_kernels.h"

#include <math.h>

static inline float maxpool_scalar_at(const float *x, int H, int W,
                                      int oh, int ow,
                                      int kH, int kW,
                                      int sH, int sW,
                                      int pH, int pW)
{
    float maxv = -INFINITY;
    for (int kh = 0; kh < kH; kh++) {
        int ih = oh * sH + kh - pH;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kW; kw++) {
            int iw = ow * sW + kw - pW;
            if (iw < 0 || iw >= W) continue;
            float v = x[(size_t)ih * W + iw];
            if (v > maxv) maxv = v;
        }
    }
    return maxv;
}

static inline float max25_center(const float *x, int W, int oh, int ow)
{
    const float *r0 = x + (size_t)(oh - 2) * W + ow - 2;
    const float *r1 = r0 + W;
    const float *r2 = r1 + W;
    const float *r3 = r2 + W;
    const float *r4 = r3 + W;
    float m = r0[0];
#define MAXV(v) do { float _v = (v); if (_v > m) m = _v; } while (0)
    MAXV(r0[1]); MAXV(r0[2]); MAXV(r0[3]); MAXV(r0[4]);
    MAXV(r1[0]); MAXV(r1[1]); MAXV(r1[2]); MAXV(r1[3]); MAXV(r1[4]);
    MAXV(r2[0]); MAXV(r2[1]); MAXV(r2[2]); MAXV(r2[3]); MAXV(r2[4]);
    MAXV(r3[0]); MAXV(r3[1]); MAXV(r3[2]); MAXV(r3[3]); MAXV(r3[4]);
    MAXV(r4[0]); MAXV(r4[1]); MAXV(r4[2]); MAXV(r4[3]); MAXV(r4[4]);
#undef MAXV
    return m;
}

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

    if (attr.kernel_shape[0] == 5 && attr.kernel_shape[1] == 5 &&
        attr.strides[0] == 1 && attr.strides[1] == 1 &&
        attr.pads[0] == 2 && attr.pads[1] == 2 &&
        outH == H && outW == W) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                const float *x_nc = x + ((size_t)n * C + c) * H * W;
                float *y_nc = y + ((size_t)n * C + c) * outH * outW;
                for (int oh = 0; oh < outH; oh++) {
                    int ow = 0;
                    for (; ow < outW && ow < 2; ow++) {
                        y_nc[(size_t)oh * outW + ow] =
                            maxpool_scalar_at(x_nc, H, W, oh, ow, 5, 5, 1, 1, 2, 2);
                    }
                    if (oh >= 2 && oh + 2 < H) {
                        for (; ow + 2 < outW; ow++) {
                            y_nc[(size_t)oh * outW + ow] = max25_center(x_nc, W, oh, ow);
                        }
                    }
                    for (; ow < outW; ow++) {
                        y_nc[(size_t)oh * outW + ow] =
                            maxpool_scalar_at(x_nc, H, W, oh, ow, 5, 5, 1, 1, 2, 2);
                    }
                }
            }
        }
        return 0;
    }

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float *x_nc = x + ((size_t)n * C + c) * H * W;
            float *y_nc = y + ((size_t)n * C + c) * outH * outW;
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    y_nc[(size_t)oh * outW + ow] =
                        maxpool_scalar_at(x_nc, H, W, oh, ow,
                                          attr.kernel_shape[0], attr.kernel_shape[1],
                                          attr.strides[0], attr.strides[1],
                                          attr.pads[0], attr.pads[1]);
                }
            }
        }
    }
    return 0;
}
