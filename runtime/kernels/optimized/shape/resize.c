#include "simd_common.h"

#if defined(__AVX2__) || defined(__ARM_NEON)

#include <string.h>
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

int kernel_resize_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    if (x_rec->rank != 4 || y_rec->rank != 4) return -11;

    int N = (int)y_rec->shape[0], C = (int)y_rec->shape[1];
    int outH = (int)y_rec->shape[2], outW = (int)y_rec->shape[3];
    int H = (int)x_rec->shape[2], W = (int)x_rec->shape[3];
    int nc_total = N * C;

    /* Fast path: exact 2x nearest-neighbor upsampling (common in YOLO neck) */
    if (outH == 2 * H && outW == 2 * W) {
#if defined(__APPLE__)
        spkv2_apply((size_t)nc_total, ^(size_t nc) {
            const float *src = x + nc * (size_t)(H * W);
            float *dst = y + nc * (size_t)(outH * outW);
            for (int ih = 0; ih < H; ih++) {
                const float *srow = src + ih * W;
                float *drow0 = dst + (2 * ih) * outW;
                float *drow1 = dst + (2 * ih + 1) * outW;
                int iw = 0;
#if defined(__ARM_NEON)
                for (; iw + 3 < W; iw += 4) {
                    float32x4_t v = vld1q_f32(srow + iw);
                    float32x4x2_t vv = vzipq_f32(v, v);
                    vst1q_f32(drow0 + 2*iw,     vv.val[0]);
                    vst1q_f32(drow0 + 2*iw + 4, vv.val[1]);
                }
#endif
                for (; iw < W; iw++)
                    drow0[2*iw] = drow0[2*iw+1] = srow[iw];
                memcpy(drow1, drow0, (size_t)outW * sizeof(float));
            }
        });
#else
        for (int nc = 0; nc < nc_total; nc++) {
            const float *src = x + nc * (size_t)(H * W);
            float *dst = y + nc * (size_t)(outH * outW);
            for (int ih = 0; ih < H; ih++) {
                const float *srow = src + ih * W;
                float *drow0 = dst + (2 * ih) * outW;
                float *drow1 = dst + (2 * ih + 1) * outW;
                for (int iw = 0; iw < W; iw++) drow0[2*iw] = drow0[2*iw+1] = srow[iw];
                memcpy(drow1, drow0, (size_t)outW * sizeof(float));
            }
        }
#endif
        return 0;
    }

    /* Generic nearest-neighbor: GCD parallel over N*C planes */
    float scale_h = (float)H / (float)outH;
    float scale_w = (float)W / (float)outW;
#if defined(__APPLE__)
    spkv2_apply((size_t)nc_total, ^(size_t nc) {
        const float *src = x + nc * (size_t)(H * W);
        float *dst = y + nc * (size_t)(outH * outW);
        for (int oh = 0; oh < outH; oh++) {
            int ih = (int)(oh * scale_h); if (ih >= H) ih = H - 1;
            const float *srow = src + ih * W;
            float *drow = dst + oh * outW;
            for (int ow = 0; ow < outW; ow++) {
                int iw = (int)(ow * scale_w); if (iw >= W) iw = W - 1;
                drow[ow] = srow[iw];
            }
        }
    });
#else
    for (int nc = 0; nc < nc_total; nc++) {
        const float *src = x + nc * (size_t)(H * W);
        float *dst = y + nc * (size_t)(outH * outW);
        for (int oh = 0; oh < outH; oh++) {
            int ih = (int)(oh * scale_h); if (ih >= H) ih = H - 1;
            const float *srow = src + ih * W;
            float *drow = dst + oh * outW;
            for (int ow = 0; ow < outW; ow++) {
                int iw = (int)(ow * scale_w); if (iw >= W) iw = W - 1;
                drow[ow] = srow[iw];
            }
        }
    }
#endif
    return 0;
}

#endif /* __AVX2__ || __ARM_NEON */
