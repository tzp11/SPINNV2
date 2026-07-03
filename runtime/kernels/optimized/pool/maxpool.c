#include "simd_common.h"

#if defined(__AVX2__) || defined(__ARM_NEON)

#include <math.h>
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

int kernel_maxpool_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = simd_get_attr(ctx, node, &attr);
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
    int kH = attr.kernel_shape[0];
    int kW = attr.kernel_shape[1];
    int sH = attr.strides[0];
    int sW = attr.strides[1];
    int pH = attr.pads[0];
    int pW = attr.pads[1];

    size_t in_plane = (size_t)H * W;
    size_t out_plane = (size_t)outH * outW;
    int total_channels = N * C;

    /* Parallelize over channels with GCD on Apple, sequential otherwise */
#if defined(__APPLE__)
    spkv2_apply((size_t)total_channels, ^(size_t _idx) {
        int nc = (int)_idx;
#else
    for (int nc = 0; nc < total_channels; nc++) {
#endif
        const float *x_c = x + (size_t)nc * in_plane;
        float *y_c = y + (size_t)nc * out_plane;

        for (int oh = 0; oh < outH; oh++) {
            int ih_start = oh * sH - pH;
            int ih_end = ih_start + kH;
            if (ih_start < 0) ih_start = 0;
            if (ih_end > H) ih_end = H;

            for (int ow = 0; ow < outW; ow++) {
                int iw_start = ow * sW - pW;
                int iw_end = iw_start + kW;
                if (iw_start < 0) iw_start = 0;
                if (iw_end > W) iw_end = W;

                float maxv = -INFINITY;
                for (int ih = ih_start; ih < ih_end; ih++) {
                    const float *row = x_c + (size_t)ih * W + iw_start;
                    int rw = iw_end - iw_start;
                    int j = 0;
#if defined(__ARM_NEON)
                    float32x4_t vmax = vdupq_n_f32(-INFINITY);
                    for (; j + 3 < rw; j += 4)
                        vmax = vmaxq_f32(vmax, vld1q_f32(row + j));
                    float row_max = vmaxvq_f32(vmax);
                    for (; j < rw; j++)
                        if (row[j] > row_max) row_max = row[j];
#elif defined(__AVX2__)
                    __m256 vmax = _mm256_set1_ps(-INFINITY);
                    for (; j + 7 < rw; j += 8)
                        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row + j));
                    float row_max = hmax_avx2(vmax);
                    for (; j < rw; j++)
                        if (row[j] > row_max) row_max = row[j];
#else
                    float row_max = -INFINITY;
                    for (; j < rw; j++)
                        if (row[j] > row_max) row_max = row[j];
#endif
                    if (row_max > maxv) maxv = row_max;
                }
                y_c[oh * outW + ow] = maxv;
            }
        }
#if defined(__APPLE__)
    });
#else
    }
#endif
    return 0;
}

#endif /* __AVX2__ || __ARM_NEON */
