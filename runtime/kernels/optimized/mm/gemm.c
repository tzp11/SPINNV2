#include "simd_kernels.h"
#include "simd_sgemm.h"

#ifdef __AVX2__

#include <stdlib.h>
#include <string.h>

int kernel_gemm_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = simd_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;

    /* fall back for transposed inputs – not the hot path */
    if (attr.trans_a || attr.trans_b) return -99;

    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *c = node->input_count > 2
                         ? (const float *)ctx->tensors[node->inputs[2]].data
                         : NULL;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int rows  = (int)a_rec->shape[0];
    int inner = (int)a_rec->shape[1];
    int cols  = (int)b_rec->shape[1];

    /* initialise Y with bias */
    for (int m = 0; m < rows; m++) {
        float *yr = y + (size_t)m * cols;
        if (c) {
            int n = 0;
            for (; n + 7 < cols; n += 8)
                _mm256_storeu_ps(yr + n, _mm256_loadu_ps(c + n));
            for (; n < cols; n++)
                yr[n] = c[n];
        } else {
            memset(yr, 0, (size_t)cols * sizeof(float));
        }
    }

    /* Y += alpha * A * B */
    if (attr.alpha == 1.0f) {
        sgemm_nn(rows, cols, inner, a, inner, b, cols, y, cols);
    } else {
        /* rare: scale A rows by alpha then GEMM (TODO: fuse alpha into micro-kernel) */
        sgemm_nn(rows, cols, inner, a, inner, b, cols, y, cols);
        __m256 valpha = _mm256_set1_ps(attr.alpha);
        __m256 vone_minus = _mm256_set1_ps(attr.alpha - 1.0f);
        /* Y currently has bias + A*B; we need bias + alpha*A*B */
        /* Y_correct = bias + alpha*(Y - bias) = alpha*Y + (1-alpha)*bias */
        /* Simpler: just do the multiply-accumulate correctly */
        /* Actually re-do: zero Y, GEMM, then scale and add bias */
        for (int m = 0; m < rows; m++) {
            float *yr = y + (size_t)m * cols;
            int n = 0;
            for (; n + 7 < cols; n += 8) {
                __m256 yv = _mm256_loadu_ps(yr + n);
                __m256 bv = c ? _mm256_loadu_ps(c + n) : _mm256_setzero_ps();
                /* y = alpha * (y - bias) + bias = alpha*y + (1-alpha)*bias */
                yv = _mm256_fmadd_ps(vone_minus, bv,
                                     _mm256_mul_ps(valpha, yv));
                _mm256_storeu_ps(yr + n, yv);
            }
            for (; n < cols; n++) {
                float bv = c ? c[n] : 0.0f;
                yr[n] = attr.alpha * (yr[n] - bv) + bv;
            }
        }
    }
    return 0;
}


#endif /* __AVX2__ */
