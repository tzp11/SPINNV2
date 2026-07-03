#include "simd_kernels.h"
#include "simd_sgemm.h"

#if defined(__AVX2__)

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
    if (b_rec->dtype == SPKV2_DTYPE_FP16) return -99;
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


#elif defined(__ARM_NEON)

#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

int kernel_gemm_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = simd_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;

    /* fall back for transposed inputs – not the hot path */
#if defined(__APPLE__)
    if (attr.trans_a || attr.trans_b) {
        const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
        const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
        const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
        const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
        if (b_rec->dtype == SPKV2_DTYPE_FP16 && ctx->node_cache
            && node->id < ctx->node_cache_count) {
            if (!ctx->node_cache[node->id]) {
                size_t b_elems = 1;
                for (int d = 0; d < b_rec->rank; d++) b_elems *= b_rec->shape[d];
                float *b32 = (float *)malloc(b_elems * sizeof(float));
                if (!b32) return -1;
                const _Float16 *b16 = (const _Float16 *)ctx->tensors[node->inputs[1]].data;
                for (size_t i = 0; i < b_elems; i++) b32[i] = (float)b16[i];
                ctx->node_cache[node->id] = b32;
            }
            b = (const float *)ctx->node_cache[node->id];
        }
        const float *bias = node->input_count > 2
                             ? (const float *)ctx->tensors[node->inputs[2]].data
                             : NULL;
        float *y = (float *)ctx->tensors[node->outputs[0]].data;

        int rows  = attr.trans_a ? (int)a_rec->shape[1] : (int)a_rec->shape[0];
        int inner = attr.trans_a ? (int)a_rec->shape[0] : (int)a_rec->shape[1];
        int cols  = attr.trans_b ? (int)b_rec->shape[0] : (int)b_rec->shape[1];
        float alpha = attr.alpha;

        enum CBLAS_TRANSPOSE trans_a_flag = attr.trans_a ? CblasTrans : CblasNoTrans;
        enum CBLAS_TRANSPOSE trans_b_flag = attr.trans_b ? CblasTrans : CblasNoTrans;
        int lda = attr.trans_a ? rows : inner;
        int ldb = attr.trans_b ? inner : cols;

        /* Init with bias */
        for (int m = 0; m < rows; m++) {
            for (int n = 0; n < cols; n++)
                y[m * cols + n] = bias ? bias[n] : 0.0f;
        }

        float beta_t = bias ? attr.beta : 0.0f;
        cblas_sgemm(CblasRowMajor, trans_a_flag, trans_b_flag,
                    rows, cols, inner, alpha, a, lda, b, ldb, beta_t, y, cols);
        return 0;
    }
#else
    if (attr.trans_a || attr.trans_b) return -99;
#endif

    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    if (b_rec->dtype == SPKV2_DTYPE_FP16 && ctx->node_cache
        && node->id < ctx->node_cache_count) {
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
        if (!ctx->node_cache[node->id]) {
            size_t b_elems = 1;
            for (int d = 0; d < b_rec->rank; d++) b_elems *= b_rec->shape[d];
            float *b32 = (float *)malloc(b_elems * sizeof(float));
            if (!b32) return -1;
            const _Float16 *b16 = (const _Float16 *)ctx->tensors[node->inputs[1]].data;
            for (size_t i = 0; i < b_elems; i++) b32[i] = (float)b16[i];
            ctx->node_cache[node->id] = b32;
        }
        b = (const float *)ctx->node_cache[node->id];
#else
        return -99;
#endif
    }
    const float *bias = node->input_count > 2
                         ? (const float *)ctx->tensors[node->inputs[2]].data
                         : NULL;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int rows  = (int)a_rec->shape[0];
    int inner = (int)a_rec->shape[1];
    int cols  = (int)b_rec->shape[1];

    float alpha = attr.alpha;

#if defined(__APPLE__)
    /* Fast path: fill Y with bias, then cblas_sgemm with beta */
    for (int m = 0; m < rows; m++) {
        if (bias)
            memcpy(y + (size_t)m * cols, bias, (size_t)cols * sizeof(float));
        else {
            float zero = 0.0f;
            vDSP_vfill(&zero, y + (size_t)m * cols, 1, (vDSP_Length)cols);
        }
    }
    float beta_val = bias ? attr.beta : 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, cols, inner, alpha, a, inner, b, cols, beta_val, y, cols);
    return 0;
#endif

    /* initialise Y with bias */
    for (int m = 0; m < rows; m++) {
        float *yr = y + (size_t)m * cols;
        if (bias) {
            int n = 0;
            for (; n + 3 < cols; n += 4)
                vst1q_f32(yr + n, vld1q_f32(bias + n));
            for (; n < cols; n++)
                yr[n] = bias[n];
        } else {
            memset(yr, 0, (size_t)cols * sizeof(float));
        }
    }

    /* Y += alpha * A * B */
    if (alpha == 1.0f) {
        sgemm_nn(rows, cols, inner, a, inner, b, cols, y, cols);
    } else {
        /* rare: scale A rows by alpha then GEMM */
        sgemm_nn(rows, cols, inner, a, inner, b, cols, y, cols);
        float32x4_t valpha = vdupq_n_f32(alpha);
        float32x4_t vone_minus = vdupq_n_f32(alpha - 1.0f);
        for (int m = 0; m < rows; m++) {
            float *yr = y + (size_t)m * cols;
            int n = 0;
            for (; n + 3 < cols; n += 4) {
                float32x4_t yv = vld1q_f32(yr + n);
                float32x4_t bv = bias ? vld1q_f32(bias + n) : vdupq_n_f32(0);
                /* y = alpha * (y - bias) + bias = alpha*y + (1-alpha)*bias */
                yv = vfmaq_f32(vmulq_f32(valpha, yv), vone_minus, bv);
                vst1q_f32(yr + n, yv);
            }
            for (; n < cols; n++) {
                float bv = bias ? bias[n] : 0.0f;
                yr[n] = alpha * (yr[n] - bv) + bv;
            }
        }
    }
    return 0;
}


#endif /* __AVX2__ || __ARM_NEON */
