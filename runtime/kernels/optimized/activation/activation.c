#include "simd_kernels.h"

#ifdef __AVX2__

#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int kernel_relu_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = simd_elem_count(x_rec);

    __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(y + i,
                         _mm256_max_ps(_mm256_loadu_ps(x + i), vzero));
    for (; i < n; i++)
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    return 0;
}



int kernel_sigmoid_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = simd_elem_count(x_rec);

    size_t nb = n & ~(size_t)7;
    #pragma omp parallel for if(nb >= 32768) schedule(static)
    for (size_t i = 0; i < nb; i += 8)
        _mm256_storeu_ps(y + i, sigmoid_avx2(_mm256_loadu_ps(x + i)));
    for (size_t i = nb; i < n; i++)
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    return 0;
}



int kernel_softmax_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    Spkv2AttrRecord attr;
    if (simd_get_attr(ctx, node, &attr) != 0) return -10;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int rank = (int)x_rec->rank;
    int axis = attr.axis < 0 ? rank + attr.axis : attr.axis;
    if (axis < 0 || axis >= rank) return -11;

    size_t outer = 1, inner = 1;
    size_t dim = (size_t)x_rec->shape[axis];
    for (int i = 0; i < axis; i++) outer *= (size_t)x_rec->shape[i];
    for (int i = axis + 1; i < rank; i++) inner *= (size_t)x_rec->shape[i];

    if (inner == 1) {
        /* ---- Fast path: softmax along last axis (contiguous) ---- */
        #pragma omp parallel for if(outer >= 16) schedule(static)
        for (size_t o = 0; o < outer; o++) {
            const float *row = x + o * dim;
            float *out = y + o * dim;

            /* 1) max */
            size_t j = 0;
            __m256 vmax = _mm256_set1_ps(-INFINITY);
            for (; j + 8 <= dim; j += 8)
                vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row + j));
            float maxv = (dim >= 8) ? hmax_avx2(vmax) : -INFINITY;
            for (; j < dim; j++) if (row[j] > maxv) maxv = row[j];

            /* 2) e = exp(x - max), sum */
            __m256 vmaxb = _mm256_set1_ps(maxv);
            __m256 vsum = _mm256_setzero_ps();
            j = 0;
            for (; j + 8 <= dim; j += 8) {
                __m256 e = fast_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(row + j), vmaxb));
                _mm256_storeu_ps(out + j, e);
                vsum = _mm256_add_ps(vsum, e);
            }
            float sum = (dim >= 8) ? hsum_avx2(vsum) : 0.0f;
            for (; j < dim; j++) {
                float e = expf(row[j] - maxv);
                out[j] = e;
                sum += e;
            }

            /* 3) normalize */
            float inv = 1.0f / sum;
            __m256 vinv = _mm256_set1_ps(inv);
            j = 0;
            for (; j + 8 <= dim; j += 8)
                _mm256_storeu_ps(out + j, _mm256_mul_ps(_mm256_loadu_ps(out + j), vinv));
            for (; j < dim; j++) out[j] *= inv;
        }
        return 0;
    }

    /* ---- Middle/outer axis softmax: vectorize across the inner dim ---- */
    /* For each (o, i) pair, compute softmax over d in [0, dim). */
    #pragma omp parallel for if(outer >= 8) schedule(static)
    for (size_t o = 0; o < outer; o++) {
        float *yo = y + o * dim * inner;
        const float *xo = x + o * dim * inner;
        size_t i = 0;
        /* SIMD: process 8 inner positions at a time. */
        for (; i + 8 <= inner; i += 8) {
            /* 1) max over d */
            __m256 vmax = _mm256_loadu_ps(xo + i);
            for (size_t d = 1; d < dim; d++)
                vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(xo + d * inner + i));
            /* 2) exp(x - max), sum */
            __m256 vsum = _mm256_setzero_ps();
            for (size_t d = 0; d < dim; d++) {
                __m256 e = fast_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(xo + d * inner + i), vmax));
                _mm256_storeu_ps(yo + d * inner + i, e);
                vsum = _mm256_add_ps(vsum, e);
            }
            /* 3) normalize: y /= sum ≈ y * rcp(sum) with NR step */
            __m256 rcp = _mm256_rcp_ps(vsum);
            rcp = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(rcp, vsum, _mm256_set1_ps(2.0f)));
            for (size_t d = 0; d < dim; d++)
                _mm256_storeu_ps(yo + d * inner + i,
                    _mm256_mul_ps(_mm256_loadu_ps(yo + d * inner + i), rcp));
        }
        /* Scalar tail */
        for (; i < inner; i++) {
            float maxv = -INFINITY;
            for (size_t d = 0; d < dim; d++) {
                float v = xo[d * inner + i];
                if (v > maxv) maxv = v;
            }
            float sum = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                float e = expf(xo[d * inner + i] - maxv);
                yo[d * inner + i] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            for (size_t d = 0; d < dim; d++) yo[d * inner + i] *= inv;
        }
    }
    return 0;
}


#elif defined(__ARM_NEON)

#include <math.h>
#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

int kernel_relu_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = simd_elem_count(x_rec);

    float32x4_t vzero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(y + i, vmaxq_f32(vld1q_f32(x + i), vzero));
    for (; i < n; i++)
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    return 0;
}



int kernel_sigmoid_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = simd_elem_count(x_rec);

#if defined(__APPLE__)
    int nn = (int)n;
    vDSP_vneg(x, 1, y, 1, n);
    vvexpf(y, y, &nn);
    float one = 1.0f;
    vDSP_vsadd(y, 1, &one, y, 1, n);
    vDSP_svdiv(&one, y, 1, y, 1, n);
#else
    size_t nb = n & ~(size_t)3;
    #pragma omp parallel for if(nb >= 32768) schedule(static)
    for (size_t i = 0; i < nb; i += 4)
        vst1q_f32(y + i, sigmoid_neon(vld1q_f32(x + i)));
    for (size_t i = nb; i < n; i++)
        y[i] = 1.0f / (1.0f + expf(-x[i]));
#endif
    return 0;
}



int kernel_softmax_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    Spkv2AttrRecord attr;
    if (simd_get_attr(ctx, node, &attr) != 0) return -10;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int rank = (int)x_rec->rank;
    int axis = attr.axis < 0 ? rank + attr.axis : attr.axis;
    if (axis < 0 || axis >= rank) return -11;

    size_t outer = 1, inner = 1;
    size_t dim = (size_t)x_rec->shape[axis];
    for (int i = 0; i < axis; i++) outer *= (size_t)x_rec->shape[i];
    for (int i = axis + 1; i < rank; i++) inner *= (size_t)x_rec->shape[i];

    if (inner == 1) {
        /* ---- Fast path: softmax along last axis (contiguous) ---- */
        #pragma omp parallel for if(outer >= 16) schedule(static)
        for (size_t o = 0; o < outer; o++) {
            const float *row = x + o * dim;
            float *out = y + o * dim;

#if defined(__APPLE__)
            /* 1) max */
            float maxv;
            vDSP_maxv(row, 1, &maxv, dim);
            /* 2) exp(x - max) */
            float neg_max = -maxv;
            vDSP_vsadd(row, 1, &neg_max, out, 1, dim);
            int nn = (int)dim;
            vvexpf(out, out, &nn);
            /* 3) sum + normalize */
            float sum;
            vDSP_sve(out, 1, &sum, dim);
            float inv = 1.0f / sum;
            vDSP_vsmul(out, 1, &inv, out, 1, dim);
#else
            /* 1) max */
            size_t j = 0;
            float32x4_t vmax_val = vdupq_n_f32(-INFINITY);
            for (; j + 4 <= dim; j += 4)
                vmax_val = vmaxq_f32(vmax_val, vld1q_f32(row + j));
            float maxv = (dim >= 4) ? vmaxvq_f32(vmax_val) : -INFINITY;
            for (; j < dim; j++) if (row[j] > maxv) maxv = row[j];
            /* 2) exp(x - max), sum */
            float32x4_t vmxb = vdupq_n_f32(maxv);
            float32x4_t vsum = vdupq_n_f32(0.0f);
            j = 0;
            for (; j + 4 <= dim; j += 4) {
                float32x4_t e = fast_exp_neon(vsubq_f32(vld1q_f32(row + j), vmxb));
                vst1q_f32(out + j, e);
                vsum = vaddq_f32(vsum, e);
            }
            float sum = (dim >= 4) ? vaddvq_f32(vsum) : 0.0f;
            for (; j < dim; j++) { float e = expf(row[j] - maxv); out[j] = e; sum += e; }
            /* 3) normalize */
            float inv = 1.0f / sum;
            float32x4_t vinv = vdupq_n_f32(inv);
            j = 0;
            for (; j + 4 <= dim; j += 4)
                vst1q_f32(out + j, vmulq_f32(vld1q_f32(out + j), vinv));
            for (; j < dim; j++) out[j] *= inv;
#endif
        }
        return 0;
    }

    /* ---- Middle/outer axis softmax: vectorize across the inner dim ---- */
    /* For each (o, i) pair, compute softmax over d in [0, dim). */
#if defined(__APPLE__)
    for (size_t o = 0; o < outer; o++) {
        float *yo = y + o * dim * inner;
        const float *xo = x + o * dim * inner;
        /* 1) find max across dim for each inner position */
        memcpy(yo, xo, inner * sizeof(float));
        for (size_t d = 1; d < dim; d++)
            vDSP_vmax(yo, 1, xo + d * inner, 1, yo, 1, inner);
        /* 2) exp(x - max) and accumulate sum */
        float *sum_buf = (float *)calloc(inner, sizeof(float));
        if (!sum_buf) return -1;
        for (size_t d = 0; d < dim; d++) {
            float *out_d = yo + d * inner;
            vDSP_vsub(yo, 1, xo + d * inner, 1, out_d, 1, inner);
            int nn = (int)inner;
            vvexpf(out_d, out_d, &nn);
            vDSP_vadd(sum_buf, 1, out_d, 1, sum_buf, 1, inner);
        }
        /* 3) normalize */
        for (size_t d = 0; d < dim; d++)
            vDSP_vdiv(sum_buf, 1, yo + d * inner, 1, yo + d * inner, 1, inner);
        free(sum_buf);
    }
    return 0;
#endif
    #pragma omp parallel for if(outer >= 8) schedule(static)
    for (size_t o = 0; o < outer; o++) {
        float *yo = y + o * dim * inner;
        const float *xo = x + o * dim * inner;
        size_t i = 0;
        /* SIMD: process 4 inner positions at a time. */
        for (; i + 4 <= inner; i += 4) {
            /* 1) max over d */
            float32x4_t vmax = vld1q_f32(xo + i);
            for (size_t d = 1; d < dim; d++)
                vmax = vmaxq_f32(vmax, vld1q_f32(xo + d * inner + i));
            /* 2) exp(x - max), sum */
            float32x4_t vsum = vdupq_n_f32(0.0f);
            for (size_t d = 0; d < dim; d++) {
                float32x4_t e = fast_exp_neon(vsubq_f32(vld1q_f32(xo + d * inner + i), vmax));
                vst1q_f32(yo + d * inner + i, e);
                vsum = vaddq_f32(vsum, e);
            }
            /* 3) normalize: y /= sum using reciprocal estimate + one NR step */
            float32x4_t rcp = vrecpeq_f32(vsum);
            rcp = vmulq_f32(rcp, vrecpsq_f32(rcp, vsum));
            for (size_t d = 0; d < dim; d++)
                vst1q_f32(yo + d * inner + i,
                    vmulq_f32(vld1q_f32(yo + d * inner + i), rcp));
        }
        /* Scalar tail */
        for (; i < inner; i++) {
            float maxv = -INFINITY;
            for (size_t d = 0; d < dim; d++) {
                float v = xo[d * inner + i];
                if (v > maxv) maxv = v;
            }
            float sum = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                float e = expf(xo[d * inner + i] - maxv);
                yo[d * inner + i] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            for (size_t d = 0; d < dim; d++) yo[d * inner + i] *= inv;
        }
    }
    return 0;
}


#endif /* __AVX2__ || __ARM_NEON */
