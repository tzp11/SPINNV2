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


#endif /* __AVX2__ */
