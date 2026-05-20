#include "simd_kernels.h"

#ifdef __AVX2__

#include <math.h>

int kernel_add_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    Spkv2AttrRecord attr;
    int rc = simd_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = simd_elem_count(y_rec);

    if (same_shape(a_rec, y_rec) && same_shape(b_rec, y_rec)) {
        /* fast path: no broadcast */
        size_t nb = n & ~(size_t)7;
        __m256 vzero = _mm256_setzero_ps();
        #pragma omp parallel for if(nb >= 65536) schedule(static)
        for (size_t i = 0; i < nb; i += 8) {
            __m256 v = _mm256_add_ps(_mm256_loadu_ps(a + i),
                                     _mm256_loadu_ps(b + i));
            if (attr.fused_activation == 1) v = _mm256_max_ps(v, vzero);
            _mm256_storeu_ps(y + i, v);
        }
        for (size_t i = nb; i < n; i++)
            y[i] = attr.fused_activation == 1 && a[i] + b[i] < 0.0f ? 0.0f : a[i] + b[i];
    } else {
        /* broadcast fallback */
        for (size_t i = 0; i < n; i++) {
            float v = a[simd_broadcast_index(a_rec, y_rec, i)] +
                      b[simd_broadcast_index(b_rec, y_rec, i)];
            y[i] = attr.fused_activation == 1 && v < 0.0f ? 0.0f : v;
        }
    }
    return 0;
}



int kernel_mul_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = simd_elem_count(y_rec);

    if (b_rec->rank == 0 && same_shape(a_rec, y_rec)) {
        float bv = b[0];
        __m256 vb = _mm256_set1_ps(bv);
        size_t nb = n & ~(size_t)7;
        #pragma omp parallel for if(nb >= 65536) schedule(static)
        for (size_t i = 0; i < nb; i += 8)
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), vb));
        for (size_t i = nb; i < n; i++)
            y[i] = a[i] * bv;
    } else if (a_rec->rank == 0 && same_shape(b_rec, y_rec)) {
        float av = a[0];
        __m256 va = _mm256_set1_ps(av);
        size_t nb = n & ~(size_t)7;
        #pragma omp parallel for if(nb >= 65536) schedule(static)
        for (size_t i = 0; i < nb; i += 8)
            _mm256_storeu_ps(y + i, _mm256_mul_ps(va, _mm256_loadu_ps(b + i)));
        for (size_t i = nb; i < n; i++)
            y[i] = av * b[i];
    } else if (a_rec->rank == 3 && b_rec->rank == 2 && y_rec->rank == 3 &&
               a_rec->shape[0] == 1 && b_rec->shape[0] == 1 &&
               y_rec->shape[0] == 1 &&
               a_rec->shape[1] == y_rec->shape[1] &&
               a_rec->shape[2] == y_rec->shape[2] &&
               b_rec->shape[1] == y_rec->shape[2]) {
        size_t channels = y_rec->shape[1];
        size_t width = y_rec->shape[2];
        size_t nb = width & ~(size_t)7;
        #pragma omp parallel for if(channels * width >= 65536) schedule(static)
        for (size_t c = 0; c < channels; c++) {
            const float *a_ch = a + c * width;
            float *y_ch = y + c * width;
            size_t i = 0;
            for (; i < nb; i += 8)
                _mm256_storeu_ps(y_ch + i,
                                 _mm256_mul_ps(_mm256_loadu_ps(a_ch + i),
                                               _mm256_loadu_ps(b + i)));
            for (; i < width; i++)
                y_ch[i] = a_ch[i] * b[i];
        }
    } else if (b_rec->rank == 3 && a_rec->rank == 2 && y_rec->rank == 3 &&
               b_rec->shape[0] == 1 && a_rec->shape[0] == 1 &&
               y_rec->shape[0] == 1 &&
               b_rec->shape[1] == y_rec->shape[1] &&
               b_rec->shape[2] == y_rec->shape[2] &&
               a_rec->shape[1] == y_rec->shape[2]) {
        size_t channels = y_rec->shape[1];
        size_t width = y_rec->shape[2];
        size_t nb = width & ~(size_t)7;
        #pragma omp parallel for if(channels * width >= 65536) schedule(static)
        for (size_t c = 0; c < channels; c++) {
            const float *b_ch = b + c * width;
            float *y_ch = y + c * width;
            size_t i = 0;
            for (; i < nb; i += 8)
                _mm256_storeu_ps(y_ch + i,
                                 _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                               _mm256_loadu_ps(b_ch + i)));
            for (; i < width; i++)
                y_ch[i] = a[i] * b_ch[i];
        }
    } else
    if (same_shape(a_rec, y_rec) && same_shape(b_rec, y_rec)) {
        size_t nb = n & ~(size_t)7;
        #pragma omp parallel for if(nb >= 65536) schedule(static)
        for (size_t i = 0; i < nb; i += 8)
            _mm256_storeu_ps(y + i,
                             _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                           _mm256_loadu_ps(b + i)));
        for (size_t i = nb; i < n; i++)
            y[i] = a[i] * b[i];
    } else {
        for (size_t i = 0; i < n; i++)
            y[i] = a[simd_broadcast_index(a_rec, y_rec, i)] *
                    b[simd_broadcast_index(b_rec, y_rec, i)];
    }
    return 0;
}



int kernel_sub_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = simd_elem_count(y_rec);

    if (same_shape(a_rec, y_rec) && same_shape(b_rec, y_rec)) {
        size_t nb = n & ~(size_t)7;
        #pragma omp parallel for if(nb >= 65536) schedule(static)
        for (size_t i = 0; i < nb; i += 8)
            _mm256_storeu_ps(y + i,
                             _mm256_sub_ps(_mm256_loadu_ps(a + i),
                                           _mm256_loadu_ps(b + i)));
        for (size_t i = nb; i < n; i++)
            y[i] = a[i] - b[i];
    } else {
        for (size_t i = 0; i < n; i++)
            y[i] = a[simd_broadcast_index(a_rec, y_rec, i)] -
                    b[simd_broadcast_index(b_rec, y_rec, i)];
    }
    return 0;
}



int kernel_div_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    const Spkv2TensorRecord *a_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *b_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *a = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *b = (const float *)ctx->tensors[node->inputs[1]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    size_t n = simd_elem_count(y_rec);

    if (same_shape(a_rec, y_rec) && same_shape(b_rec, y_rec)) {
        size_t nb = n & ~(size_t)7;
        #pragma omp parallel for if(nb >= 65536) schedule(static)
        for (size_t i = 0; i < nb; i += 8)
            _mm256_storeu_ps(y + i,
                             _mm256_div_ps(_mm256_loadu_ps(a + i),
                                           _mm256_loadu_ps(b + i)));
        for (size_t i = nb; i < n; i++)
            y[i] = a[i] / b[i];
    } else {
        for (size_t i = 0; i < n; i++)
            y[i] = a[simd_broadcast_index(a_rec, y_rec, i)] /
                    b[simd_broadcast_index(b_rec, y_rec, i)];
    }
    return 0;
}


#endif /* __AVX2__ */
