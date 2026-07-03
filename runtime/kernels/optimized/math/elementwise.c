#include "simd_kernels.h"

#if defined(__AVX2__)

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


#elif defined(__ARM_NEON)

#include <math.h>
#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

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
#if defined(__APPLE__)
        vDSP_vadd(a, 1, b, 1, y, 1, n);
        if (attr.fused_activation == 1) {
            float zero = 0.0f;
            vDSP_vthres(y, 1, &zero, y, 1, n);
        } else if (attr.fused_activation == 2) {
            fused_activation_pass(y, n, attr.fused_activation);
        }
#else
        size_t i = 0;
        size_t n4 = n & ~(size_t)3;
        for (i = 0; i < n4; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vy = vaddq_f32(va, vb);
            if (attr.fused_activation == 1)
                vy = vmaxq_f32(vy, vdupq_n_f32(0.0f));
            else if (attr.fused_activation == 2)
                vy = apply_activation_neon(vy, attr.fused_activation);
            vst1q_f32(y + i, vy);
        }
        for (; i < n; i++) {
            y[i] = a[i] + b[i];
            y[i] = apply_activation_scalar_simd(y[i], attr.fused_activation);
        }
#endif
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

    if (same_shape(a_rec, y_rec) && same_shape(b_rec, y_rec)) {
#if defined(__APPLE__)
        vDSP_vmul(a, 1, b, 1, y, 1, n);
#else
        size_t i = 0;
        size_t n4 = n & ~(size_t)3;
        #pragma omp parallel for if(n4 >= 65536) schedule(static)
        for (i = 0; i < n4; i += 4)
            vst1q_f32(y + i,
                      vmulq_f32(vld1q_f32(a + i),
                                vld1q_f32(b + i)));
        for (; i < n; i++)
            y[i] = a[i] * b[i];
#endif
    } else {
#if defined(__APPLE__)
        size_t na = simd_elem_count(a_rec);
        size_t nb_count = simd_elem_count(b_rec);
        if (nb_count == 1) {
            vDSP_vsmul(a, 1, b, y, 1, n);
            return 0;
        } else if (na == 1) {
            vDSP_vsmul(b, 1, a, y, 1, n);
            return 0;
        }
#endif
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
        size_t i = 0;
        size_t n4 = n & ~(size_t)3;
        #pragma omp parallel for if(n4 >= 65536) schedule(static)
        for (i = 0; i < n4; i += 4)
            vst1q_f32(y + i,
                      vsubq_f32(vld1q_f32(a + i),
                                vld1q_f32(b + i)));
        for (; i < n; i++)
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
        size_t i = 0;
        size_t n4 = n & ~(size_t)3;
        #pragma omp parallel for if(n4 >= 65536) schedule(static)
        for (i = 0; i < n4; i += 4)
            vst1q_f32(y + i,
                      vdivq_f32(vld1q_f32(a + i),
                                vld1q_f32(b + i)));
        for (; i < n; i++)
            y[i] = a[i] / b[i];
    } else {
        for (size_t i = 0; i < n; i++)
            y[i] = a[simd_broadcast_index(a_rec, y_rec, i)] /
                    b[simd_broadcast_index(b_rec, y_rec, i)];
    }
    return 0;
}


#endif /* __AVX2__ / __ARM_NEON */
