#include "simd_common.h"

#ifdef __AVX2__

#include <math.h>
#include <string.h>

size_t simd_elem_count(const Spkv2TensorRecord *r)
{
    size_t c = 1;
    for (uint16_t i = 0; i < r->rank; i++)
        c *= r->shape[i];
    return c;
}



int simd_get_attr(const Spkv2Context *ctx,
                                const Spkv2NodeRecord *node,
                                Spkv2AttrRecord *attr)
{
    if (node->attr_offset > ctx->attrs_size ||
        node->attr_size > ctx->attrs_size - node->attr_offset ||
        node->attr_size < sizeof(Spkv2AttrRecord))
        return -10;
    memcpy(attr, ctx->attrs + node->attr_offset, sizeof(Spkv2AttrRecord));
    return 0;
}



__m256 sigmoid_avx2(__m256 x);



void fused_activation_pass(float *data, size_t count, int act_type)
{
    if (act_type == 0) return;
    size_t i = 0;
    if (act_type == 1) { /* Relu */
        __m256 vzero = _mm256_setzero_ps();
        for (; i + 7 < count; i += 8)
            _mm256_storeu_ps(data + i,
                             _mm256_max_ps(_mm256_loadu_ps(data + i), vzero));
        for (; i < count; i++)
            if (data[i] < 0.0f) data[i] = 0.0f;
    } else if (act_type == 2) { /* SiLU: x * sigmoid(x) */
        for (; i + 7 < count; i += 8) {
            __m256 x = _mm256_loadu_ps(data + i);
            _mm256_storeu_ps(data + i, _mm256_mul_ps(x, sigmoid_avx2(x)));
        }
        for (; i < count; i++)
            data[i] = data[i] / (1.0f + expf(-data[i]));
    }
}



float apply_activation_scalar_simd(float x, int act_type)
{
    if (act_type == 1) return x > 0.0f ? x : 0.0f;
    if (act_type == 2) return x / (1.0f + expf(-x));
    return x;
}



__m256 apply_activation_avx2(__m256 x, int act_type)
{
    if (act_type == 1) return _mm256_max_ps(x, _mm256_setzero_ps());
    if (act_type == 2) return _mm256_mul_ps(x, sigmoid_avx2(x));
    return x;
}



const Spkv2KernelSpecRecord *simd_node_spec(const Spkv2Context *ctx,
                                                    const Spkv2NodeRecord *node)
{
    if (node->kernel_spec_id == 0xFFFFFFFFu || node->kernel_spec_id >= ctx->kernel_spec_count)
        return NULL;
    return &ctx->kernel_spec_records[node->kernel_spec_id];
}



size_t simd_broadcast_index(const Spkv2TensorRecord *in_rec,
                                   const Spkv2TensorRecord *out_rec,
                                   size_t out_index)
{
    uint32_t out_c[8] = {0}, in_c[8] = {0};
    size_t tmp = out_index;
    for (int i = (int)out_rec->rank - 1; i >= 0; i--) {
        uint32_t d = out_rec->shape[i];
        out_c[i] = d == 0 ? 0 : (uint32_t)(tmp % d);
        tmp = d == 0 ? 0 : tmp / d;
    }
    int delta = (int)out_rec->rank - (int)in_rec->rank;
    for (uint16_t i = 0; i < in_rec->rank; i++) {
        uint32_t coord = out_c[i + delta];
        in_c[i] = in_rec->shape[i] == 1 ? 0 : coord;
    }
    size_t idx = 0;
    for (uint16_t i = 0; i < in_rec->rank; i++)
        idx = idx * in_rec->shape[i] + in_c[i];
    return idx;
}



int same_shape(const Spkv2TensorRecord *a, const Spkv2TensorRecord *b)
{
    if (a->rank != b->rank) return 0;
    for (uint16_t i = 0; i < a->rank; i++)
        if (a->shape[i] != b->shape[i]) return 0;
    return 1;
}



__m256 fast_exp_avx2(__m256 x)
{
    /* Clamp to prevent overflow/underflow */
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3f));

    /* exp(x) = 2^(n+f) where n = round(x/ln2), f = x/ln2 - n */
    const __m256 log2e  = _mm256_set1_ps(1.4426950408889634f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);

    __m256 fx = _mm256_mul_ps(x, log2e);
    __m256 n  = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    /* f = x - n * ln2 (high precision) */
    __m256 f = _mm256_sub_ps(x, _mm256_mul_ps(n, ln2_hi));
    f = _mm256_sub_ps(f, _mm256_mul_ps(n, ln2_lo));

    /* Polynomial: exp(f) ≈ 1 + f + f²/2 + f³/6 + f⁴/24 + f⁵/120 */
    __m256 y = _mm256_set1_ps(1.0f / 120.0f);
    y = _mm256_fmadd_ps(y, f, _mm256_set1_ps(1.0f / 24.0f));
    y = _mm256_fmadd_ps(y, f, _mm256_set1_ps(1.0f / 6.0f));
    y = _mm256_fmadd_ps(y, f, _mm256_set1_ps(0.5f));
    y = _mm256_fmadd_ps(y, f, _mm256_set1_ps(1.0f));
    y = _mm256_fmadd_ps(y, f, _mm256_set1_ps(1.0f));

    /* 2^n: construct float with exponent = n+127 */
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);
    __m256 pow2n = _mm256_castsi256_ps(ni);

    return _mm256_mul_ps(y, pow2n);
}



__m256 sigmoid_avx2(__m256 x)
{
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg_x = fast_exp_avx2(neg_x);
    __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_x);
    __m256 rcp = _mm256_rcp_ps(denom);
    rcp = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(rcp, denom, _mm256_set1_ps(2.0f)));
    return rcp;
}



float hmax_avx2(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m  = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55));
    return _mm_cvtss_f32(m);
}



float hsum_avx2(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}


#endif /* __AVX2__ */
