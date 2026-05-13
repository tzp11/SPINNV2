#include "simd_kernels.h"

#ifdef __AVX2__

#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static int simd_reduce_get_axes(const Spkv2Context *ctx,
                                 const Spkv2NodeRecord *node,
                                 const Spkv2AttrRecord *attr,
                                 int rank, int *out_axes, int *out_count)
{
    *out_count = 0;
    if (attr->extra_count > 0) {
        for (int i = 0; i < attr->extra_count && i < 8; i++) {
            int a = attr->extra[i];
            if (a < 0) a += rank;
            if (a < 0 || a >= rank) return -10;
            out_axes[(*out_count)++] = a;
        }
        return 0;
    }
    if (node->input_count >= 2) {
        const Spkv2TensorRecord *axes_rec = ctx->tensors[node->inputs[1]].record;
        const float *axes_data = (const float *)ctx->tensors[node->inputs[1]].data;
        if (axes_rec && axes_data) {
            size_t n = 1;
            for (uint16_t i = 0; i < axes_rec->rank; i++) n *= axes_rec->shape[i];
            for (size_t i = 0; i < n && i < 8; i++) {
                int a = (int)axes_data[i];
                if (a < 0) a += rank;
                if (a < 0 || a >= rank) return -10;
                out_axes[(*out_count)++] = a;
            }
        }
    }
    return 0;
}



int kernel_reduce_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    Spkv2AttrRecord attr;
    if (simd_get_attr(ctx, node, &attr) != 0) return -10;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int rank = (int)x_rec->rank;
    int axes[8];
    int axis_count = 0;
    if (simd_reduce_get_axes(ctx, node, &attr, rank, axes, &axis_count) != 0)
        return -10;

    /* Fall back to reference when no axes (full reduction) or multi-axis cases */
    if (axis_count != 1) return -99;

    int axis = axes[0];

    /* Compute outer / inner sizes */
    size_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= (size_t)x_rec->shape[i];
    for (int i = axis + 1; i < rank; i++) inner *= (size_t)x_rec->shape[i];
    size_t dim = (size_t)x_rec->shape[axis];
    (void)y_rec;

    int is_max = (node->op_type == SPKV2_OP_REDUCEMAX);

    if (inner == 1) {
        /* ---- Fast path: reduce the last axis (contiguous along dim) ---- */
        #pragma omp parallel for if(outer >= 64) schedule(static)
        for (size_t o = 0; o < outer; o++) {
            const float *row = x + o * dim;
            if (is_max) {
                size_t j = 0;
                __m256 vmax = _mm256_set1_ps(-INFINITY);
                for (; j + 8 <= dim; j += 8)
                    vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(row + j));
                float m = (dim >= 8) ? hmax_avx2(vmax) : -INFINITY;
                for (; j < dim; j++)
                    if (row[j] > m) m = row[j];
                y[o] = m;
            } else {
                /* ReduceMean */
                size_t j = 0;
                __m256 vsum = _mm256_setzero_ps();
                for (; j + 8 <= dim; j += 8)
                    vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(row + j));
                float s = (dim >= 8) ? hsum_avx2(vsum) : 0.0f;
                for (; j < dim; j++) s += row[j];
                y[o] = s / (float)dim;
            }
        }
        return 0;
    }

    if (outer == 1) {
        /* ---- Fast path: reduce the first axis (stride = inner) ---- */
        #pragma omp parallel for if(inner >= 64) schedule(static)
        for (size_t i = 0; i < inner; i++) {
            if (is_max) {
                float m = x[i];
                for (size_t d = 1; d < dim; d++) {
                    float v = x[d * inner + i];
                    if (v > m) m = v;
                }
                y[i] = m;
            } else {
                float s = 0.0f;
                for (size_t d = 0; d < dim; d++)
                    s += x[d * inner + i];
                y[i] = s / (float)dim;
            }
        }
        return 0;
    }

    /* ---- Middle-axis reduction: vectorize across the inner dim ---- */
    #pragma omp parallel for if(outer >= 16) schedule(static)
    for (size_t o = 0; o < outer; o++) {
        float *yo = y + o * inner;
        const float *xo = x + o * dim * inner;
        if (is_max) {
            /* init with d=0 row */
            size_t i = 0;
            for (; i + 8 <= inner; i += 8)
                _mm256_storeu_ps(yo + i, _mm256_loadu_ps(xo + i));
            for (; i < inner; i++) yo[i] = xo[i];
            for (size_t d = 1; d < dim; d++) {
                const float *xd = xo + d * inner;
                i = 0;
                for (; i + 8 <= inner; i += 8)
                    _mm256_storeu_ps(yo + i,
                        _mm256_max_ps(_mm256_loadu_ps(yo + i), _mm256_loadu_ps(xd + i)));
                for (; i < inner; i++)
                    if (xd[i] > yo[i]) yo[i] = xd[i];
            }
        } else {
            size_t i = 0;
            for (; i + 8 <= inner; i += 8)
                _mm256_storeu_ps(yo + i, _mm256_setzero_ps());
            for (; i < inner; i++) yo[i] = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                const float *xd = xo + d * inner;
                i = 0;
                for (; i + 8 <= inner; i += 8)
                    _mm256_storeu_ps(yo + i,
                        _mm256_add_ps(_mm256_loadu_ps(yo + i), _mm256_loadu_ps(xd + i)));
                for (; i < inner; i++) yo[i] += xd[i];
            }
            __m256 vinv = _mm256_set1_ps(1.0f / (float)dim);
            i = 0;
            for (; i + 8 <= inner; i += 8)
                _mm256_storeu_ps(yo + i, _mm256_mul_ps(_mm256_loadu_ps(yo + i), vinv));
            for (; i < inner; i++) yo[i] *= 1.0f / (float)dim;
        }
    }
    return 0;
}


#endif /* __AVX2__ */
