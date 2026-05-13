#include "simd_kernels.h"

#ifdef __AVX2__

#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline void transpose8x8_avx2(const float *src, int src_stride,
                                      float *dst, int dst_stride)
{
    __m256 r0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 r1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 r2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 r3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 r4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 r5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 r6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 r7 = _mm256_loadu_ps(src + 7 * src_stride);

    __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));
    __m256 u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));
    __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));
    __m256 u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));
    __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1,0,1,0));
    __m256 u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3,2,3,2));
    __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1,0,1,0));
    __m256 u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3,2,3,2));

    r0 = _mm256_permute2f128_ps(u0, u4, 0x20);
    r1 = _mm256_permute2f128_ps(u1, u5, 0x20);
    r2 = _mm256_permute2f128_ps(u2, u6, 0x20);
    r3 = _mm256_permute2f128_ps(u3, u7, 0x20);
    r4 = _mm256_permute2f128_ps(u0, u4, 0x31);
    r5 = _mm256_permute2f128_ps(u1, u5, 0x31);
    r6 = _mm256_permute2f128_ps(u2, u6, 0x31);
    r7 = _mm256_permute2f128_ps(u3, u7, 0x31);

    _mm256_storeu_ps(dst + 0 * dst_stride, r0);
    _mm256_storeu_ps(dst + 1 * dst_stride, r1);
    _mm256_storeu_ps(dst + 2 * dst_stride, r2);
    _mm256_storeu_ps(dst + 3 * dst_stride, r3);
    _mm256_storeu_ps(dst + 4 * dst_stride, r4);
    _mm256_storeu_ps(dst + 5 * dst_stride, r5);
    _mm256_storeu_ps(dst + 6 * dst_stride, r6);
    _mm256_storeu_ps(dst + 7 * dst_stride, r7);
}



static void transpose_matrix(const float *src, float *dst, int rows, int cols)
{
    int r = 0;
    for (; r + 8 <= rows; r += 8) {
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            transpose8x8_avx2(src + (size_t)r * cols + c, cols,
                              dst + (size_t)c * rows + r, rows);
        }
        /* Border columns: scalar 8 rows × (cols - c) */
        for (; c < cols; c++) {
            for (int rr = 0; rr < 8; rr++)
                dst[(size_t)c * rows + r + rr] = src[(size_t)(r + rr) * cols + c];
        }
    }
    /* Border rows */
    for (; r < rows; r++) {
        for (int c = 0; c < cols; c++)
            dst[(size_t)c * rows + r] = src[(size_t)r * cols + c];
    }
}



int kernel_transpose_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    (void)scratch;
    Spkv2AttrRecord attr;
    if (simd_get_attr(ctx, node, &attr) != 0) return -10;
    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;

    int rank = (int)x_rec->rank;
    if (rank < 1 || rank > 8) return -10;

    /* Resolve perm: default reverse if extra_count == 0 */
    int perm[8];
    for (int i = 0; i < rank; i++) {
        perm[i] = (attr.extra_count > 0) ? attr.extra[i] : (rank - 1 - i);
        if (perm[i] < 0 || perm[i] >= rank) return -10;
    }

    /* Identity permutation → memcpy */
    int is_identity = 1;
    for (int i = 0; i < rank; i++)
        if (perm[i] != i) { is_identity = 0; break; }
    if (is_identity) {
        memcpy(y, x, simd_elem_count(x_rec) * sizeof(float));
        return 0;
    }

    /* Cache input shape; compute output shape via perm. */
    int xshape[8], yshape[8];
    for (int i = 0; i < rank; i++) xshape[i] = (int)x_rec->shape[i];
    for (int i = 0; i < rank; i++) yshape[i] = xshape[perm[i]];

    /* ---- Fast path A: last two axes swapped, all earlier axes unchanged ---- */
    /* perm = [0, 1, ..., rank-3, rank-1, rank-2] */
    if (rank >= 2) {
        int last_two_swap = (perm[rank - 1] == rank - 2 && perm[rank - 2] == rank - 1);
        for (int i = 0; i < rank - 2 && last_two_swap; i++)
            if (perm[i] != i) last_two_swap = 0;
        if (last_two_swap) {
            int rows = xshape[rank - 2];
            int cols = xshape[rank - 1];
            size_t batch = 1;
            for (int i = 0; i < rank - 2; i++) batch *= (size_t)xshape[i];
            size_t plane = (size_t)rows * cols;

            /* Parallelize over (batch, row-tile-8) pairs so that even
               batch=1 large matrices use all threads. */
            int row_tiles_full = rows / 8;        /* number of 8-row strips */
            int row_tail       = rows - row_tiles_full * 8;
            long total_tiles = (long)batch * row_tiles_full;
            int do_par = (long)rows * cols * (long)batch >= 32768;

            #pragma omp parallel for if(do_par) schedule(static)
            for (long t = 0; t < total_tiles; t++) {
                size_t b = (size_t)(t / row_tiles_full);
                int    r = (int)(t % row_tiles_full) * 8;
                const float *src = x + b * plane;
                float       *dst = y + b * plane;
                int c = 0;
                for (; c + 8 <= cols; c += 8) {
                    transpose8x8_avx2(src + (size_t)r * cols + c, cols,
                                      dst + (size_t)c * rows + r, rows);
                }
                for (; c < cols; c++) {
                    for (int rr = 0; rr < 8; rr++)
                        dst[(size_t)c * rows + r + rr] =
                            src[(size_t)(r + rr) * cols + c];
                }
            }
            /* Tail rows (rows % 8) — sequential, small. */
            if (row_tail > 0) {
                for (size_t b = 0; b < batch; b++) {
                    const float *src = x + b * plane;
                    float       *dst = y + b * plane;
                    int r0 = row_tiles_full * 8;
                    for (int r = r0; r < rows; r++) {
                        for (int c = 0; c < cols; c++)
                            dst[(size_t)c * rows + r] =
                                src[(size_t)r * cols + c];
                    }
                }
            }
            return 0;
        }
    }

    /* ---- Fast path B: inner dim preserved (perm[rank-1] == rank-1) ---- */
    /* Output's last axis = input's last axis → inner block size is contiguous
       in both src and dst. Permute outer indices, memcpy inner blocks. */
    if (perm[rank - 1] == rank - 1 && rank >= 2) {
        int outer = rank - 1;
        size_t inner = (size_t)xshape[rank - 1];

        /* Compute input strides for each outer axis (in elements). */
        size_t x_outer_stride[8];
        x_outer_stride[outer - 1] = inner;
        for (int i = outer - 2; i >= 0; i--)
            x_outer_stride[i] = x_outer_stride[i + 1] * (size_t)xshape[i + 1];

        /* Output outer shape = yshape[0..outer-1]; output stride is the
           product of inner * (yshape[outer-1] * ... * yshape[i+1]). */
        size_t y_outer_dims[8];
        for (int i = 0; i < outer; i++) y_outer_dims[i] = (size_t)yshape[i];
        size_t total_outer = 1;
        for (int i = 0; i < outer; i++) total_outer *= y_outer_dims[i];

        /* For each linear output outer index, map to input outer coords. */
        #pragma omp parallel for if(total_outer >= 64) schedule(static)
        for (size_t lin = 0; lin < total_outer; lin++) {
            /* Decode lin to output coords. */
            size_t y_coords[8];
            size_t rem = lin;
            for (int i = outer - 1; i >= 0; i--) {
                y_coords[i] = rem % y_outer_dims[i];
                rem /= y_outer_dims[i];
            }
            /* Map to input coords via perm. */
            size_t src_off = 0;
            for (int i = 0; i < outer; i++) {
                src_off += y_coords[i] * x_outer_stride[perm[i]];
            }
            memcpy(y + lin * inner, x + src_off, inner * sizeof(float));
        }
        return 0;
    }

    /* ---- General fallback: precomputed strides, no div/mod per element ---- */
    size_t x_stride[8];
    x_stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--)
        x_stride[i] = x_stride[i + 1] * (size_t)xshape[i + 1];

    size_t total = 1;
    for (int i = 0; i < rank; i++) total *= (size_t)yshape[i];

    #pragma omp parallel for if(total >= 65536) schedule(static)
    for (size_t lin = 0; lin < total; lin++) {
        size_t y_coords[8];
        size_t rem = lin;
        for (int i = rank - 1; i >= 0; i--) {
            y_coords[i] = rem % (size_t)yshape[i];
            rem /= (size_t)yshape[i];
        }
        size_t src_off = 0;
        for (int i = 0; i < rank; i++)
            src_off += y_coords[i] * x_stride[perm[i]];
        y[lin] = x[src_off];
    }
    return 0;
}


#endif /* __AVX2__ */
