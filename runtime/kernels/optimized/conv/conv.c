#include "simd_kernels.h"
#include "simd_sgemm.h"

#if defined(__AVX2__)

#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static void depthwise_conv_s1d1(const float *x_ch, const float *w_ch, float bv,
                                 float *y_ch, int H, int W,
                                 int kH, int kW, int outH, int outW,
                                 int pH, int pW)
{
    for (int oh = 0; oh < outH; oh++) {
        int ow = 0;

        /* Determine safe interior range where all kernel taps are valid */
        int ow_start = SIMD_MIN(pW, outW);
        int ow_end   = (W + pW >= kW) ? SIMD_MIN(W + pW - kW + 1, outW) : 0;
        if (ow_end < ow_start) ow_end = ow_start;

        /* Left border (scalar) */
        for (; ow < ow_start; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh + kh - pH;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow + kw - pW;
                    if (iw >= 0 && iw < W)
                        acc += w_ch[kh * kW + kw] * x_ch[ih * W + iw];
                }
            }
            y_ch[oh * outW + ow] = acc;
        }

        /* Interior: all kernel taps valid → direct AVX2 loadu, no bounds checks */
        for (; ow + 7 < ow_end; ow += 8) {
            __m256 acc = _mm256_set1_ps(bv);
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh + kh - pH;
                if (ih < 0 || ih >= H) continue;
                const float *row = x_ch + ih * W;
                for (int kw = 0; kw < kW; kw++) {
                    __m256 vw = _mm256_set1_ps(w_ch[kh * kW + kw]);
                    /* ow + kw - pW is valid for all 8 lanes */
                    acc = _mm256_fmadd_ps(vw, _mm256_loadu_ps(row + ow + kw - pW), acc);
                }
            }
            _mm256_storeu_ps(y_ch + oh * outW + ow, acc);
        }
        /* Interior scalar tail */
        for (; ow < ow_end; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh + kh - pH;
                if (ih < 0 || ih >= H) continue;
                const float *row = x_ch + ih * W;
                for (int kw = 0; kw < kW; kw++)
                    acc += w_ch[kh * kW + kw] * row[ow + kw - pW];
            }
            y_ch[oh * outW + ow] = acc;
        }

        /* Right border (scalar) */
        for (; ow < outW; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh + kh - pH;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow + kw - pW;
                    if (iw >= 0 && iw < W)
                        acc += w_ch[kh * kW + kw] * x_ch[ih * W + iw];
                }
            }
            y_ch[oh * outW + ow] = acc;
        }
    }
}



static void depthwise_conv_generic(const float *x_ch, const float *w_ch, float bv,
                                    float *y_ch, int H, int W,
                                    int kH, int kW, int outH, int outW,
                                    int sH, int sW, int pH, int pW,
                                    int dH, int dW)
{
    /* Safe interior range: all kernel taps in bounds */
    int oh_start = pH > 0 ? (pH + sH - 1) / sH : 0;
    int oh_end   = H >= (kH - 1) * dH + 1 ? (H + pH - (kH - 1) * dH) / sH : 0;
    if (oh_start < 0) oh_start = 0;
    if (oh_end > outH) oh_end = outH;
    if (oh_end < oh_start) oh_end = oh_start;

    int ow_start = pW > 0 ? (pW + sW - 1) / sW : 0;
    int ow_end   = W >= (kW - 1) * dW + 1 ? (W + pW - (kW - 1) * dW) / sW : 0;
    if (ow_start < 0) ow_start = 0;
    if (ow_end > outW) ow_end = outW;
    if (ow_end < ow_start) ow_end = ow_start;

    for (int oh = 0; oh < outH; oh++) {
        int ow = 0;

        /* Border row: full scalar with bounds checks */
        if (oh < oh_start || oh >= oh_end) {
            for (; ow < outW; ow++) {
                float acc = bv;
                for (int kh = 0; kh < kH; kh++) {
                    int ih = oh * sH + kh * dH - pH;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < kW; kw++) {
                        int iw = ow * sW + kw * dW - pW;
                        if (iw >= 0 && iw < W)
                            acc += w_ch[kh * kW + kw] * x_ch[ih * W + iw];
                    }
                }
                y_ch[oh * outW + ow] = acc;
            }
            continue;
        }

        /* Left border (scalar) */
        for (; ow < ow_start; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh * sH + kh * dH - pH;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow * sW + kw * dW - pW;
                    if (iw >= 0 && iw < W)
                        acc += w_ch[kh * kW + kw] * x_ch[ih * W + iw];
                }
            }
            y_ch[oh * outW + ow] = acc;
        }

        /* Interior: no bounds checks needed */
        for (; ow < ow_end; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh * sH + kh * dH - pH;
                const float *row = x_ch + ih * W;
                for (int kw = 0; kw < kW; kw++)
                    acc += w_ch[kh * kW + kw] * row[ow * sW + kw * dW - pW];
            }
            y_ch[oh * outW + ow] = acc;
        }

        /* Right border (scalar) */
        for (; ow < outW; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh * sH + kh * dH - pH;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow * sW + kw * dW - pW;
                    if (iw >= 0 && iw < W)
                        acc += w_ch[kh * kW + kw] * x_ch[ih * W + iw];
                }
            }
            y_ch[oh * outW + ow] = acc;
        }
    }
}



static void depthwise_conv_simd(const float *x, const float *w, const float *bias,
                                float *y, int C, int H, int W,
                                int kH, int kW, int outH, int outW,
                                int sH, int sW, int pH, int pW,
                                int dH, int dW)
{
    int is_s1d1 = (sH == 1 && sW == 1 && dH == 1 && dW == 1);

    #pragma omp parallel for if(C * outH * outW > 50000) schedule(static)
    for (int c = 0; c < C; c++) {
        const float *x_ch = x + (size_t)c * H * W;
        const float *w_ch = w + (size_t)c * kH * kW;
        float *y_ch = y + (size_t)c * outH * outW;
        float bv = bias ? bias[c] : 0.0f;

        if (is_s1d1) {
            depthwise_conv_s1d1(x_ch, w_ch, bv, y_ch, H, W,
                                kH, kW, outH, outW, pH, pW);
        } else {
            depthwise_conv_generic(x_ch, w_ch, bv, y_ch, H, W,
                                    kH, kW, outH, outW, sH, sW, pH, pW, dH, dW);
        }
    }
}



static void im2col_3x3_s1p1(const float *im, int C, int H, int W, float *col)
{
    const int ohw = H * W;
    int row = 0;
    for (int c = 0; c < C; c++) {
        const float *xc = im + (size_t)c * ohw;
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                float *dst = col + (size_t)row * ohw;
                int ow_start = (kw == 0) ? 1 : 0;
                int ow_end   = (kw == 2) ? (W - 1) : W;
                int copy_len = ow_end - ow_start;
                int src_offset = ow_start - 1 + kw;

                for (int oh = 0; oh < H; oh++) {
                    int ih = oh - 1 + kh;
                    float *dst_row = dst + oh * W;
                    if (ih < 0 || ih >= H) {
                        memset(dst_row, 0, (size_t)W * sizeof(float));
                    } else {
                        const float *src_row = xc + ih * W;
                        if (ow_start > 0) dst_row[0] = 0.0f;
                        if (copy_len > 0)
                            memcpy(dst_row + ow_start, src_row + src_offset,
                                   (size_t)copy_len * sizeof(float));
                        if (ow_end < W) dst_row[W - 1] = 0.0f;
                    }
                }
                row++;
            }
        }
    }
}



static void im2col_full(const float *im, int C, int H, int W,
                         int kH, int kW, int sH, int sW, int pH, int pW,
                         int dH, int dW, int outH, int outW, float *col)
{
    int ohw = outH * outW;
    int row = 0;
    for (int c = 0; c < C; c++) {
        const float *xc = im + (size_t)c * H * W;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                float *dst = col + (size_t)row * ohw;
                for (int oh = 0; oh < outH; oh++) {
                    int ih = oh * sH - pH + kh * dH;
                    if (ih < 0 || ih >= H) {
                        memset(dst + oh * outW, 0, (size_t)outW * sizeof(float));
                    } else {
                        const float *xr = xc + ih * W;
                        for (int ow = 0; ow < outW; ow++) {
                            int iw = ow * sW - pW + kw * dW;
                            dst[oh * outW + ow] = (iw >= 0 && iw < W) ? xr[iw] : 0.0f;
                        }
                    }
                }
                row++;
            }
        }
    }
}

static void im2col_segment(const float *im, int C, int H, int W,
                            int kH, int kW, int sH, int sW, int pH, int pW,
                            int dH, int dW, int outW,
                            int oh_start, int oh_end, float *col)
{
    int seg_rows = oh_end - oh_start;
    int seg_cols = seg_rows * outW;
    int row = 0;
    for (int c = 0; c < C; c++) {
        const float *xc = im + (size_t)c * H * W;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                float *dst = col + (size_t)row * seg_cols;
                for (int oh = oh_start; oh < oh_end; oh++) {
                    int local = oh - oh_start;
                    int ih = oh * sH - pH + kh * dH;
                    if (ih < 0 || ih >= H) {
                        memset(dst + local * outW, 0, (size_t)outW * sizeof(float));
                    } else {
                        const float *xr = xc + ih * W;
                        for (int ow = 0; ow < outW; ow++) {
                            int iw = ow * sW - pW + kw * dW;
                            dst[local * outW + ow] = (iw >= 0 && iw < W) ? xr[iw] : 0.0f;
                        }
                    }
                }
                row++;
            }
        }
    }
}



static void bias_init_row(float *y, float bias_val, int N)
{
    int n = 0;
    __m256 vb = _mm256_set1_ps(bias_val);
    for (; n + 7 < N; n += 8)
        _mm256_storeu_ps(y + n, vb);
    for (; n < N; n++)
        y[n] = bias_val;
}



static inline void winograd_transform_input_4x4(const float d[4][4], float v[4][4])
{
    float t[4][4];
    for (int j = 0; j < 4; j++) {
        t[0][j] = d[0][j] - d[2][j];
        t[1][j] = d[1][j] + d[2][j];
        t[2][j] = -d[1][j] + d[2][j];
        t[3][j] = d[1][j] - d[3][j];
    }
    for (int i = 0; i < 4; i++) {
        v[i][0] = t[i][0] - t[i][2];
        v[i][1] = t[i][1] + t[i][2];
        v[i][2] = -t[i][1] + t[i][2];
        v[i][3] = t[i][1] - t[i][3];
    }
}



static inline void winograd_transform_weight_3x3(const float g[3][3], float u[4][4])
{
    float t[4][3];
    for (int j = 0; j < 3; j++) {
        t[0][j] = g[0][j];
        t[1][j] = 0.5f * (g[0][j] + g[1][j] + g[2][j]);
        t[2][j] = 0.5f * (g[0][j] - g[1][j] + g[2][j]);
        t[3][j] = g[2][j];
    }
    for (int i = 0; i < 4; i++) {
        u[i][0] = t[i][0];
        u[i][1] = 0.5f * (t[i][0] + t[i][1] + t[i][2]);
        u[i][2] = 0.5f * (t[i][0] - t[i][1] + t[i][2]);
        u[i][3] = t[i][2];
    }
}



static inline void winograd_transform_output_2x2(const float m[4][4], float out[2][2])
{
    float t[2][4];
    for (int j = 0; j < 4; j++) {
        t[0][j] = m[0][j] + m[1][j] + m[2][j];
        t[1][j] = m[1][j] - m[2][j] - m[3][j];
    }
    for (int i = 0; i < 2; i++) {
        out[i][0] = t[i][0] + t[i][1] + t[i][2];
        out[i][1] = t[i][1] - t[i][2] - t[i][3];
    }
}


/* ── Winograd F(4,3): output tile 4×4, input tile 6×6 ── */

static inline void winograd_f43_transform_input_6x6(const float d[6][6], float v[6][6])
{
    float t[6][6];
    for (int j = 0; j < 6; j++) {
        t[0][j] =  4.0f*d[0][j]              - 5.0f*d[2][j]              + d[4][j];
        t[1][j] =            - 4.0f*d[1][j]  - 4.0f*d[2][j] +    d[3][j] + d[4][j];
        t[2][j] =              4.0f*d[1][j]  - 4.0f*d[2][j] -    d[3][j] + d[4][j];
        t[3][j] =            - 2.0f*d[1][j]  -      d[2][j] + 2.0f*d[3][j] + d[4][j];
        t[4][j] =              2.0f*d[1][j]  -      d[2][j] - 2.0f*d[3][j] + d[4][j];
        t[5][j] =              4.0f*d[1][j]              - 5.0f*d[3][j]              + d[5][j];
    }
    for (int i = 0; i < 6; i++) {
        v[i][0] =  4.0f*t[i][0]              - 5.0f*t[i][2]              + t[i][4];
        v[i][1] =            - 4.0f*t[i][1]  - 4.0f*t[i][2] +    t[i][3] + t[i][4];
        v[i][2] =              4.0f*t[i][1]  - 4.0f*t[i][2] -    t[i][3] + t[i][4];
        v[i][3] =            - 2.0f*t[i][1]  -      t[i][2] + 2.0f*t[i][3] + t[i][4];
        v[i][4] =              2.0f*t[i][1]  -      t[i][2] - 2.0f*t[i][3] + t[i][4];
        v[i][5] =              4.0f*t[i][1]              - 5.0f*t[i][3]              + t[i][5];
    }
}

static inline void winograd_f43_transform_weight_3x3(const float g[3][3], float u[6][6])
{
    const float c14  = 0.25f;
    const float c16  = 1.0f / 6.0f;
    const float c112 = 1.0f / 12.0f;
    const float c124 = 1.0f / 24.0f;
    float t[6][3];
    for (int j = 0; j < 3; j++) {
        t[0][j] =  c14  * g[0][j];
        t[1][j] = -c16  * (g[0][j] + g[1][j] + g[2][j]);
        t[2][j] = -c16  * (g[0][j] - g[1][j] + g[2][j]);
        t[3][j] =  c124 * g[0][j] + c112 * g[1][j] + c16 * g[2][j];
        t[4][j] =  c124 * g[0][j] - c112 * g[1][j] + c16 * g[2][j];
        t[5][j] = g[2][j];
    }
    for (int i = 0; i < 6; i++) {
        u[i][0] =  c14  * t[i][0];
        u[i][1] = -c16  * (t[i][0] + t[i][1] + t[i][2]);
        u[i][2] = -c16  * (t[i][0] - t[i][1] + t[i][2]);
        u[i][3] =  c124 * t[i][0] + c112 * t[i][1] + c16 * t[i][2];
        u[i][4] =  c124 * t[i][0] - c112 * t[i][1] + c16 * t[i][2];
        u[i][5] = t[i][2];
    }
}

static inline void winograd_f43_transform_output_4x4(const float m[6][6], float out[4][4])
{
    float t[4][6];
    for (int j = 0; j < 6; j++) {
        t[0][j] = m[0][j] + m[1][j] + m[2][j] +      m[3][j] +      m[4][j];
        t[1][j] =           m[1][j] - m[2][j] + 2.0f*m[3][j] - 2.0f*m[4][j];
        t[2][j] =           m[1][j] + m[2][j] + 4.0f*m[3][j] + 4.0f*m[4][j];
        t[3][j] =           m[1][j] - m[2][j] + 8.0f*m[3][j] - 8.0f*m[4][j] + m[5][j];
    }
    for (int i = 0; i < 4; i++) {
        out[i][0] = t[i][0] + t[i][1] + t[i][2] +      t[i][3] +      t[i][4];
        out[i][1] =           t[i][1] - t[i][2] + 2.0f*t[i][3] - 2.0f*t[i][4];
        out[i][2] =           t[i][1] + t[i][2] + 4.0f*t[i][3] + 4.0f*t[i][4];
        out[i][3] =           t[i][1] - t[i][2] + 8.0f*t[i][3] - 8.0f*t[i][4] + t[i][5];
    }
}



static float *winograd_pack_weights_3x3(int OC, int C, const float *w)
{
    int num_m_blocks = (OC + SGEMM_MR - 1) / SGEMM_MR;
    size_t one_alpha = (size_t)num_m_blocks * C * SGEMM_MR;
    float *packed = NULL;
    if (posix_memalign((void **)&packed, 32, 16 * one_alpha * sizeof(float)) != 0)
        return NULL;
    memset(packed, 0, 16 * one_alpha * sizeof(float));

    for (int oc = 0; oc < OC; oc++) {
        int mi = oc / SGEMM_MR;
        int mr = oc % SGEMM_MR;
        for (int ic = 0; ic < C; ic++) {
            const float *gptr = w + ((size_t)oc * C + ic) * 9;
            float g[3][3], u[4][4];
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    g[kh][kw] = gptr[kh * 3 + kw];
            winograd_transform_weight_3x3(g, u);

            for (int a = 0; a < 16; a++) {
                packed[(size_t)a * one_alpha + (size_t)mi * C * SGEMM_MR +
                       (size_t)ic * SGEMM_MR + mr] = u[a / 4][a % 4];
            }
        }
    }
    return packed;
}



static void winograd_gemm_small_tiles(int OC, int tiles, int C,
                                      const float *packed_u,
                                      size_t one_alpha_packed,
                                      const float *V,
                                      float *M)
{
    #pragma omp parallel for schedule(static) if((long long)OC * tiles * C >= 200000)
    for (int oc = 0; oc < OC; oc++) {
        int mi = oc / SGEMM_MR;
        int mr = oc % SGEMM_MR;
        for (int a = 0; a < 16; a++) {
            const float *u_base = packed_u + (size_t)a * one_alpha_packed +
                                  (size_t)mi * C * SGEMM_MR + mr;
            const float *v_base = V + (size_t)a * C * tiles;
            float *m_base = M + (size_t)a * OC * tiles + (size_t)oc * tiles;
            int t = 0;
            for (; t + 7 < tiles; t += 8) {
                __m256 acc = _mm256_setzero_ps();
                for (int c = 0; c < C; c++) {
                    __m256 vv = _mm256_loadu_ps(v_base + (size_t)c * tiles + t);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(u_base[(size_t)c * SGEMM_MR]),
                                          vv, acc);
                }
                _mm256_storeu_ps(m_base + t, acc);
            }
            for (; t < tiles; t++) {
                float acc = 0.0f;
                for (int c = 0; c < C; c++) {
                    acc += u_base[(size_t)c * SGEMM_MR] * v_base[(size_t)c * tiles + t];
                }
                m_base[t] = acc;
            }
        }
    }
}



static int winograd_conv3x3s1p1(Spkv2Context *ctx,
                                const Spkv2NodeRecord *node,
                                const float *x, const float *w,
                                const float *bias, float *y,
                                int N_batch, int C, int H, int W,
                                int OC, int OH, int OW,
                                int act_type, void *scratch)
{
    if (!scratch) return -13;
    int tile_h = (OH + 1) / 2;
    int tile_w = (OW + 1) / 2;
    int tiles = tile_h * tile_w;
    if (tiles <= 0) return -99;

    size_t v_size = (size_t)16 * C * tiles;
    size_t m_size = (size_t)16 * OC * tiles;
    float *V = (float *)scratch;
    float *M = V + v_size;

    if (!ctx->node_cache || node->id >= ctx->node_cache_count) return -99;
    if (!ctx->node_cache[node->id]) {
        ctx->node_cache[node->id] = winograd_pack_weights_3x3(OC, C, w);
        if (!ctx->node_cache[node->id]) return -1;
    }
    const float *packed_u = (const float *)ctx->node_cache[node->id];
    int num_m_blocks = (OC + SGEMM_MR - 1) / SGEMM_MR;
    size_t one_alpha_packed = (size_t)num_m_blocks * C * SGEMM_MR;

    for (int n = 0; n < N_batch; n++) {
        const float *xn = x + (size_t)n * C * H * W;
        float *yn = y + (size_t)n * OC * OH * OW;
        memset(V, 0, v_size * sizeof(float));
        memset(M, 0, m_size * sizeof(float));

        #pragma omp parallel for schedule(static) if((long long)C * tiles >= 512)
        for (int ic = 0; ic < C; ic++) {
            const float *xc = xn + (size_t)ic * H * W;
            for (int th = 0; th < tile_h; th++) {
                int oh0 = th * 2;
                for (int tw = 0; tw < tile_w; tw++) {
                    int ow0 = tw * 2;
                    int tile = th * tile_w + tw;
                    float d[4][4], vt[4][4];
                    for (int i = 0; i < 4; i++) {
                        int ih = oh0 + i - 1;
                        for (int j = 0; j < 4; j++) {
                            int iw = ow0 + j - 1;
                            d[i][j] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                          ? xc[(size_t)ih * W + iw]
                                          : 0.0f;
                        }
                    }
                    winograd_transform_input_4x4(d, vt);
                    for (int a = 0; a < 16; a++) {
                        V[(size_t)a * C * tiles + (size_t)ic * tiles + tile] =
                            vt[a / 4][a % 4];
                    }
                }
            }
        }

        if (tiles <= 256) {
            winograd_gemm_small_tiles(OC, tiles, C, packed_u,
                                      one_alpha_packed, V, M);
        } else {
            int allow_gemm_parallel = tiles >= 256;
            for (int a = 0; a < 16; a++) {
                sgemm_nn_packed_a_impl_run(OC, tiles, C,
                                           packed_u + (size_t)a * one_alpha_packed,
                                           V + (size_t)a * C * tiles, tiles,
                                           M + (size_t)a * OC * tiles, tiles,
                                           allow_gemm_parallel);
            }
        }

        #pragma omp parallel for schedule(static) if((long long)OC * tiles >= 512)
        for (int oc = 0; oc < OC; oc++) {
            float *yoc = yn + (size_t)oc * OH * OW;
            float bv = bias ? bias[oc] : 0.0f;
            for (int th = 0; th < tile_h; th++) {
                int oh0 = th * 2;
                for (int tw = 0; tw < tile_w; tw++) {
                    int ow0 = tw * 2;
                    int tile = th * tile_w + tw;
                    float mt[4][4], out[2][2];
                    for (int a = 0; a < 16; a++) {
                        mt[a / 4][a % 4] =
                            M[(size_t)a * OC * tiles + (size_t)oc * tiles + tile];
                    }
                    winograd_transform_output_2x2(mt, out);
                    for (int i = 0; i < 2 && oh0 + i < OH; i++) {
                        for (int j = 0; j < 2 && ow0 + j < OW; j++) {
                            float value = out[i][j] + bv;
                            yoc[(size_t)(oh0 + i) * OW + ow0 + j] =
                                apply_activation_scalar_simd(value, act_type);
                        }
                    }
                }
            }
        }
    }
    return 0;
}



/* ── Winograd F(4,3) weight packing: 36 alpha-matrices ── */

static float *winograd_f43_pack_weights_3x3(int OC, int C, const float *w)
{
    int num_m_blocks = (OC + SGEMM_MR - 1) / SGEMM_MR;
    size_t one_alpha = (size_t)num_m_blocks * C * SGEMM_MR;
    float *packed = NULL;
    if (posix_memalign((void **)&packed, 32, 36 * one_alpha * sizeof(float)) != 0)
        return NULL;
    memset(packed, 0, 36 * one_alpha * sizeof(float));

    for (int oc = 0; oc < OC; oc++) {
        int mi = oc / SGEMM_MR;
        int mr = oc % SGEMM_MR;
        for (int ic = 0; ic < C; ic++) {
            const float *gptr = w + ((size_t)oc * C + ic) * 9;
            float g[3][3], u[6][6];
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    g[kh][kw] = gptr[kh * 3 + kw];
            winograd_f43_transform_weight_3x3(g, u);

            for (int a = 0; a < 36; a++) {
                packed[(size_t)a * one_alpha + (size_t)mi * C * SGEMM_MR +
                       (size_t)ic * SGEMM_MR + mr] = u[a / 6][a % 6];
            }
        }
    }
    return packed;
}


static void winograd_f43_gemm_small_tiles(int OC, int tiles, int C,
                                          const float *packed_u,
                                          size_t one_alpha_packed,
                                          const float *V,
                                          float *M)
{
    #pragma omp parallel for schedule(static) if((long long)OC * tiles * C >= 200000)
    for (int oc = 0; oc < OC; oc++) {
        int mi = oc / SGEMM_MR;
        int mr = oc % SGEMM_MR;
        for (int a = 0; a < 36; a++) {
            const float *u_base = packed_u + (size_t)a * one_alpha_packed +
                                  (size_t)mi * C * SGEMM_MR + mr;
            const float *v_base = V + (size_t)a * C * tiles;
            float *m_base = M + (size_t)a * OC * tiles + (size_t)oc * tiles;
            int t = 0;
#if defined(__AVX2__)
            for (; t + 7 < tiles; t += 8) {
                __m256 acc = _mm256_setzero_ps();
                for (int c = 0; c < C; c++) {
                    __m256 bv = _mm256_loadu_ps(v_base + (size_t)c * tiles + t);
                    __m256 av = _mm256_set1_ps(u_base[(size_t)c * SGEMM_MR]);
                    acc = _mm256_fmadd_ps(av, bv, acc);
                }
                _mm256_storeu_ps(m_base + t, acc);
            }
#elif defined(__ARM_NEON)
            for (; t + 3 < tiles; t += 4) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int c = 0; c < C; c++) {
                    float32x4_t bv = vld1q_f32(v_base + (size_t)c * tiles + t);
                    acc = vfmaq_n_f32(acc, bv, u_base[(size_t)c * SGEMM_MR]);
                }
                vst1q_f32(m_base + t, acc);
            }
#endif
            for (; t < tiles; t++) {
                float acc = 0.0f;
                for (int c = 0; c < C; c++) {
                    acc += u_base[(size_t)c * SGEMM_MR] * v_base[(size_t)c * tiles + t];
                }
                m_base[t] = acc;
            }
        }
    }
}


static int winograd_f43_conv3x3s1p1(Spkv2Context *ctx,
                                     const Spkv2NodeRecord *node,
                                     const float *x, const float *w,
                                     const float *bias, float *y,
                                     int N_batch, int C, int H, int W,
                                     int OC, int OH, int OW,
                                     int act_type, const float *residual,
                                     void *scratch)
{
    if (!scratch) return -13;
    int tile_h = (OH + 3) / 4;
    int tile_w = (OW + 3) / 4;
    int tiles = tile_h * tile_w;
    if (tiles <= 0) return -99;

    size_t v_size = (size_t)36 * C * tiles;
    size_t m_size = (size_t)36 * OC * tiles;
    float *V = (float *)scratch;
    float *M = V + v_size;

    if (!ctx->node_cache || node->id >= ctx->node_cache_count) return -99;
    if (!ctx->node_cache[node->id]) {
        ctx->node_cache[node->id] = winograd_f43_pack_weights_3x3(OC, C, w);
        if (!ctx->node_cache[node->id]) return -1;
    }
    const float *packed_u = (const float *)ctx->node_cache[node->id];
    int num_m_blocks = (OC + SGEMM_MR - 1) / SGEMM_MR;
    size_t one_alpha_packed = (size_t)num_m_blocks * C * SGEMM_MR;

    for (int n = 0; n < N_batch; n++) {
        const float *xn = x + (size_t)n * C * H * W;
        float *yn = y + (size_t)n * OC * OH * OW;
        memset(V, 0, v_size * sizeof(float));
        memset(M, 0, m_size * sizeof(float));

        #pragma omp parallel for schedule(static) if((long long)C * tiles >= 512)
        for (int ic = 0; ic < C; ic++) {
            const float *xc = xn + (size_t)ic * H * W;
            for (int th = 0; th < tile_h; th++) {
                int oh0 = th * 4;
                for (int tw = 0; tw < tile_w; tw++) {
                    int ow0 = tw * 4;
                    int tile = th * tile_w + tw;
                    float d[6][6], vt[6][6];
                    for (int i = 0; i < 6; i++) {
                        int ih = oh0 + i - 1;
                        for (int j = 0; j < 6; j++) {
                            int iw = ow0 + j - 1;
                            d[i][j] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                          ? xc[(size_t)ih * W + iw]
                                          : 0.0f;
                        }
                    }
                    winograd_f43_transform_input_6x6(d, vt);
                    for (int a = 0; a < 36; a++) {
                        V[(size_t)a * C * tiles + (size_t)ic * tiles + tile] =
                            vt[a / 6][a % 6];
                    }
                }
            }
        }

        if (tiles <= 256) {
            winograd_f43_gemm_small_tiles(OC, tiles, C, packed_u,
                                          one_alpha_packed, V, M);
        } else {
            int allow_gemm_parallel = tiles >= 256;
            for (int a = 0; a < 36; a++) {
                sgemm_nn_packed_a_impl_run(OC, tiles, C,
                                           packed_u + (size_t)a * one_alpha_packed,
                                           V + (size_t)a * C * tiles, tiles,
                                           M + (size_t)a * OC * tiles, tiles,
                                           allow_gemm_parallel);
            }
        }

        #pragma omp parallel for schedule(static) if((long long)OC * tiles >= 512)
        for (int oc = 0; oc < OC; oc++) {
            float *yoc = yn + (size_t)oc * OH * OW;
            float bv = bias ? bias[oc] : 0.0f;
            for (int th = 0; th < tile_h; th++) {
                int oh0 = th * 4;
                for (int tw = 0; tw < tile_w; tw++) {
                    int ow0 = tw * 4;
                    int tile = th * tile_w + tw;
                    float mt[6][6], out[4][4];
                    for (int a = 0; a < 36; a++) {
                        mt[a / 6][a % 6] =
                            M[(size_t)a * OC * tiles + (size_t)oc * tiles + tile];
                    }
                    winograd_f43_transform_output_4x4(mt, out);
                    for (int i = 0; i < 4 && oh0 + i < OH; i++) {
                        for (int j = 0; j < 4 && ow0 + j < OW; j++) {
                            float value = out[i][j] + bv;
                            if (residual)
                                value += residual[(size_t)n * OC * OH * OW +
                                                  (size_t)oc * OH * OW +
                                                  (size_t)(oh0 + i) * OW + ow0 + j];
                            yoc[(size_t)(oh0 + i) * OW + ow0 + j] =
                                apply_activation_scalar_simd(value, act_type);
                        }
                    }
                }
            }
        }
    }
    return 0;
}



static int conv3x3_direct_avx2(const float *x, const float *w, const float *bias,
                               float *y, int N_batch, int C_in, int H, int W,
                               int C_out, int outH, int outW, int stride,
                               int act_type)
{
    if (stride != 1 && stride != 2) return -99;

    int use_par = ((long long)N_batch * C_out * outH * outW * C_in > 3000000);

    const __m256i gather_idx = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);

    #pragma omp parallel for collapse(2) schedule(static) if(use_par)
    for (int n = 0; n < N_batch; n++) {
        for (int m = 0; m < C_out; m++) {
            const float *w_m = w + (size_t)m * C_in * 9;
            float *y_m = y + (size_t)n * C_out * outH * outW + (size_t)m * outH * outW;
            float bv = bias ? bias[m] : 0.0f;

            for (int oh = 0; oh < outH; oh++) {
                int ow = 0;
                int ih_base = oh * stride - 1;

                if (stride == 1) {
                    for (; ow < outW && ow < 1; ow++) {
                        float acc = bv;
                        for (int c = 0; c < C_in; c++) {
                            const float *x_c = x + (size_t)n * C_in * H * W + (size_t)c * H * W;
                            const float *w_c = w_m + (size_t)c * 9;
                            for (int kh = 0; kh < 3; kh++) {
                                int ih = ih_base + kh;
                                if (ih < 0 || ih >= H) continue;
                                for (int kw = 0; kw < 3; kw++) {
                                    int iw = ow + kw - 1;
                                    if (iw >= 0 && iw < W)
                                        acc += x_c[(size_t)ih * W + iw] * w_c[kh * 3 + kw];
                                }
                            }
                        }
                        y_m[(size_t)oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
                    }

                    int vec_end = outW - 1;
                    for (; ow + 7 < vec_end; ow += 8) {
                        __m256 acc = _mm256_set1_ps(bv);
                        for (int c = 0; c < C_in; c++) {
                            const float *x_c = x + (size_t)n * C_in * H * W + (size_t)c * H * W;
                            const float *w_c = w_m + (size_t)c * 9;
                            for (int kh = 0; kh < 3; kh++) {
                                int ih = ih_base + kh;
                                if (ih < 0 || ih >= H) continue;
                                const float *row = x_c + (size_t)ih * W;
                                acc = _mm256_fmadd_ps(_mm256_set1_ps(w_c[kh * 3 + 0]),
                                                      _mm256_loadu_ps(row + ow - 1), acc);
                                acc = _mm256_fmadd_ps(_mm256_set1_ps(w_c[kh * 3 + 1]),
                                                      _mm256_loadu_ps(row + ow), acc);
                                acc = _mm256_fmadd_ps(_mm256_set1_ps(w_c[kh * 3 + 2]),
                                                      _mm256_loadu_ps(row + ow + 1), acc);
                            }
                        }
                        _mm256_storeu_ps(y_m + (size_t)oh * outW + ow,
                                         apply_activation_avx2(acc, act_type));
                    }
                } else {
                    for (; ow < outW && ow < 1; ow++) {
                        float acc = bv;
                        for (int c = 0; c < C_in; c++) {
                            const float *x_c = x + (size_t)n * C_in * H * W + (size_t)c * H * W;
                            const float *w_c = w_m + (size_t)c * 9;
                            for (int kh = 0; kh < 3; kh++) {
                                int ih = ih_base + kh;
                                if (ih < 0 || ih >= H) continue;
                                for (int kw = 0; kw < 3; kw++) {
                                    int iw = ow * 2 + kw - 1;
                                    if (iw >= 0 && iw < W)
                                        acc += x_c[(size_t)ih * W + iw] * w_c[kh * 3 + kw];
                                }
                            }
                        }
                        y_m[(size_t)oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
                    }

                    int vec_end = (W - 2) / 2 + 1;
                    if (vec_end > outW - 1) vec_end = outW - 1;
                    for (; ow + 7 < vec_end; ow += 8) {
                        __m256 acc = _mm256_set1_ps(bv);
                        for (int c = 0; c < C_in; c++) {
                            const float *x_c = x + (size_t)n * C_in * H * W + (size_t)c * H * W;
                            const float *w_c = w_m + (size_t)c * 9;
                            for (int kh = 0; kh < 3; kh++) {
                                int ih = ih_base + kh;
                                if (ih < 0 || ih >= H) continue;
                                const float *row = x_c + (size_t)ih * W + ow * 2 - 1;
                                acc = _mm256_fmadd_ps(_mm256_set1_ps(w_c[kh * 3 + 0]),
                                                      _mm256_i32gather_ps(row, gather_idx, 4), acc);
                                acc = _mm256_fmadd_ps(_mm256_set1_ps(w_c[kh * 3 + 1]),
                                                      _mm256_i32gather_ps(row + 1, gather_idx, 4), acc);
                                acc = _mm256_fmadd_ps(_mm256_set1_ps(w_c[kh * 3 + 2]),
                                                      _mm256_i32gather_ps(row + 2, gather_idx, 4), acc);
                            }
                        }
                        _mm256_storeu_ps(y_m + (size_t)oh * outW + ow,
                                         apply_activation_avx2(acc, act_type));
                    }
                }

                for (; ow < outW; ow++) {
                    float acc = bv;
                    for (int c = 0; c < C_in; c++) {
                        const float *x_c = x + (size_t)n * C_in * H * W + (size_t)c * H * W;
                        const float *w_c = w_m + (size_t)c * 9;
                        for (int kh = 0; kh < 3; kh++) {
                            int ih = ih_base + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < 3; kw++) {
                                int iw = ow * stride + kw - 1;
                                if (iw >= 0 && iw < W)
                                    acc += x_c[(size_t)ih * W + iw] * w_c[kh * 3 + kw];
                            }
                        }
                    }
                    y_m[(size_t)oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
                }
            }
        }
    }
    return 0;
}



int kernel_conv_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    Spkv2AttrRecord attr;
    int rc = simd_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    if (!scratch && node->scratch_bytes > 0) return -13;

    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *w_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *w = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *bias = node->input_count > 2
                            ? (const float *)ctx->tensors[node->inputs[2]].data
                            : NULL;
    const float *residual = node->input_count >= 4
                            ? (const float *)ctx->tensors[node->inputs[3]].data
                            : NULL;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    float *col = (float *)scratch;

    int N_batch = (int)x_rec->shape[0];
    int C_in  = (int)x_rec->shape[1];
    int H  = (int)x_rec->shape[2];
    int W  = (int)x_rec->shape[3];
    int C_out = (int)w_rec->shape[0];
    int kH = (int)w_rec->shape[2];
    int kW = (int)w_rec->shape[3];
    int outH = (int)y_rec->shape[2];
    int outW = (int)y_rec->shape[3];
    int spatial = outH * outW;
    int group = attr.group;
    const Spkv2KernelSpecRecord *spec = simd_node_spec(ctx, node);
    uint16_t kernel_kind = spec ? spec->kernel_kind : SPKV2_KERNEL_IM2COL_GEMM;

    /* FP16 weight promotion: convert to FP32 once and cache in node_cache */
    if (w_rec->dtype == SPKV2_DTYPE_FP16 && ctx->node_cache
        && node->id < ctx->node_cache_count) {
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
        if (!ctx->node_cache[node->id]) {
            size_t w_elems = 1;
            for (int d = 0; d < w_rec->rank; d++) w_elems *= w_rec->shape[d];
            float *w32 = (float *)malloc(w_elems * sizeof(float));
            if (!w32) return -1;
            const _Float16 *w16 = (const _Float16 *)ctx->tensors[node->inputs[1]].data;
            for (size_t i = 0; i < w_elems; i++) w32[i] = (float)w16[i];
            ctx->node_cache[node->id] = w32;
        }
        w = (const float *)ctx->node_cache[node->id];
#else
        return -99;
#endif
    }

    /* ── Depthwise conv: group == C_in == C_out ── */
    if (group == C_in && group == C_out) {
        for (int n = 0; n < N_batch; n++) {
            depthwise_conv_simd(x + (size_t)n * C_in * H * W,
                                w, bias,
                                y + (size_t)n * C_out * spatial,
                                C_in, H, W, kH, kW, outH, outW,
                                attr.strides[0], attr.strides[1],
                                attr.pads[0], attr.pads[1],
                                attr.dilations[0], attr.dilations[1]);
            if (residual) {
                float *y_n = y + (size_t)n * C_out * spatial;
                const float *r_n = residual + (size_t)n * C_out * spatial;
#if defined(__APPLE__)
                vDSP_vadd(y_n, 1, r_n, 1, y_n, 1, (vDSP_Length)(C_out * spatial));
#else
                for (size_t i = 0; i < (size_t)C_out * spatial; i++)
                    y_n[i] += r_n[i];
#endif
            }
            fused_activation_pass(y + (size_t)n * C_out * spatial,
                                  (size_t)C_out * spatial, attr.fused_activation);
        }
        return 0;
    }

    /* ── Grouped / standard conv: im2col + SGEMM per group ── */
    int C_in_g  = C_in / group;
    int C_out_g = C_out / group;
    int K_g     = C_in_g * kH * kW;  /* im2col row count per group */

    int is_1x1_s1_p0 = (kH == 1 && kW == 1 &&
                         attr.strides[0] == 1 && attr.strides[1] == 1 &&
                         attr.pads[0] == 0 && attr.pads[1] == 0);

    int is_3x3_s1_p1_d1 = (kH == 3 && kW == 3 &&
                             attr.strides[0] == 1 && attr.strides[1] == 1 &&
                             attr.pads[0] == 1 && attr.pads[1] == 1 &&
                             attr.dilations[0] == 1 && attr.dilations[1] == 1 &&
                             outH == H && outW == W);

    if (kernel_kind == SPKV2_KERNEL_WINOGRAD_3X3S1) {
        if (group != 1 || !is_3x3_s1_p1_d1) {
            return -99;
        }
        return winograd_conv3x3s1p1(ctx, node, x, w, bias, y,
                                    N_batch, C_in, H, W, C_out, outH, outW,
                                    attr.fused_activation, scratch);
    }

    if (kernel_kind == SPKV2_KERNEL_WINOGRAD_F43) {
        if (group != 1 || !is_3x3_s1_p1_d1) {
            return -99;
        }
        return winograd_f43_conv3x3s1p1(ctx, node, x, w, bias, y,
                                         N_batch, C_in, H, W, C_out, outH, outW,
                                         attr.fused_activation, residual, scratch);
    }

    if (kernel_kind == SPKV2_KERNEL_CONV3X3S2_DIRECT) {
        if (group != 1 || kH != 3 || kW != 3 ||
            attr.strides[0] != 2 || attr.strides[1] != 2 ||
            attr.pads[0] != 1 || attr.pads[1] != 1 ||
            attr.dilations[0] != 1 || attr.dilations[1] != 1) {
            return -99;
        }
        return conv3x3_direct_avx2(x, w, bias, y, N_batch, C_in, H, W,
                                   C_out, outH, outW, 2,
                                   attr.fused_activation);
    }

    /* ── Lazy weight pre-packing (via node_cache) ── */
    /* Skipped for FP16: node_cache already holds raw FP32 from promotion above */
    float *packed_w = NULL;
    if (group == 1 && ctx->node_cache && node->id < ctx->node_cache_count
        && w_rec->dtype != SPKV2_DTYPE_FP16) {
        if (!ctx->node_cache[node->id]) {
            ctx->node_cache[node->id] = sgemm_pack_a_impl(C_out_g, K_g, w, K_g);
        }
        packed_w = (float *)ctx->node_cache[node->id];
    }

    /* ── 1×1 conv fast path: skip im2col, SGEMM directly on input ── */
    if (is_1x1_s1_p0) {
        for (int n = 0; n < N_batch; n++) {
            float *y_n = y + (size_t)n * C_out * spatial;

            for (int g = 0; g < group; g++) {
                const float *x_g = x + (size_t)n * C_in * H * W + (size_t)g * C_in_g * H * W;
                const float *w_g = w + (size_t)g * C_out_g * K_g;
                const float *bias_g = bias ? bias + g * C_out_g : NULL;
                float *y_g = y_n + (size_t)g * C_out_g * spatial;

                /* init Y with bias */
#if defined(__APPLE__)
                {
                    int cout = C_out_g; int sp = spatial;
                    spkv2_apply((size_t)cout, ^(size_t m) {
                        float bv = bias_g ? bias_g[m] : 0.0f;
                        vDSP_vfill(&bv, y_g + m * (size_t)sp, 1, (size_t)sp);
                    });
                }
#else
                for (int m = 0; m < C_out_g; m++) {
                    float bv = bias_g ? bias_g[m] : 0.0f;
                    bias_init_row(y_g + (size_t)m * spatial, bv, spatial);
                }
#endif

                if (packed_w && g == 0)
                    sgemm_nn_packed_a(C_out_g, spatial, C_in_g,
                                       packed_w, x_g, spatial, y_g, spatial);
                else
                    sgemm_nn(C_out_g, spatial, C_in_g,
                             w_g, C_in_g, x_g, spatial, y_g, spatial);
            }

            if (residual) {
                const float *r_n = residual + (size_t)n * C_out * spatial;
#if defined(__APPLE__)
                vDSP_vadd(y_n, 1, r_n, 1, y_n, 1, (vDSP_Length)(C_out * spatial));
#else
                for (size_t i = 0; i < (size_t)C_out * spatial; i++)
                    y_n[i] += r_n[i];
#endif
            }
            fused_activation_pass(y_n, (size_t)C_out * spatial, attr.fused_activation);
        }
        return 0;
    }

    /* ── General conv: segmented im2col + SGEMM per group ── */
    int scratch_floats = (int)(node->scratch_bytes / sizeof(float));
    int seg_spatial = K_g > 0 ? scratch_floats / K_g : spatial;
    if (seg_spatial <= 0) seg_spatial = spatial;
    seg_spatial = (seg_spatial / outW) * outW;
    if (seg_spatial <= 0) seg_spatial = outW;
    if (seg_spatial > spatial) seg_spatial = spatial;
    int seg_outH = seg_spatial / outW;
    if (seg_outH <= 0) seg_outH = 1;

    for (int n = 0; n < N_batch; n++) {
        float *y_n = y + (size_t)n * C_out * spatial;

        for (int g = 0; g < group; g++) {
            const float *x_g = x + (size_t)n * C_in * H * W + (size_t)g * C_in_g * H * W;
            const float *w_g = w + (size_t)g * C_out_g * K_g;
            const float *bias_g = bias ? bias + g * C_out_g : NULL;
            float *y_g = y_n + (size_t)g * C_out_g * spatial;

            /* initialise Y with bias */
#if defined(__APPLE__)
            {
                int cout = C_out_g; int sp = spatial;
                spkv2_apply((size_t)cout, ^(size_t m) {
                    float bv = bias_g ? bias_g[m] : 0.0f;
                    vDSP_vfill(&bv, y_g + m * (size_t)sp, 1, (size_t)sp);
                });
            }
#else
            for (int m = 0; m < C_out_g; m++) {
                float bv = bias_g ? bias_g[m] : 0.0f;
                bias_init_row(y_g + (size_t)m * spatial, bv, spatial);
            }
#endif

            /* im2col + SGEMM: full-col path (parallel on Apple) when scratch fits */
            if (seg_spatial >= spatial) {
                im2col_full(x_g, C_in_g, H, W, kH, kW,
                            attr.strides[0], attr.strides[1],
                            attr.pads[0], attr.pads[1],
                            attr.dilations[0], attr.dilations[1],
                            outH, outW, col);
                if (packed_w && g == 0)
                    sgemm_nn_packed_a(C_out_g, spatial, K_g, packed_w, col, spatial, y_g, spatial);
                else
                    sgemm_nn(C_out_g, spatial, K_g, w_g, K_g, col, spatial, y_g, spatial);
            } else {
                for (int oh_start = 0; oh_start < outH; oh_start += seg_outH) {
                    int oh_end = oh_start + seg_outH;
                    if (oh_end > outH) oh_end = outH;
                    int seg_cols = (oh_end - oh_start) * outW;

                    im2col_segment(x_g, C_in_g, H, W, kH, kW,
                                   attr.strides[0], attr.strides[1],
                                   attr.pads[0], attr.pads[1],
                                   attr.dilations[0], attr.dilations[1],
                                   outW, oh_start, oh_end, col);

                    int col_offset = oh_start * outW;
                    if (packed_w && g == 0)
                        sgemm_nn_packed_a(C_out_g, seg_cols, K_g,
                                           packed_w, col, seg_cols,
                                           y_g + col_offset, spatial);
                    else
                        sgemm_nn(C_out_g, seg_cols, K_g,
                                 w_g, K_g, col, seg_cols,
                                 y_g + col_offset, spatial);
                }
            }
        }

        /* residual add + fused activation */
        if (residual) {
            const float *r_n = residual + (size_t)n * C_out * spatial;
#if defined(__APPLE__)
            vDSP_vadd(y_n, 1, r_n, 1, y_n, 1, (vDSP_Length)(C_out * spatial));
#else
            for (size_t i = 0; i < (size_t)C_out * spatial; i++)
                y_n[i] += r_n[i];
#endif
        }
        fused_activation_pass(y_n, (size_t)C_out * spatial, attr.fused_activation);
    }
    return 0;
}


#elif defined(__ARM_NEON)

#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

static void depthwise_conv_s1d1(const float *x_ch, const float *w_ch, float bv,
                                 float *y_ch, int H, int W,
                                 int kH, int kW, int outH, int outW,
                                 int pH, int pW)
{
    for (int oh = 0; oh < outH; oh++) {
        int ow = 0;

        /* Determine safe interior range where all kernel taps are valid */
        int ow_start = SIMD_MIN(pW, outW);
        int ow_end   = (W + pW >= kW) ? SIMD_MIN(W + pW - kW + 1, outW) : 0;
        if (ow_end < ow_start) ow_end = ow_start;

        /* Left border (scalar) */
        for (; ow < ow_start; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh + kh - pH;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow + kw - pW;
                    if (iw >= 0 && iw < W)
                        acc += w_ch[kh * kW + kw] * x_ch[ih * W + iw];
                }
            }
            y_ch[oh * outW + ow] = acc;
        }

        /* Interior: all kernel taps valid → NEON vectorized, no bounds checks */
        for (; ow + 3 < ow_end; ow += 4) {
            float32x4_t acc = vdupq_n_f32(bv);
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh + kh - pH;
                if (ih < 0 || ih >= H) continue;
                const float *row = x_ch + ih * W;
                for (int kw = 0; kw < kW; kw++) {
                    float32x4_t vw = vdupq_n_f32(w_ch[kh * kW + kw]);
                    acc = vfmaq_f32(acc, vw, vld1q_f32(row + ow + kw - pW));
                }
            }
            vst1q_f32(y_ch + oh * outW + ow, acc);
        }
        /* Interior scalar tail */
        for (; ow < ow_end; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh + kh - pH;
                if (ih < 0 || ih >= H) continue;
                const float *row = x_ch + ih * W;
                for (int kw = 0; kw < kW; kw++)
                    acc += w_ch[kh * kW + kw] * row[ow + kw - pW];
            }
            y_ch[oh * outW + ow] = acc;
        }

        /* Right border (scalar) */
        for (; ow < outW; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh + kh - pH;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow + kw - pW;
                    if (iw >= 0 && iw < W)
                        acc += w_ch[kh * kW + kw] * x_ch[ih * W + iw];
                }
            }
            y_ch[oh * outW + ow] = acc;
        }
    }
}



static void depthwise_conv_generic(const float *x_ch, const float *w_ch, float bv,
                                    float *y_ch, int H, int W,
                                    int kH, int kW, int outH, int outW,
                                    int sH, int sW, int pH, int pW,
                                    int dH, int dW)
{
    /* Safe interior range: all kernel taps in bounds */
    int oh_start = pH > 0 ? (pH + sH - 1) / sH : 0;
    int oh_end   = H >= (kH - 1) * dH + 1 ? (H + pH - (kH - 1) * dH) / sH : 0;
    if (oh_start < 0) oh_start = 0;
    if (oh_end > outH) oh_end = outH;
    if (oh_end < oh_start) oh_end = oh_start;

    int ow_start = pW > 0 ? (pW + sW - 1) / sW : 0;
    int ow_end   = W >= (kW - 1) * dW + 1 ? (W + pW - (kW - 1) * dW) / sW : 0;
    if (ow_start < 0) ow_start = 0;
    if (ow_end > outW) ow_end = outW;
    if (ow_end < ow_start) ow_end = ow_start;

    for (int oh = 0; oh < outH; oh++) {
        int ow = 0;

        /* Border row: full scalar with bounds checks */
        if (oh < oh_start || oh >= oh_end) {
            for (; ow < outW; ow++) {
                float acc = bv;
                for (int kh = 0; kh < kH; kh++) {
                    int ih = oh * sH + kh * dH - pH;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < kW; kw++) {
                        int iw = ow * sW + kw * dW - pW;
                        if (iw >= 0 && iw < W)
                            acc += w_ch[kh * kW + kw] * x_ch[ih * W + iw];
                    }
                }
                y_ch[oh * outW + ow] = acc;
            }
            continue;
        }

        /* Left border (scalar) */
        for (; ow < ow_start; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh * sH + kh * dH - pH;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow * sW + kw * dW - pW;
                    if (iw >= 0 && iw < W)
                        acc += w_ch[kh * kW + kw] * x_ch[ih * W + iw];
                }
            }
            y_ch[oh * outW + ow] = acc;
        }

        /* Interior: no bounds checks needed */
        for (; ow < ow_end; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh * sH + kh * dH - pH;
                const float *row = x_ch + ih * W;
                for (int kw = 0; kw < kW; kw++)
                    acc += w_ch[kh * kW + kw] * row[ow * sW + kw * dW - pW];
            }
            y_ch[oh * outW + ow] = acc;
        }

        /* Right border (scalar) */
        for (; ow < outW; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh * sH + kh * dH - pH;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow * sW + kw * dW - pW;
                    if (iw >= 0 && iw < W)
                        acc += w_ch[kh * kW + kw] * x_ch[ih * W + iw];
                }
            }
            y_ch[oh * outW + ow] = acc;
        }
    }
}



static void depthwise_conv_simd(const float *x, const float *w, const float *bias,
                                float *y, int C, int H, int W,
                                int kH, int kW, int outH, int outW,
                                int sH, int sW, int pH, int pW,
                                int dH, int dW)
{
    int is_s1d1 = (sH == 1 && sW == 1 && dH == 1 && dW == 1);

#if defined(__APPLE__)
    /* Use GCD dispatch_apply for multi-threaded depthwise conv on Apple,
       since Apple Clang does not support OpenMP. */
    if (C * outH * outW > 50000) {
        spkv2_apply((size_t)C, ^(size_t _c) {
            int c = (int)_c;
            const float *x_ch = x + (size_t)c * H * W;
            const float *w_ch = w + (size_t)c * kH * kW;
            float *y_ch = y + (size_t)c * outH * outW;
            float bv = bias ? bias[c] : 0.0f;

            if (is_s1d1) {
                depthwise_conv_s1d1(x_ch, w_ch, bv, y_ch, H, W,
                                    kH, kW, outH, outW, pH, pW);
            } else {
                depthwise_conv_generic(x_ch, w_ch, bv, y_ch, H, W,
                                        kH, kW, outH, outW, sH, sW, pH, pW, dH, dW);
            }
        });
    } else {
        for (int c = 0; c < C; c++) {
            const float *x_ch = x + (size_t)c * H * W;
            const float *w_ch = w + (size_t)c * kH * kW;
            float *y_ch = y + (size_t)c * outH * outW;
            float bv = bias ? bias[c] : 0.0f;

            if (is_s1d1) {
                depthwise_conv_s1d1(x_ch, w_ch, bv, y_ch, H, W,
                                    kH, kW, outH, outW, pH, pW);
            } else {
                depthwise_conv_generic(x_ch, w_ch, bv, y_ch, H, W,
                                        kH, kW, outH, outW, sH, sW, pH, pW, dH, dW);
            }
        }
    }
#else
    #pragma omp parallel for if(C * outH * outW > 50000) schedule(static)
    for (int c = 0; c < C; c++) {
        const float *x_ch = x + (size_t)c * H * W;
        const float *w_ch = w + (size_t)c * kH * kW;
        float *y_ch = y + (size_t)c * outH * outW;
        float bv = bias ? bias[c] : 0.0f;

        if (is_s1d1) {
            depthwise_conv_s1d1(x_ch, w_ch, bv, y_ch, H, W,
                                kH, kW, outH, outW, pH, pW);
        } else {
            depthwise_conv_generic(x_ch, w_ch, bv, y_ch, H, W,
                                    kH, kW, outH, outW, sH, sW, pH, pW, dH, dW);
        }
    }
#endif /* __APPLE__ */
}



static void im2col_3x3_s1p1(const float *im, int C, int H, int W, float *col)
{
    const int ohw = H * W;
#if defined(__APPLE__)
    int K9 = C * 9;
    spkv2_apply((size_t)K9, ^(size_t row) {
        int c  = (int)row / 9;
        int kh = ((int)row % 9) / 3;
        int kw = (int)row % 3;
        const float *xc = im + (size_t)c * ohw;
        float *dst = col + row * (size_t)ohw;
        int ow_start = (kw == 0) ? 1 : 0;
        int ow_end   = (kw == 2) ? (W - 1) : W;
        int copy_len = ow_end - ow_start;
        int src_offset = ow_start - 1 + kw;
        for (int oh = 0; oh < H; oh++) {
            int ih = oh - 1 + kh;
            float *dst_row = dst + oh * W;
            if (ih < 0 || ih >= H) {
                memset(dst_row, 0, (size_t)W * sizeof(float));
            } else {
                const float *src_row = xc + ih * W;
                if (ow_start > 0) dst_row[0] = 0.0f;
                if (copy_len > 0)
                    memcpy(dst_row + ow_start, src_row + src_offset,
                           (size_t)copy_len * sizeof(float));
                if (ow_end < W) dst_row[W - 1] = 0.0f;
            }
        }
    });
#else
    int row = 0;
    for (int c = 0; c < C; c++) {
        const float *xc = im + (size_t)c * ohw;
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                float *dst = col + (size_t)row * ohw;
                int ow_start = (kw == 0) ? 1 : 0;
                int ow_end   = (kw == 2) ? (W - 1) : W;
                int copy_len = ow_end - ow_start;
                int src_offset = ow_start - 1 + kw;
                for (int oh = 0; oh < H; oh++) {
                    int ih = oh - 1 + kh;
                    float *dst_row = dst + oh * W;
                    if (ih < 0 || ih >= H) {
                        memset(dst_row, 0, (size_t)W * sizeof(float));
                    } else {
                        const float *src_row = xc + ih * W;
                        if (ow_start > 0) dst_row[0] = 0.0f;
                        if (copy_len > 0)
                            memcpy(dst_row + ow_start, src_row + src_offset,
                                   (size_t)copy_len * sizeof(float));
                        if (ow_end < W) dst_row[W - 1] = 0.0f;
                    }
                }
                row++;
            }
        }
    }
#endif
}






static void im2col_full(const float *im, int C, int H, int W,
                         int kH, int kW, int sH, int sW, int pH, int pW,
                         int dH, int dW, int outH, int outW, float *col)
{
    int ohw = outH * outW;
    int K = C * kH * kW;
#if defined(__APPLE__)
    spkv2_apply((size_t)K, ^(size_t row) {
        int c  = (int)row / (kH * kW);
        int kh = ((int)row % (kH * kW)) / kW;
        int kw = (int)row % kW;
        const float *xc = im + (size_t)c * H * W;
        float *dst = col + row * (size_t)ohw;
        for (int oh = 0; oh < outH; oh++) {
            int ih = oh * sH - pH + kh * dH;
            if (ih < 0 || ih >= H) {
                memset(dst + oh * outW, 0, (size_t)outW * sizeof(float));
            } else {
                const float *xr = xc + ih * W;
                for (int ow = 0; ow < outW; ow++) {
                    int iw = ow * sW - pW + kw * dW;
                    dst[oh * outW + ow] = (iw >= 0 && iw < W) ? xr[iw] : 0.0f;
                }
            }
        }
    });
#else
    int row = 0;
    for (int c = 0; c < C; c++) {
        const float *xc = im + (size_t)c * H * W;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                float *dst = col + (size_t)row * ohw;
                for (int oh = 0; oh < outH; oh++) {
                    int ih = oh * sH - pH + kh * dH;
                    if (ih < 0 || ih >= H) {
                        memset(dst + oh * outW, 0, (size_t)outW * sizeof(float));
                    } else {
                        const float *xr = xc + ih * W;
                        for (int ow = 0; ow < outW; ow++) {
                            int iw = ow * sW - pW + kw * dW;
                            dst[oh * outW + ow] = (iw >= 0 && iw < W) ? xr[iw] : 0.0f;
                        }
                    }
                }
                row++;
            }
        }
    }
#endif
}

static void im2col_segment(const float *im, int C, int H, int W,
                            int kH, int kW, int sH, int sW, int pH, int pW,
                            int dH, int dW, int outW,
                            int oh_start, int oh_end, float *col)
{
    int seg_rows = oh_end - oh_start;
    int seg_cols = seg_rows * outW;
    int row = 0;
    for (int c = 0; c < C; c++) {
        const float *xc = im + (size_t)c * H * W;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                float *dst = col + (size_t)row * seg_cols;
                for (int oh = oh_start; oh < oh_end; oh++) {
                    int local = oh - oh_start;
                    int ih = oh * sH - pH + kh * dH;
                    if (ih < 0 || ih >= H) {
                        memset(dst + local * outW, 0, (size_t)outW * sizeof(float));
                    } else {
                        const float *xr = xc + ih * W;
                        for (int ow = 0; ow < outW; ow++) {
                            int iw = ow * sW - pW + kw * dW;
                            dst[local * outW + ow] = (iw >= 0 && iw < W) ? xr[iw] : 0.0f;
                        }
                    }
                }
                row++;
            }
        }
    }
}

static void im2col_3x3_s1p1_segment(const float *im, int C, int H, int W,
                                     int oh_start, int oh_end, float *col)
{
    int seg_rows = oh_end - oh_start;
    int seg_cols = seg_rows * W;
    int K9 = C * 9;
#if defined(__APPLE__)
    spkv2_apply((size_t)K9, ^(size_t row) {
        int c  = (int)row / 9;
        int kh = ((int)row % 9) / 3;
        int kw = (int)row % 3;
        const float *xc = im + (size_t)c * H * W;
        float *dst = col + row * (size_t)seg_cols;
        int ow_start = (kw == 0) ? 1 : 0;
        int ow_end   = (kw == 2) ? (W - 1) : W;
        int copy_len = ow_end - ow_start;
        int src_offset = ow_start - 1 + kw;
        for (int oh = oh_start; oh < oh_end; oh++) {
            int local = oh - oh_start;
            int ih = oh - 1 + kh;
            float *dst_row = dst + local * W;
            if (ih < 0 || ih >= H) {
                memset(dst_row, 0, (size_t)W * sizeof(float));
            } else {
                const float *src_row = xc + ih * W;
                if (ow_start > 0) dst_row[0] = 0.0f;
                if (copy_len > 0)
                    memcpy(dst_row + ow_start, src_row + src_offset,
                           (size_t)copy_len * sizeof(float));
                if (ow_end < W) dst_row[W - 1] = 0.0f;
            }
        }
    });
#else
    int row = 0;
    for (int c = 0; c < C; c++) {
        const float *xc = im + (size_t)c * H * W;
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                float *dst = col + (size_t)row * seg_cols;
                int ow_start = (kw == 0) ? 1 : 0;
                int ow_end   = (kw == 2) ? (W - 1) : W;
                int copy_len = ow_end - ow_start;
                int src_offset = ow_start - 1 + kw;
                for (int oh = oh_start; oh < oh_end; oh++) {
                    int local = oh - oh_start;
                    int ih = oh - 1 + kh;
                    float *dst_row = dst + local * W;
                    if (ih < 0 || ih >= H) {
                        memset(dst_row, 0, (size_t)W * sizeof(float));
                    } else {
                        const float *src_row = xc + ih * W;
                        if (ow_start > 0) dst_row[0] = 0.0f;
                        if (copy_len > 0)
                            memcpy(dst_row + ow_start, src_row + src_offset,
                                   (size_t)copy_len * sizeof(float));
                        if (ow_end < W) dst_row[W - 1] = 0.0f;
                    }
                }
                row++;
            }
        }
    }
#endif
}


static void bias_init_row(float *y, float bias_val, int N)
{
    int n = 0;
    float32x4_t vb = vdupq_n_f32(bias_val);
    for (; n + 3 < N; n += 4)
        vst1q_f32(y + n, vb);
    for (; n < N; n++)
        y[n] = bias_val;
}



static inline void winograd_transform_input_4x4(const float d[4][4], float v[4][4])
{
    float t[4][4];
    for (int j = 0; j < 4; j++) {
        t[0][j] = d[0][j] - d[2][j];
        t[1][j] = d[1][j] + d[2][j];
        t[2][j] = -d[1][j] + d[2][j];
        t[3][j] = d[1][j] - d[3][j];
    }
    for (int i = 0; i < 4; i++) {
        v[i][0] = t[i][0] - t[i][2];
        v[i][1] = t[i][1] + t[i][2];
        v[i][2] = -t[i][1] + t[i][2];
        v[i][3] = t[i][1] - t[i][3];
    }
}



static inline void winograd_transform_weight_3x3(const float g[3][3], float u[4][4])
{
    float t[4][3];
    for (int j = 0; j < 3; j++) {
        t[0][j] = g[0][j];
        t[1][j] = 0.5f * (g[0][j] + g[1][j] + g[2][j]);
        t[2][j] = 0.5f * (g[0][j] - g[1][j] + g[2][j]);
        t[3][j] = g[2][j];
    }
    for (int i = 0; i < 4; i++) {
        u[i][0] = t[i][0];
        u[i][1] = 0.5f * (t[i][0] + t[i][1] + t[i][2]);
        u[i][2] = 0.5f * (t[i][0] - t[i][1] + t[i][2]);
        u[i][3] = t[i][2];
    }
}



static inline void winograd_transform_output_2x2(const float m[4][4], float out[2][2])
{
    float t[2][4];
    for (int j = 0; j < 4; j++) {
        t[0][j] = m[0][j] + m[1][j] + m[2][j];
        t[1][j] = m[1][j] - m[2][j] - m[3][j];
    }
    for (int i = 0; i < 2; i++) {
        out[i][0] = t[i][0] + t[i][1] + t[i][2];
        out[i][1] = t[i][1] - t[i][2] - t[i][3];
    }
}


/* ── Winograd F(4,3): output tile 4×4, input tile 6×6 ── */

static inline void winograd_f43_transform_input_6x6(const float d[6][6], float v[6][6])
{
    float t[6][6];
    for (int j = 0; j < 6; j++) {
        t[0][j] =  4.0f*d[0][j]              - 5.0f*d[2][j]              + d[4][j];
        t[1][j] =            - 4.0f*d[1][j]  - 4.0f*d[2][j] +    d[3][j] + d[4][j];
        t[2][j] =              4.0f*d[1][j]  - 4.0f*d[2][j] -    d[3][j] + d[4][j];
        t[3][j] =            - 2.0f*d[1][j]  -      d[2][j] + 2.0f*d[3][j] + d[4][j];
        t[4][j] =              2.0f*d[1][j]  -      d[2][j] - 2.0f*d[3][j] + d[4][j];
        t[5][j] =              4.0f*d[1][j]              - 5.0f*d[3][j]              + d[5][j];
    }
    for (int i = 0; i < 6; i++) {
        v[i][0] =  4.0f*t[i][0]              - 5.0f*t[i][2]              + t[i][4];
        v[i][1] =            - 4.0f*t[i][1]  - 4.0f*t[i][2] +    t[i][3] + t[i][4];
        v[i][2] =              4.0f*t[i][1]  - 4.0f*t[i][2] -    t[i][3] + t[i][4];
        v[i][3] =            - 2.0f*t[i][1]  -      t[i][2] + 2.0f*t[i][3] + t[i][4];
        v[i][4] =              2.0f*t[i][1]  -      t[i][2] - 2.0f*t[i][3] + t[i][4];
        v[i][5] =              4.0f*t[i][1]              - 5.0f*t[i][3]              + t[i][5];
    }
}

static inline void winograd_f43_transform_weight_3x3(const float g[3][3], float u[6][6])
{
    const float c14  = 0.25f;
    const float c16  = 1.0f / 6.0f;
    const float c112 = 1.0f / 12.0f;
    const float c124 = 1.0f / 24.0f;
    float t[6][3];
    for (int j = 0; j < 3; j++) {
        t[0][j] =  c14  * g[0][j];
        t[1][j] = -c16  * (g[0][j] + g[1][j] + g[2][j]);
        t[2][j] = -c16  * (g[0][j] - g[1][j] + g[2][j]);
        t[3][j] =  c124 * g[0][j] + c112 * g[1][j] + c16 * g[2][j];
        t[4][j] =  c124 * g[0][j] - c112 * g[1][j] + c16 * g[2][j];
        t[5][j] = g[2][j];
    }
    for (int i = 0; i < 6; i++) {
        u[i][0] =  c14  * t[i][0];
        u[i][1] = -c16  * (t[i][0] + t[i][1] + t[i][2]);
        u[i][2] = -c16  * (t[i][0] - t[i][1] + t[i][2]);
        u[i][3] =  c124 * t[i][0] + c112 * t[i][1] + c16 * t[i][2];
        u[i][4] =  c124 * t[i][0] - c112 * t[i][1] + c16 * t[i][2];
        u[i][5] = t[i][2];
    }
}

static inline void winograd_f43_transform_output_4x4(const float m[6][6], float out[4][4])
{
    float t[4][6];
    for (int j = 0; j < 6; j++) {
        t[0][j] = m[0][j] + m[1][j] + m[2][j] +      m[3][j] +      m[4][j];
        t[1][j] =           m[1][j] - m[2][j] + 2.0f*m[3][j] - 2.0f*m[4][j];
        t[2][j] =           m[1][j] + m[2][j] + 4.0f*m[3][j] + 4.0f*m[4][j];
        t[3][j] =           m[1][j] - m[2][j] + 8.0f*m[3][j] - 8.0f*m[4][j] + m[5][j];
    }
    for (int i = 0; i < 4; i++) {
        out[i][0] = t[i][0] + t[i][1] + t[i][2] +      t[i][3] +      t[i][4];
        out[i][1] =           t[i][1] - t[i][2] + 2.0f*t[i][3] - 2.0f*t[i][4];
        out[i][2] =           t[i][1] + t[i][2] + 4.0f*t[i][3] + 4.0f*t[i][4];
        out[i][3] =           t[i][1] - t[i][2] + 8.0f*t[i][3] - 8.0f*t[i][4] + t[i][5];
    }
}



static float *winograd_pack_weights_3x3(int OC, int C, const float *w)
{
    int num_m_blocks = (OC + SGEMM_MR - 1) / SGEMM_MR;
    size_t one_alpha = (size_t)num_m_blocks * C * SGEMM_MR;
    float *packed = NULL;
    if (posix_memalign((void **)&packed, 32, 16 * one_alpha * sizeof(float)) != 0)
        return NULL;
    memset(packed, 0, 16 * one_alpha * sizeof(float));

    for (int oc = 0; oc < OC; oc++) {
        int mi = oc / SGEMM_MR;
        int mr = oc % SGEMM_MR;
        for (int ic = 0; ic < C; ic++) {
            const float *gptr = w + ((size_t)oc * C + ic) * 9;
            float g[3][3], u[4][4];
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    g[kh][kw] = gptr[kh * 3 + kw];
            winograd_transform_weight_3x3(g, u);

            for (int a = 0; a < 16; a++) {
                packed[(size_t)a * one_alpha + (size_t)mi * C * SGEMM_MR +
                       (size_t)ic * SGEMM_MR + mr] = u[a / 4][a % 4];
            }
        }
    }
    return packed;
}



static void winograd_gemm_small_tiles(int OC, int tiles, int C,
                                      const float *packed_u,
                                      size_t one_alpha_packed,
                                      const float *V,
                                      float *M)
{
    #pragma omp parallel for schedule(static) if((long long)OC * tiles * C >= 200000)
    for (int oc = 0; oc < OC; oc++) {
        int mi = oc / SGEMM_MR;
        int mr = oc % SGEMM_MR;
        for (int a = 0; a < 16; a++) {
            const float *u_base = packed_u + (size_t)a * one_alpha_packed +
                                  (size_t)mi * C * SGEMM_MR + mr;
            const float *v_base = V + (size_t)a * C * tiles;
            float *m_base = M + (size_t)a * OC * tiles + (size_t)oc * tiles;
            int t = 0;
            for (; t + 3 < tiles; t += 4) {
                float32x4_t acc = vdupq_n_f32(0);
                for (int c = 0; c < C; c++) {
                    float32x4_t vv = vld1q_f32(v_base + (size_t)c * tiles + t);
                    acc = vfmaq_f32(acc, vdupq_n_f32(u_base[(size_t)c * SGEMM_MR]),
                                    vv);
                }
                vst1q_f32(m_base + t, acc);
            }
            for (; t < tiles; t++) {
                float acc = 0.0f;
                for (int c = 0; c < C; c++) {
                    acc += u_base[(size_t)c * SGEMM_MR] * v_base[(size_t)c * tiles + t];
                }
                m_base[t] = acc;
            }
        }
    }
}



static int winograd_conv3x3s1p1(Spkv2Context *ctx,
                                const Spkv2NodeRecord *node,
                                const float *x, const float *w,
                                const float *bias, float *y,
                                int N_batch, int C, int H, int W,
                                int OC, int OH, int OW,
                                int act_type, void *scratch)
{
    if (!scratch) return -13;
    int tile_h = (OH + 1) / 2;
    int tile_w = (OW + 1) / 2;
    int tiles = tile_h * tile_w;
    if (tiles <= 0) return -99;

    size_t v_size = (size_t)16 * C * tiles;
    size_t m_size = (size_t)16 * OC * tiles;
    float *V = (float *)scratch;
    float *M_buf = V + v_size;

    if (!ctx->node_cache || node->id >= ctx->node_cache_count) return -99;
    if (!ctx->node_cache[node->id]) {
        ctx->node_cache[node->id] = winograd_pack_weights_3x3(OC, C, w);
        if (!ctx->node_cache[node->id]) return -1;
    }
    const float *packed_u = (const float *)ctx->node_cache[node->id];
    int num_m_blocks = (OC + SGEMM_MR - 1) / SGEMM_MR;
    size_t one_alpha_packed = (size_t)num_m_blocks * C * SGEMM_MR;

    for (int n = 0; n < N_batch; n++) {
        const float *xn = x + (size_t)n * C * H * W;
        float *yn = y + (size_t)n * OC * OH * OW;
        memset(V, 0, v_size * sizeof(float));
        memset(M_buf, 0, m_size * sizeof(float));

        #pragma omp parallel for schedule(static) if((long long)C * tiles >= 512)
        for (int ic = 0; ic < C; ic++) {
            const float *xc = xn + (size_t)ic * H * W;
            for (int th = 0; th < tile_h; th++) {
                int oh0 = th * 2;
                for (int tw = 0; tw < tile_w; tw++) {
                    int ow0 = tw * 2;
                    int tile = th * tile_w + tw;
                    float d[4][4], vt[4][4];
                    for (int i = 0; i < 4; i++) {
                        int ih = oh0 + i - 1;
                        for (int j = 0; j < 4; j++) {
                            int iw = ow0 + j - 1;
                            d[i][j] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                          ? xc[(size_t)ih * W + iw]
                                          : 0.0f;
                        }
                    }
                    winograd_transform_input_4x4(d, vt);
                    for (int a = 0; a < 16; a++) {
                        V[(size_t)a * C * tiles + (size_t)ic * tiles + tile] =
                            vt[a / 4][a % 4];
                    }
                }
            }
        }

        if (tiles <= 256) {
            winograd_gemm_small_tiles(OC, tiles, C, packed_u,
                                      one_alpha_packed, V, M_buf);
        } else {
            int allow_gemm_parallel = tiles >= 256;
            for (int a = 0; a < 16; a++) {
                sgemm_nn_packed_a_impl_run(OC, tiles, C,
                                           packed_u + (size_t)a * one_alpha_packed,
                                           V + (size_t)a * C * tiles, tiles,
                                           M_buf + (size_t)a * OC * tiles, tiles,
                                           allow_gemm_parallel);
            }
        }

        #pragma omp parallel for schedule(static) if((long long)OC * tiles >= 512)
        for (int oc = 0; oc < OC; oc++) {
            float *yoc = yn + (size_t)oc * OH * OW;
            float bv = bias ? bias[oc] : 0.0f;
            for (int th = 0; th < tile_h; th++) {
                int oh0 = th * 2;
                for (int tw = 0; tw < tile_w; tw++) {
                    int ow0 = tw * 2;
                    int tile = th * tile_w + tw;
                    float mt[4][4], out[2][2];
                    for (int a = 0; a < 16; a++) {
                        mt[a / 4][a % 4] =
                            M_buf[(size_t)a * OC * tiles + (size_t)oc * tiles + tile];
                    }
                    winograd_transform_output_2x2(mt, out);
                    for (int i = 0; i < 2 && oh0 + i < OH; i++) {
                        for (int j = 0; j < 2 && ow0 + j < OW; j++) {
                            float value = out[i][j] + bv;
                            yoc[(size_t)(oh0 + i) * OW + ow0 + j] =
                                apply_activation_scalar_simd(value, act_type);
                        }
                    }
                }
            }
        }
    }
    return 0;
}



/* ── Winograd F(4,3) weight packing: 36 alpha-matrices ── */

static float *winograd_f43_pack_weights_3x3(int OC, int C, const float *w)
{
    int num_m_blocks = (OC + SGEMM_MR - 1) / SGEMM_MR;
    size_t one_alpha = (size_t)num_m_blocks * C * SGEMM_MR;
    float *packed = NULL;
    if (posix_memalign((void **)&packed, 32, 36 * one_alpha * sizeof(float)) != 0)
        return NULL;
    memset(packed, 0, 36 * one_alpha * sizeof(float));

    for (int oc = 0; oc < OC; oc++) {
        int mi = oc / SGEMM_MR;
        int mr = oc % SGEMM_MR;
        for (int ic = 0; ic < C; ic++) {
            const float *gptr = w + ((size_t)oc * C + ic) * 9;
            float g[3][3], u[6][6];
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    g[kh][kw] = gptr[kh * 3 + kw];
            winograd_f43_transform_weight_3x3(g, u);

            for (int a = 0; a < 36; a++) {
                packed[(size_t)a * one_alpha + (size_t)mi * C * SGEMM_MR +
                       (size_t)ic * SGEMM_MR + mr] = u[a / 6][a % 6];
            }
        }
    }
    return packed;
}


static void winograd_f43_gemm_small_tiles(int OC, int tiles, int C,
                                          const float *packed_u,
                                          size_t one_alpha_packed,
                                          const float *V,
                                          float *M)
{
    #pragma omp parallel for schedule(static) if((long long)OC * tiles * C >= 200000)
    for (int oc = 0; oc < OC; oc++) {
        int mi = oc / SGEMM_MR;
        int mr = oc % SGEMM_MR;
        for (int a = 0; a < 36; a++) {
            const float *u_base = packed_u + (size_t)a * one_alpha_packed +
                                  (size_t)mi * C * SGEMM_MR + mr;
            const float *v_base = V + (size_t)a * C * tiles;
            float *m_base = M + (size_t)a * OC * tiles + (size_t)oc * tiles;
            int t = 0;
            for (; t + 3 < tiles; t += 4) {
                float32x4_t acc = vdupq_n_f32(0);
                for (int c = 0; c < C; c++) {
                    float32x4_t vv = vld1q_f32(v_base + (size_t)c * tiles + t);
                    acc = vfmaq_f32(acc, vdupq_n_f32(u_base[(size_t)c * SGEMM_MR]),
                                    vv);
                }
                vst1q_f32(m_base + t, acc);
            }
            for (; t < tiles; t++) {
                float acc = 0.0f;
                for (int c = 0; c < C; c++) {
                    acc += u_base[(size_t)c * SGEMM_MR] * v_base[(size_t)c * tiles + t];
                }
                m_base[t] = acc;
            }
        }
    }
}

#if defined(__APPLE__)
static float *winograd_f43_pack_weights_3x3_cblas(int OC, int C, const float *w)
{
    size_t one_alpha = (size_t)OC * C;
    float *packed = NULL;
    if (posix_memalign((void **)&packed, 32, 36 * one_alpha * sizeof(float)) != 0)
        return NULL;

    for (int oc = 0; oc < OC; oc++) {
        for (int ic = 0; ic < C; ic++) {
            const float *gptr = w + ((size_t)oc * C + ic) * 9;
            float g[3][3], u[6][6];
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    g[kh][kw] = gptr[kh * 3 + kw];
            winograd_f43_transform_weight_3x3(g, u);
            for (int a = 0; a < 36; a++)
                packed[(size_t)a * one_alpha + (size_t)oc * C + ic] = u[a / 6][a % 6];
        }
    }
    return packed;
}
#endif

static int winograd_f43_conv3x3s1p1(Spkv2Context *ctx,
                                     const Spkv2NodeRecord *node,
                                     const float *x, const float *w,
                                     const float *bias, float *y,
                                     int N_batch, int C, int H, int W,
                                     int OC, int OH, int OW,
                                     int act_type, const float *residual,
                                     void *scratch)
{
    if (!scratch) return -13;
    int tile_h = (OH + 3) / 4;
    int tile_w = (OW + 3) / 4;
    int tiles = tile_h * tile_w;
    if (tiles <= 0) return -99;

    size_t v_size = (size_t)36 * C * tiles;
    size_t m_size = (size_t)36 * OC * tiles;
    float *V = (float *)scratch;
    float *M = V + v_size;

    if (!ctx->node_cache || node->id >= ctx->node_cache_count) return -99;
#if defined(__APPLE__)
    if (!ctx->node_cache[node->id]) {
        ctx->node_cache[node->id] = winograd_f43_pack_weights_3x3_cblas(OC, C, w);
        if (!ctx->node_cache[node->id]) return -1;
    }
    const float *packed_u = (const float *)ctx->node_cache[node->id];
    size_t one_alpha_packed = (size_t)OC * C;
#else
    if (!ctx->node_cache[node->id]) {
        ctx->node_cache[node->id] = winograd_f43_pack_weights_3x3(OC, C, w);
        if (!ctx->node_cache[node->id]) return -1;
    }
    const float *packed_u = (const float *)ctx->node_cache[node->id];
    int num_m_blocks = (OC + SGEMM_MR - 1) / SGEMM_MR;
    size_t one_alpha_packed = (size_t)num_m_blocks * C * SGEMM_MR;
#endif

    for (int n = 0; n < N_batch; n++) {
        const float *xn = x + (size_t)n * C * H * W;
        float *yn = y + (size_t)n * OC * OH * OW;
        memset(M, 0, m_size * sizeof(float));

        #pragma omp parallel for schedule(static) if((long long)C * tiles >= 512)
        for (int ic = 0; ic < C; ic++) {
            const float *xc = xn + (size_t)ic * H * W;
            for (int th = 0; th < tile_h; th++) {
                int oh0 = th * 4;
                for (int tw = 0; tw < tile_w; tw++) {
                    int ow0 = tw * 4;
                    int tile = th * tile_w + tw;
                    float d[6][6], vt[6][6];
                    for (int i = 0; i < 6; i++) {
                        int ih = oh0 + i - 1;
                        for (int j = 0; j < 6; j++) {
                            int iw = ow0 + j - 1;
                            d[i][j] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                          ? xc[(size_t)ih * W + iw]
                                          : 0.0f;
                        }
                    }
                    winograd_f43_transform_input_6x6(d, vt);
                    for (int a = 0; a < 36; a++) {
                        V[(size_t)a * C * tiles + (size_t)ic * tiles + tile] =
                            vt[a / 6][a % 6];
                    }
                }
            }
        }

#if defined(__APPLE__)
        spkv2_apply(36, ^(size_t a) {
            sgemm_nn(OC, tiles, C,
                     packed_u + (size_t)a * one_alpha_packed, C,
                     V + (size_t)a * C * tiles, tiles,
                     M + (size_t)a * OC * tiles, tiles);
        });
#else
        if (tiles <= 256) {
            winograd_f43_gemm_small_tiles(OC, tiles, C, packed_u,
                                          one_alpha_packed, V, M);
        } else {
            int allow_gemm_parallel = tiles >= 256;
            for (int a = 0; a < 36; a++) {
                sgemm_nn_packed_a_impl_run(OC, tiles, C,
                                           packed_u + (size_t)a * one_alpha_packed,
                                           V + (size_t)a * C * tiles, tiles,
                                           M + (size_t)a * OC * tiles, tiles,
                                           allow_gemm_parallel);
            }
        }
#endif

        #pragma omp parallel for schedule(static) if((long long)OC * tiles >= 512)
        for (int oc = 0; oc < OC; oc++) {
            float *yoc = yn + (size_t)oc * OH * OW;
            float bv = bias ? bias[oc] : 0.0f;
            for (int th = 0; th < tile_h; th++) {
                int oh0 = th * 4;
                for (int tw = 0; tw < tile_w; tw++) {
                    int ow0 = tw * 4;
                    int tile = th * tile_w + tw;
                    float mt[6][6], out[4][4];
                    for (int a = 0; a < 36; a++) {
                        mt[a / 6][a % 6] =
                            M[(size_t)a * OC * tiles + (size_t)oc * tiles + tile];
                    }
                    winograd_f43_transform_output_4x4(mt, out);
                    for (int i = 0; i < 4 && oh0 + i < OH; i++) {
                        for (int j = 0; j < 4 && ow0 + j < OW; j++) {
                            float value = out[i][j] + bv;
                            if (residual)
                                value += residual[(size_t)n * OC * OH * OW +
                                                  (size_t)oc * OH * OW +
                                                  (size_t)(oh0 + i) * OW + ow0 + j];
                            yoc[(size_t)(oh0 + i) * OW + ow0 + j] =
                                apply_activation_scalar_simd(value, act_type);
                        }
                    }
                }
            }
        }
    }
    return 0;
}



/* -----------------------------------------------------------------------
 * Direct 3x3 stride-2 convolution using NEON vld2q_f32 deinterleave load.
 * Avoids the large im2col buffer (11-14 MB for YOLOv10 early layers).
 * Parallel over output channels via dispatch_apply.
 * Only handles pad==1 (most common case for YOLOv10/ResNet).
 * ----------------------------------------------------------------------- */
#if defined(__APPLE__)
static void conv3x3_stride2_neon(const float *x, const float *w, const float *bias,
                                   float *y, int N, int C_in, int H, int W,
                                   int C_out, int outH, int outW, int fused_act)
{
    int nc_total = N * C_out;
    spkv2_apply((size_t)nc_total, ^(size_t nc_idx) {
        int ni = (int)nc_idx / C_out;
        int oc = (int)nc_idx % C_out;
        const float *xn = x + (size_t)ni * C_in * H * W;
        float *y_oc = y + (size_t)ni * C_out * outH * outW + (size_t)oc * outH * outW;
        float bv = bias ? bias[oc] : 0.0f;

        vDSP_vfill(&bv, y_oc, 1, (vDSP_Length)((size_t)outH * outW));

        for (int ic = 0; ic < C_in; ic++) {
            const float *x_ic = xn + (size_t)ic * H * W;
            const float *w_ic = w + ((size_t)oc * C_in + ic) * 9;

            for (int kh = 0; kh < 3; kh++) {
                for (int oh = 0; oh < outH; oh++) {
                    int ih = 2 * oh + kh - 1;
                    if (ih < 0 || ih >= H) continue;
                    const float *xr = x_ic + (size_t)ih * W;
                    float *yr = y_oc + (size_t)oh * outW;

                    for (int kw = 0; kw < 3; kw++) {
                        float wv = w_ic[kh * 3 + kw];
                        float32x4_t vw = vdupq_n_f32(wv);
                        /* iw = 2*ow + iw_base where iw_base = kw - 1 (pad=1) */
                        int iw_base = kw - 1;
                        /* Skip ow=0 for kw=0 (iw=-1 is padding zero) */
                        int ow_start = (iw_base < 0) ? 1 : 0;
                        /* vld2q_f32 reads 8 floats: need iw + 7 < W */
                        int ow_neon_end = (W - 7 - iw_base) / 2;
                        if (ow_neon_end > outW) ow_neon_end = outW;

                        int ow = ow_start;
                        for (; ow + 3 < ow_neon_end; ow += 4) {
                            int iw = 2 * ow + iw_base;
                            /* vld2q_f32: val[0]={xr[iw],xr[iw+2],xr[iw+4],xr[iw+6]} */
                            float32x4x2_t v = vld2q_f32(xr + iw);
                            float32x4_t acc = vld1q_f32(yr + ow);
                            vst1q_f32(yr + ow, vfmaq_f32(acc, v.val[0], vw));
                        }
                        /* Scalar tail + boundaries */
                        for (; ow < outW; ow++) {
                            int iw = 2 * ow + iw_base;
                            if (iw >= 0 && iw < W)
                                yr[ow] += wv * xr[iw];
                        }
                    }
                }
            }
        }
        fused_activation_pass(y_oc, (size_t)outH * outW, fused_act);
    });
}
#endif



static int conv3x3_direct_neon(const float *x, const float *w, const float *bias,
                               float *y, int N_batch, int C_in, int H, int W,
                               int C_out, int outH, int outW, int stride,
                               int act_type)
{
    /* NEON direct 3x3: stride==1 uses NEON FMA; stride==2 uses vld2q deinterleave.
       Both paths parallelize over output channels via dispatch_apply on Apple. */
#if defined(__APPLE__)
    if (stride == 2) {
        /* pad=1 is required for stride-2 direct conv (standard for 3x3 s2 in YOLO/ResNet) */
        conv3x3_stride2_neon(x, w, bias, y, N_batch, C_in, H, W,
                             C_out, outH, outW, act_type);
        return 0;
    }
#endif
    if (stride != 1) return -99;  /* non-Apple non-stride-1 fallback */

    int use_par = ((long long)N_batch * C_out * outH * outW * C_in > 3000000);

    #pragma omp parallel for collapse(2) schedule(static) if(use_par)
    for (int n = 0; n < N_batch; n++) {
        for (int m = 0; m < C_out; m++) {
            const float *w_m = w + (size_t)m * C_in * 9;
            float *y_m = y + (size_t)n * C_out * outH * outW + (size_t)m * outH * outW;
            float bv = bias ? bias[m] : 0.0f;

            for (int oh = 0; oh < outH; oh++) {
                int ow = 0;
                int ih_base = oh - 1;

                /* Left border (scalar) */
                for (; ow < outW && ow < 1; ow++) {
                    float acc = bv;
                    for (int c = 0; c < C_in; c++) {
                        const float *x_c = x + (size_t)n * C_in * H * W + (size_t)c * H * W;
                        const float *w_c = w_m + (size_t)c * 9;
                        for (int kh = 0; kh < 3; kh++) {
                            int ih = ih_base + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < 3; kw++) {
                                int iw = ow + kw - 1;
                                if (iw >= 0 && iw < W)
                                    acc += x_c[(size_t)ih * W + iw] * w_c[kh * 3 + kw];
                            }
                        }
                    }
                    y_m[(size_t)oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
                }

                /* Interior: NEON vectorized stride-1 */
                int vec_end = outW - 1;
                for (; ow + 3 < vec_end; ow += 4) {
                    float32x4_t acc = vdupq_n_f32(bv);
                    for (int c = 0; c < C_in; c++) {
                        const float *x_c = x + (size_t)n * C_in * H * W + (size_t)c * H * W;
                        const float *w_c = w_m + (size_t)c * 9;
                        for (int kh = 0; kh < 3; kh++) {
                            int ih = ih_base + kh;
                            if (ih < 0 || ih >= H) continue;
                            const float *row = x_c + (size_t)ih * W;
                            acc = vfmaq_f32(acc, vdupq_n_f32(w_c[kh * 3 + 0]),
                                            vld1q_f32(row + ow - 1));
                            acc = vfmaq_f32(acc, vdupq_n_f32(w_c[kh * 3 + 1]),
                                            vld1q_f32(row + ow));
                            acc = vfmaq_f32(acc, vdupq_n_f32(w_c[kh * 3 + 2]),
                                            vld1q_f32(row + ow + 1));
                        }
                    }
                    vst1q_f32(y_m + (size_t)oh * outW + ow,
                              apply_activation_neon(acc, act_type));
                }

                /* Right border + scalar tail */
                for (; ow < outW; ow++) {
                    float acc = bv;
                    for (int c = 0; c < C_in; c++) {
                        const float *x_c = x + (size_t)n * C_in * H * W + (size_t)c * H * W;
                        const float *w_c = w_m + (size_t)c * 9;
                        for (int kh = 0; kh < 3; kh++) {
                            int ih = ih_base + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < 3; kw++) {
                                int iw = ow + kw - 1;
                                if (iw >= 0 && iw < W)
                                    acc += x_c[(size_t)ih * W + iw] * w_c[kh * 3 + kw];
                            }
                        }
                    }
                    y_m[(size_t)oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
                }
            }
        }
    }
    return 0;
}



int kernel_conv_simd(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch)
{
    Spkv2AttrRecord attr;
    int rc = simd_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    if (!scratch && node->scratch_bytes > 0) return -13;

    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *w_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;
    const float *x = (const float *)ctx->tensors[node->inputs[0]].data;
    const float *w = (const float *)ctx->tensors[node->inputs[1]].data;
    const float *bias = node->input_count > 2
                            ? (const float *)ctx->tensors[node->inputs[2]].data
                            : NULL;
    const float *residual = node->input_count >= 4
                            ? (const float *)ctx->tensors[node->inputs[3]].data
                            : NULL;
    float *y = (float *)ctx->tensors[node->outputs[0]].data;
    float *col = (float *)scratch;

    int N_batch = (int)x_rec->shape[0];
    int C_in  = (int)x_rec->shape[1];
    int H  = (int)x_rec->shape[2];
    int W  = (int)x_rec->shape[3];
    int C_out = (int)w_rec->shape[0];
    int kH = (int)w_rec->shape[2];
    int kW = (int)w_rec->shape[3];
    int outH = (int)y_rec->shape[2];
    int outW = (int)y_rec->shape[3];
    int spatial = outH * outW;
    int group = attr.group;
    const Spkv2KernelSpecRecord *spec = simd_node_spec(ctx, node);
    uint16_t kernel_kind = spec ? spec->kernel_kind : SPKV2_KERNEL_IM2COL_GEMM;

    /* FP16 weight promotion: convert to FP32 once and cache in node_cache */
    if (w_rec->dtype == SPKV2_DTYPE_FP16 && ctx->node_cache
        && node->id < ctx->node_cache_count) {
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
        if (!ctx->node_cache[node->id]) {
            size_t w_elems = 1;
            for (int d = 0; d < w_rec->rank; d++) w_elems *= w_rec->shape[d];
            float *w32 = (float *)malloc(w_elems * sizeof(float));
            if (!w32) return -1;
            const _Float16 *w16 = (const _Float16 *)ctx->tensors[node->inputs[1]].data;
            for (size_t i = 0; i < w_elems; i++) w32[i] = (float)w16[i];
            ctx->node_cache[node->id] = w32;
        }
        w = (const float *)ctx->node_cache[node->id];
#else
        return -99;
#endif
    }

    /* -- Depthwise conv: group == C_in == C_out -- */
    if (group == C_in && group == C_out) {
        for (int n = 0; n < N_batch; n++) {
            depthwise_conv_simd(x + (size_t)n * C_in * H * W,
                                w, bias,
                                y + (size_t)n * C_out * spatial,
                                C_in, H, W, kH, kW, outH, outW,
                                attr.strides[0], attr.strides[1],
                                attr.pads[0], attr.pads[1],
                                attr.dilations[0], attr.dilations[1]);
            if (residual) {
                float *y_n = y + (size_t)n * C_out * spatial;
                const float *r_n = residual + (size_t)n * C_out * spatial;
#if defined(__APPLE__)
                vDSP_vadd(y_n, 1, r_n, 1, y_n, 1, (vDSP_Length)(C_out * spatial));
#else
                for (size_t i = 0; i < (size_t)C_out * spatial; i++)
                    y_n[i] += r_n[i];
#endif
            }
            fused_activation_pass(y + (size_t)n * C_out * spatial,
                                  (size_t)C_out * spatial, attr.fused_activation);
        }
        return 0;
    }

    /* -- Grouped / standard conv: im2col + SGEMM per group -- */
    int C_in_g  = C_in / group;
    int C_out_g = C_out / group;
    int K_g     = C_in_g * kH * kW;

    int is_1x1_s1_p0 = (kH == 1 && kW == 1 &&
                         attr.strides[0] == 1 && attr.strides[1] == 1 &&
                         attr.pads[0] == 0 && attr.pads[1] == 0);

    int is_3x3_s1_p1_d1 = (kH == 3 && kW == 3 &&
                             attr.strides[0] == 1 && attr.strides[1] == 1 &&
                             attr.pads[0] == 1 && attr.pads[1] == 1 &&
                             attr.dilations[0] == 1 && attr.dilations[1] == 1 &&
                             outH == H && outW == W);

    if (kernel_kind == SPKV2_KERNEL_WINOGRAD_3X3S1) {
        if (group != 1 || !is_3x3_s1_p1_d1) {
            return -99;
        }
        return winograd_conv3x3s1p1(ctx, node, x, w, bias, y,
                                    N_batch, C_in, H, W, C_out, outH, outW,
                                    attr.fused_activation, scratch);
    }

    if (kernel_kind == SPKV2_KERNEL_WINOGRAD_F43) {
        if (group != 1 || !is_3x3_s1_p1_d1) {
            return -99;
        }
        return winograd_f43_conv3x3s1p1(ctx, node, x, w, bias, y,
                                         N_batch, C_in, H, W, C_out, outH, outW,
                                         attr.fused_activation, residual, scratch);
    }

    if (kernel_kind == SPKV2_KERNEL_CONV3X3S2_DIRECT) {
        /* On NEON, direct conv3x3 only handles stride==1; stride>1 falls back */
        if (group != 1 || kH != 3 || kW != 3 ||
            attr.pads[0] != 1 || attr.pads[1] != 1 ||
            attr.dilations[0] != 1 || attr.dilations[1] != 1) {
            return -99;
        }
        return conv3x3_direct_neon(x, w, bias, y, N_batch, C_in, H, W,
                                   C_out, outH, outW, attr.strides[0],
                                   attr.fused_activation);
    }

    /* -- Lazy weight pre-packing (via node_cache) -- */
    float *packed_w = NULL;
#if !defined(__APPLE__)
    if (group == 1 && ctx->node_cache && node->id < ctx->node_cache_count
        && w_rec->dtype != SPKV2_DTYPE_FP16) {
        if (!ctx->node_cache[node->id]) {
            ctx->node_cache[node->id] = sgemm_pack_a_impl(C_out_g, K_g, w, K_g);
        }
        packed_w = (float *)ctx->node_cache[node->id];
    }
#endif
    if (is_1x1_s1_p0) {
        for (int n = 0; n < N_batch; n++) {
            float *y_n = y + (size_t)n * C_out * spatial;

            for (int g = 0; g < group; g++) {
                const float *x_g = x + (size_t)n * C_in * H * W + (size_t)g * C_in_g * H * W;
                const float *w_g = w + (size_t)g * C_out_g * K_g;
                const float *bias_g = bias ? bias + g * C_out_g : NULL;
                float *y_g = y_n + (size_t)g * C_out_g * spatial;

                /* init Y with bias */
#if defined(__APPLE__)
                {
                    int cout = C_out_g; int sp = spatial;
                    spkv2_apply((size_t)cout, ^(size_t m) {
                        float bv = bias_g ? bias_g[m] : 0.0f;
                        vDSP_vfill(&bv, y_g + m * (size_t)sp, 1, (size_t)sp);
                    });
                }
#else
                for (int m = 0; m < C_out_g; m++) {
                    float bv = bias_g ? bias_g[m] : 0.0f;
                    bias_init_row(y_g + (size_t)m * spatial, bv, spatial);
                }
#endif

                if (packed_w && g == 0)
                    sgemm_nn_packed_a(C_out_g, spatial, C_in_g,
                                       packed_w, x_g, spatial, y_g, spatial);
                else
                    sgemm_nn(C_out_g, spatial, C_in_g,
                             w_g, C_in_g, x_g, spatial, y_g, spatial);
            }

            if (residual) {
                const float *r_n = residual + (size_t)n * C_out * spatial;
#if defined(__APPLE__)
                vDSP_vadd(y_n, 1, r_n, 1, y_n, 1, (vDSP_Length)(C_out * spatial));
#else
                for (size_t i = 0; i < (size_t)C_out * spatial; i++)
                    y_n[i] += r_n[i];
#endif
            }
            fused_activation_pass(y_n, (size_t)C_out * spatial, attr.fused_activation);
        }
        return 0;
    }

    /* ── General conv: segmented im2col + SGEMM per group ── */
    int scratch_floats = (int)(node->scratch_bytes / sizeof(float));
    int seg_spatial = K_g > 0 ? scratch_floats / K_g : spatial;
    if (seg_spatial <= 0) seg_spatial = spatial;
    seg_spatial = (seg_spatial / outW) * outW;
    if (seg_spatial <= 0) seg_spatial = outW;
    if (seg_spatial > spatial) seg_spatial = spatial;
    int seg_outH = seg_spatial / outW;
    if (seg_outH <= 0) seg_outH = 1;

    for (int n = 0; n < N_batch; n++) {
        float *y_n = y + (size_t)n * C_out * spatial;

        for (int g = 0; g < group; g++) {
            const float *x_g = x + (size_t)n * C_in * H * W + (size_t)g * C_in_g * H * W;
            const float *w_g = w + (size_t)g * C_out_g * K_g;
            const float *bias_g = bias ? bias + g * C_out_g : NULL;
            float *y_g = y_n + (size_t)g * C_out_g * spatial;

            /* initialise Y with bias */
#if defined(__APPLE__)
            {
                int cout = C_out_g; int sp = spatial;
                spkv2_apply((size_t)cout, ^(size_t m) {
                    float bv = bias_g ? bias_g[m] : 0.0f;
                    vDSP_vfill(&bv, y_g + m * (size_t)sp, 1, (size_t)sp);
                });
            }
#else
            for (int m = 0; m < C_out_g; m++) {
                float bv = bias_g ? bias_g[m] : 0.0f;
                bias_init_row(y_g + (size_t)m * spatial, bv, spatial);
            }
#endif

            /* im2col + SGEMM: use specialized 3x3s1p1 path when applicable */
            if (seg_spatial >= spatial) {
                if (is_3x3_s1_p1_d1 && group == 1)
                    im2col_3x3_s1p1(x_g, C_in_g, H, W, col);
                else
                    im2col_full(x_g, C_in_g, H, W, kH, kW,
                                attr.strides[0], attr.strides[1],
                                attr.pads[0], attr.pads[1],
                                attr.dilations[0], attr.dilations[1],
                                outH, outW, col);
                if (packed_w && g == 0)
                    sgemm_nn_packed_a(C_out_g, spatial, K_g, packed_w, col, spatial, y_g, spatial);
                else
                    sgemm_nn(C_out_g, spatial, K_g, w_g, K_g, col, spatial, y_g, spatial);
            } else {
                for (int oh_start = 0; oh_start < outH; oh_start += seg_outH) {
                    int oh_end = oh_start + seg_outH;
                    if (oh_end > outH) oh_end = outH;
                    int seg_cols = (oh_end - oh_start) * outW;

                    if (is_3x3_s1_p1_d1 && group == 1)
                        im2col_3x3_s1p1_segment(x_g, C_in_g, H, W, oh_start, oh_end, col);
                    else
                        im2col_segment(x_g, C_in_g, H, W, kH, kW,
                                       attr.strides[0], attr.strides[1],
                                       attr.pads[0], attr.pads[1],
                                       attr.dilations[0], attr.dilations[1],
                                       outW, oh_start, oh_end, col);

                    int col_offset = oh_start * outW;
                    if (packed_w && g == 0)
                        sgemm_nn_packed_a(C_out_g, seg_cols, K_g,
                                           packed_w, col, seg_cols,
                                           y_g + col_offset, spatial);
                    else
                        sgemm_nn(C_out_g, seg_cols, K_g,
                                 w_g, K_g, col, seg_cols,
                                 y_g + col_offset, spatial);
                }
            }
        }  /* end group loop */

        /* residual add + fused activation */
        if (residual) {
            const float *r_n = residual + (size_t)n * C_out * spatial;
#if defined(__APPLE__)
            vDSP_vadd(y_n, 1, r_n, 1, y_n, 1, (vDSP_Length)(C_out * spatial));
#else
            for (size_t i = 0; i < (size_t)C_out * spatial; i++)
                y_n[i] += r_n[i];
#endif
        }
        fused_activation_pass(y_n, (size_t)C_out * spatial, attr.fused_activation);
    }
    return 0;
}


#endif /* __AVX2__ || __ARM_NEON */
