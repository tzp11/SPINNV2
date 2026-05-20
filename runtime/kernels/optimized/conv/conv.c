#include "simd_kernels.h"
#include "simd_sgemm.h"

#ifdef __AVX2__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef enum {
    CONV_PHASE_DIRECT = 0,
    CONV_PHASE_PACK,
    CONV_PHASE_BIAS,
    CONV_PHASE_IM2COL,
    CONV_PHASE_SGEMM,
    CONV_PHASE_SGEMM_EPILOGUE,
    CONV_PHASE_EPILOGUE,
    CONV_PHASE_COUNT
} ConvPhase;

static int s_conv_phase_profile_enabled = -1;
static int s_conv_phase_profile_registered = 0;
static double s_conv_phase_total[CONV_PHASE_COUNT];
static unsigned long long s_conv_phase_count[CONV_PHASE_COUNT];

static const char *conv_phase_name(ConvPhase phase)
{
    switch (phase) {
    case CONV_PHASE_DIRECT: return "direct_ms";
    case CONV_PHASE_PACK: return "pack_ms";
    case CONV_PHASE_BIAS: return "bias_ms";
    case CONV_PHASE_IM2COL: return "im2col_ms";
    case CONV_PHASE_SGEMM: return "sgemm_ms";
    case CONV_PHASE_SGEMM_EPILOGUE: return "sgemm_epilogue_ms";
    case CONV_PHASE_EPILOGUE: return "epilogue_ms";
    default: return "unknown_ms";
    }
}

static double conv_phase_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void conv_phase_profile_dump(void)
{
    if (s_conv_phase_profile_enabled <= 0) return;

    unsigned long long total_count = 0;
    for (int i = 0; i < CONV_PHASE_COUNT; i++)
        total_count += s_conv_phase_count[i];
    if (total_count == 0) return;

    fprintf(stderr, "\n[SPKV2_CONV_PHASE_PROFILE]\n");
    fprintf(stderr, "  %-22s %10s %12s %12s\n", "phase", "count", "total_ms", "avg_ms");
    for (int i = 0; i < CONV_PHASE_COUNT; i++) {
        double avg = s_conv_phase_count[i] > 0
                         ? s_conv_phase_total[i] / (double)s_conv_phase_count[i]
                         : 0.0;
        fprintf(stderr, "  %-22s %10llu %12.3f %12.4f\n",
                conv_phase_name((ConvPhase)i),
                s_conv_phase_count[i],
                s_conv_phase_total[i],
                avg);
    }
}

static int conv_phase_profile_enabled(void)
{
    if (s_conv_phase_profile_enabled < 0) {
        const char *v = getenv("SPKV2_PROFILE_CONV_PHASES");
        s_conv_phase_profile_enabled = (v && v[0] && v[0] != '0') ? 1 : 0;
        if (s_conv_phase_profile_enabled && !s_conv_phase_profile_registered) {
            atexit(conv_phase_profile_dump);
            s_conv_phase_profile_registered = 1;
        }
    }
    return s_conv_phase_profile_enabled;
}

static void conv_phase_add(ConvPhase phase, double start_ms)
{
    if (!conv_phase_profile_enabled()) return;
    s_conv_phase_total[phase] += conv_phase_now_ms() - start_ms;
    s_conv_phase_count[phase]++;
}

static void depthwise_conv_s1d1(const float *x_ch, const float *w_ch, float bv,
                                 float *y_ch, int H, int W,
                                 int kH, int kW, int outH, int outW,
                                 int pH, int pW, int act_type)
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
            y_ch[oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
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
            _mm256_storeu_ps(y_ch + oh * outW + ow,
                             apply_activation_avx2(acc, act_type));
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
            y_ch[oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
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
            y_ch[oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
        }
    }
}



static void depthwise_conv_s2d1(const float *x_ch, const float *w_ch, float bv,
                                 float *y_ch, int H, int W,
                                 int kH, int kW, int outH, int outW,
                                 int pH, int pW, int act_type)
{
    const __m256i gather_idx = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);

    int oh_start = pH > 0 ? (pH + 1) / 2 : 0;
    int oh_end   = H >= kH ? (H + pH - kH + 1) / 2 : 0;
    if (oh_start < 0) oh_start = 0;
    if (oh_end > outH) oh_end = outH;
    if (oh_end < oh_start) oh_end = oh_start;

    int ow_start = pW > 0 ? (pW + 1) / 2 : 0;
    int ow_end   = W >= kW ? (W + pW - kW + 1) / 2 : 0;
    if (ow_start < 0) ow_start = 0;
    if (ow_end > outW) ow_end = outW;
    if (ow_end < ow_start) ow_end = ow_start;

    for (int oh = 0; oh < outH; oh++) {
        int ow = 0;

        if (oh < oh_start || oh >= oh_end) {
            for (; ow < outW; ow++) {
                float acc = bv;
                for (int kh = 0; kh < kH; kh++) {
                    int ih = oh * 2 + kh - pH;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < kW; kw++) {
                        int iw = ow * 2 + kw - pW;
                        if (iw >= 0 && iw < W)
                            acc += w_ch[kh * kW + kw] * x_ch[(size_t)ih * W + iw];
                    }
                }
                y_ch[(size_t)oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
            }
            continue;
        }

        for (; ow < ow_start; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh * 2 + kh - pH;
                const float *row = x_ch + (size_t)ih * W;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow * 2 + kw - pW;
                    if (iw >= 0 && iw < W)
                        acc += w_ch[kh * kW + kw] * row[iw];
                }
            }
            y_ch[(size_t)oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
        }

        for (; ow + 7 < ow_end; ow += 8) {
            __m256 acc = _mm256_set1_ps(bv);
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh * 2 + kh - pH;
                const float *row = x_ch + (size_t)ih * W + ow * 2 - pW;
                for (int kw = 0; kw < kW; kw++) {
                    __m256 x = _mm256_i32gather_ps(row + kw, gather_idx, 4);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(w_ch[kh * kW + kw]), x, acc);
                }
            }
            _mm256_storeu_ps(y_ch + (size_t)oh * outW + ow,
                             apply_activation_avx2(acc, act_type));
        }

        for (; ow < ow_end; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh * 2 + kh - pH;
                const float *row = x_ch + (size_t)ih * W;
                for (int kw = 0; kw < kW; kw++)
                    acc += w_ch[kh * kW + kw] * row[ow * 2 + kw - pW];
            }
            y_ch[(size_t)oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
        }

        for (; ow < outW; ow++) {
            float acc = bv;
            for (int kh = 0; kh < kH; kh++) {
                int ih = oh * 2 + kh - pH;
                const float *row = x_ch + (size_t)ih * W;
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow * 2 + kw - pW;
                    if (iw >= 0 && iw < W)
                        acc += w_ch[kh * kW + kw] * row[iw];
                }
            }
            y_ch[(size_t)oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
        }
    }
}



static void depthwise_conv_generic(const float *x_ch, const float *w_ch, float bv,
                                    float *y_ch, int H, int W,
                                    int kH, int kW, int outH, int outW,
                                    int sH, int sW, int pH, int pW,
                                    int dH, int dW, int act_type)
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
                y_ch[oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
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
            y_ch[oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
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
            y_ch[oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
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
            y_ch[oh * outW + ow] = apply_activation_scalar_simd(acc, act_type);
        }
    }
}



static void depthwise_conv_simd(const float *x, const float *w, const float *bias,
                                float *y, int C, int H, int W,
                                int kH, int kW, int outH, int outW,
                                int sH, int sW, int pH, int pW,
                                int dH, int dW, int act_type)
{
    int is_s1d1 = (sH == 1 && sW == 1 && dH == 1 && dW == 1);
    int is_s2d1 = (sH == 2 && sW == 2 && dH == 1 && dW == 1);

    #pragma omp parallel for if(C * outH * outW > 50000) schedule(static)
    for (int c = 0; c < C; c++) {
        const float *x_ch = x + (size_t)c * H * W;
        const float *w_ch = w + (size_t)c * kH * kW;
        float *y_ch = y + (size_t)c * outH * outW;
        float bv = bias ? bias[c] : 0.0f;

        if (is_s1d1) {
            depthwise_conv_s1d1(x_ch, w_ch, bv, y_ch, H, W,
                                kH, kW, outH, outW, pH, pW, act_type);
        } else if (is_s2d1) {
            depthwise_conv_s2d1(x_ch, w_ch, bv, y_ch, H, W,
                                kH, kW, outH, outW, pH, pW, act_type);
        } else {
            depthwise_conv_generic(x_ch, w_ch, bv, y_ch, H, W,
                                    kH, kW, outH, outW, sH, sW, pH, pW, dH, dW,
                                    act_type);
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

static void im2col_full_tile(const float *im, int C, int H, int W,
                             int kH, int kW, int sH, int sW, int pH, int pW,
                             int dH, int dW, int outW, int tile_start,
                             int tile_count, float *col)
{
    int row = 0;
    for (int c = 0; c < C; c++) {
        const float *xc = im + (size_t)c * H * W;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                float *dst = col + (size_t)row * tile_count;
                for (int t = 0; t < tile_count; t++) {
                    int out_index = tile_start + t;
                    int oh = out_index / outW;
                    int ow = out_index - oh * outW;
                    int ih = oh * sH - pH + kh * dH;
                    int iw = ow * sW - pW + kw * dW;
                    dst[t] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                               ? xc[(size_t)ih * W + iw]
                               : 0.0f;
                }
                row++;
            }
        }
    }
}

static void im2col_3x3_s1p1_tile(const float *im, int C, int H, int W,
                                 int tile_start, int tile_count, float *col)
{
    int row = 0;
    for (int c = 0; c < C; c++) {
        const float *xc = im + (size_t)c * H * W;
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                float *dst = col + (size_t)row * tile_count;
                int done = 0;
                while (done < tile_count) {
                    int out_index = tile_start + done;
                    int oh = out_index / W;
                    int ow0 = out_index - oh * W;
                    int len = SIMD_MIN(tile_count - done, W - ow0);
                    int ih = oh - 1 + kh;
                    float *dst_seg = dst + done;

                    if (ih < 0 || ih >= H) {
                        memset(dst_seg, 0, (size_t)len * sizeof(float));
                        done += len;
                        continue;
                    }

                    int ow1 = ow0 + len;
                    int valid0 = 1 - kw;
                    int valid1 = W + 1 - kw;
                    int copy0 = ow0 > valid0 ? ow0 : valid0;
                    int copy1 = ow1 < valid1 ? ow1 : valid1;

                    if (copy0 > ow0) {
                        memset(dst_seg, 0, (size_t)(copy0 - ow0) * sizeof(float));
                    }
                    if (copy1 > copy0) {
                        memcpy(dst_seg + (copy0 - ow0),
                               xc + (size_t)ih * W + (copy0 + kw - 1),
                               (size_t)(copy1 - copy0) * sizeof(float));
                    }
                    if (ow1 > copy1) {
                        memset(dst_seg + (copy1 - ow0), 0,
                               (size_t)(ow1 - copy1) * sizeof(float));
                    }
                    done += len;
                }
                row++;
            }
        }
    }
}



static void im2col_3x3_s2p1_tile(const float *im, int C, int H, int W,
                                 int outW, int tile_start, int tile_count,
                                 float *col)
{
    const __m256i gather_idx = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    int row = 0;
    for (int c = 0; c < C; c++) {
        const float *xc = im + (size_t)c * H * W;
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                float *dst = col + (size_t)row * tile_count;
                int done = 0;
                while (done < tile_count) {
                    int out_index = tile_start + done;
                    int oh = out_index / outW;
                    int ow0 = out_index - oh * outW;
                    int len = SIMD_MIN(tile_count - done, outW - ow0);
                    int ih = oh * 2 + kh - 1;
                    float *dst_seg = dst + done;

                    if (ih < 0 || ih >= H) {
                        memset(dst_seg, 0, (size_t)len * sizeof(float));
                        done += len;
                        continue;
                    }

                    int ow1 = ow0 + len;
                    int valid0 = (kw == 0) ? 1 : 0;
                    int valid1 = (W - kw) / 2 + 1;
                    if (valid1 > outW) valid1 = outW;
                    int copy0 = ow0 > valid0 ? ow0 : valid0;
                    int copy1 = ow1 < valid1 ? ow1 : valid1;

                    if (copy0 > ow0) {
                        memset(dst_seg, 0, (size_t)(copy0 - ow0) * sizeof(float));
                    }

                    int copy = copy0;
                    const float *src = xc + (size_t)ih * W + copy * 2 + kw - 1;
                    float *dst_copy = dst_seg + (copy0 - ow0);
                    for (; copy + 7 < copy1; copy += 8) {
                        __m256 v = _mm256_i32gather_ps(src, gather_idx, 4);
                        _mm256_storeu_ps(dst_copy, v);
                        src += 16;
                        dst_copy += 8;
                    }
                    for (; copy < copy1; copy++) {
                        *dst_copy++ = xc[(size_t)ih * W + copy * 2 + kw - 1];
                    }

                    if (ow1 > copy1) {
                        memset(dst_seg + (copy1 - ow0), 0,
                               (size_t)(ow1 - copy1) * sizeof(float));
                    }
                    done += len;
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
    int phase_profile = conv_phase_profile_enabled();

    /* ── Depthwise conv: group == C_in == C_out ── */
    if (group == C_in && group == C_out) {
        for (int n = 0; n < N_batch; n++) {
            double t_direct = phase_profile ? conv_phase_now_ms() : 0.0;
            depthwise_conv_simd(x + (size_t)n * C_in * H * W,
                                w, bias,
                                y + (size_t)n * C_out * spatial,
                                C_in, H, W, kH, kW, outH, outW,
                                attr.strides[0], attr.strides[1],
                                attr.pads[0], attr.pads[1],
                                attr.dilations[0], attr.dilations[1],
                                attr.fused_activation);
            if (phase_profile) conv_phase_add(CONV_PHASE_DIRECT, t_direct);
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
    int is_3x3_s2_p1_d1 = (kH == 3 && kW == 3 &&
                             attr.strides[0] == 2 && attr.strides[1] == 2 &&
                             attr.pads[0] == 1 && attr.pads[1] == 1 &&
                             attr.dilations[0] == 1 && attr.dilations[1] == 1);

    if (kernel_kind == SPKV2_KERNEL_WINOGRAD_3X3S1) {
        if (group != 1 || !is_3x3_s1_p1_d1) {
            return -99;
        }
        double t_direct = phase_profile ? conv_phase_now_ms() : 0.0;
        int wrc = winograd_conv3x3s1p1(ctx, node, x, w, bias, y,
                                       N_batch, C_in, H, W, C_out, outH, outW,
                                       attr.fused_activation, scratch);
        if (phase_profile) conv_phase_add(CONV_PHASE_DIRECT, t_direct);
        return wrc;
    }

    if (kernel_kind == SPKV2_KERNEL_CONV3X3S2_DIRECT) {
        if (group != 1 || kH != 3 || kW != 3 ||
            attr.strides[0] != 2 || attr.strides[1] != 2 ||
            attr.pads[0] != 1 || attr.pads[1] != 1 ||
            attr.dilations[0] != 1 || attr.dilations[1] != 1) {
            return -99;
        }
        double t_direct = phase_profile ? conv_phase_now_ms() : 0.0;
        int drc = conv3x3_direct_avx2(x, w, bias, y, N_batch, C_in, H, W,
                                      C_out, outH, outW, 2,
                                      attr.fused_activation);
        if (phase_profile) conv_phase_add(CONV_PHASE_DIRECT, t_direct);
        return drc;
    }

    /* ── Lazy weight pre-packing (via node_cache) ── */
    /* Only for group==1 standard convs to keep it simple */
    float *packed_w = NULL;
    if (group == 1 && ctx->node_cache && node->id < ctx->node_cache_count) {
        if (!ctx->node_cache[node->id]) {
            double t_pack = phase_profile ? conv_phase_now_ms() : 0.0;
            ctx->node_cache[node->id] = sgemm_pack_a_impl(C_out_g, K_g, w, K_g);
            if (phase_profile) conv_phase_add(CONV_PHASE_PACK, t_pack);
        }
        packed_w = (float *)ctx->node_cache[node->id];
    }

    /* ── 1×1 conv fast path: skip im2col, SGEMM directly on input ── */
    if (is_1x1_s1_p0) {
        for (int n = 0; n < N_batch; n++) {
            float *y_n = y + (size_t)n * C_out * spatial;

            if (group == 1 && packed_w) {
                const float *x_n = x + (size_t)n * C_in * H * W;
                double t_sgemm = phase_profile ? conv_phase_now_ms() : 0.0;
                sgemm_nn_packed_a_epilogue(C_out, spatial, C_in,
                                           packed_w, x_n, spatial, y_n, spatial,
                                           bias, attr.fused_activation, 1);
                if (phase_profile) conv_phase_add(CONV_PHASE_SGEMM_EPILOGUE, t_sgemm);
                continue;
            }

            for (int g = 0; g < group; g++) {
                const float *x_g = x + (size_t)n * C_in * H * W + (size_t)g * C_in_g * H * W;
                const float *w_g = w + (size_t)g * C_out_g * K_g;
                const float *bias_g = bias ? bias + g * C_out_g : NULL;
                float *y_g = y_n + (size_t)g * C_out_g * spatial;

                /* init Y with bias */
                for (int m = 0; m < C_out_g; m++) {
                    float bv = bias_g ? bias_g[m] : 0.0f;
                    double t_bias = phase_profile ? conv_phase_now_ms() : 0.0;
                    bias_init_row(y_g + (size_t)m * spatial, bv, spatial);
                    if (phase_profile) conv_phase_add(CONV_PHASE_BIAS, t_bias);
                }

                double t_sgemm = phase_profile ? conv_phase_now_ms() : 0.0;
                if (packed_w && g == 0) {
                    sgemm_nn_packed_a(C_out_g, spatial, C_in_g,
                                       packed_w, x_g, spatial, y_g, spatial);
                } else {
                    sgemm_nn(C_out_g, spatial, C_in_g,
                             w_g, C_in_g, x_g, spatial, y_g, spatial);
                }
                if (phase_profile) conv_phase_add(CONV_PHASE_SGEMM, t_sgemm);
            }

            double t_epilogue = phase_profile ? conv_phase_now_ms() : 0.0;
            fused_activation_pass(y_n, (size_t)C_out * spatial, attr.fused_activation);
            if (phase_profile) conv_phase_add(CONV_PHASE_EPILOGUE, t_epilogue);
        }
        return 0;
    }

    /* ── General conv: tiled im2col + SGEMM per group ── */
    int scratch_floats = node->scratch_bytes > 0 ? (int)(node->scratch_bytes / sizeof(float)) : 0;
    int tile_n = K_g > 0 ? scratch_floats / K_g : 0;
    if (tile_n <= 0) return -13;
    if (tile_n > SGEMM_NC) tile_n = SGEMM_NC;
    if (tile_n > spatial) tile_n = spatial;

    for (int n = 0; n < N_batch; n++) {
        float *y_n = y + (size_t)n * C_out * spatial;
        int use_epilogue = (group == 1 && packed_w != NULL);

        for (int g = 0; g < group; g++) {
            const float *x_g = x + (size_t)n * C_in * H * W + (size_t)g * C_in_g * H * W;
            const float *w_g = w + (size_t)g * C_out_g * K_g;
            const float *bias_g = bias ? bias + g * C_out_g : NULL;
            float *y_g = y_n + (size_t)g * C_out_g * spatial;

            for (int start = 0; start < spatial; start += tile_n) {
                int count = SIMD_MIN(tile_n, spatial - start);

                if (!use_epilogue) {
                    for (int m = 0; m < C_out_g; m++) {
                        float bv = bias_g ? bias_g[m] : 0.0f;
                        double t_bias = phase_profile ? conv_phase_now_ms() : 0.0;
                        bias_init_row(y_g + (size_t)m * spatial + start, bv, count);
                        if (phase_profile) conv_phase_add(CONV_PHASE_BIAS, t_bias);
                    }
                }

                double t_im2col = phase_profile ? conv_phase_now_ms() : 0.0;
                if (is_3x3_s1_p1_d1) {
                    im2col_3x3_s1p1_tile(x_g, C_in_g, H, W, start, count, col);
                } else if (is_3x3_s2_p1_d1) {
                    im2col_3x3_s2p1_tile(x_g, C_in_g, H, W, outW,
                                         start, count, col);
                } else {
                    im2col_full_tile(x_g, C_in_g, H, W, kH, kW,
                                     attr.strides[0], attr.strides[1],
                                     attr.pads[0], attr.pads[1],
                                     attr.dilations[0], attr.dilations[1],
                                     outW, start, count, col);
                }
                if (phase_profile) conv_phase_add(CONV_PHASE_IM2COL, t_im2col);

                if (use_epilogue) {
                    double t_sgemm = phase_profile ? conv_phase_now_ms() : 0.0;
                    sgemm_nn_packed_a_epilogue(C_out_g, count, K_g,
                                               packed_w, col, count,
                                               y_g + start, spatial,
                                               bias_g, attr.fused_activation, 1);
                    if (phase_profile) conv_phase_add(CONV_PHASE_SGEMM_EPILOGUE, t_sgemm);
                } else {
                    double t_sgemm = phase_profile ? conv_phase_now_ms() : 0.0;
                    sgemm_nn(C_out_g, count, K_g,
                             w_g, K_g, col, count,
                             y_g + start, spatial);
                    if (phase_profile) conv_phase_add(CONV_PHASE_SGEMM, t_sgemm);
                }
            }
        }

        /* fused activation */
        if (!use_epilogue) {
            double t_epilogue = phase_profile ? conv_phase_now_ms() : 0.0;
            fused_activation_pass(y_n, (size_t)C_out * spatial, attr.fused_activation);
            if (phase_profile) conv_phase_add(CONV_PHASE_EPILOGUE, t_epilogue);
        }
    }
    return 0;
}


#endif /* __AVX2__ */
