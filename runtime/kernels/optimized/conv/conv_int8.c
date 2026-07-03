/*
 * INT8 Conv kernel: mixed-precision im2col + INT8 GEMM.
 *
 * Input/output activations are FP32. Conv weights are INT8 (per-channel symmetric).
 * Activations are quantized on-the-fly using a calibrated per-tensor scale.
 * INT8 x INT8 -> INT32 accumulate -> dequantize to FP32 + bias + activation.
 *
 * Only compiled on ARM with dot product support (Apple Silicon M1+).
 */

#include "simd_kernels.h"

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

#include "mm/int8_gemm.h"

#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void quantize_f32_to_i8(const float *src, int8_t *dst, size_t count, float scale) {
    float inv_scale = 1.0f / scale;
    size_t i = 0;
    float32x4_t v_inv = vdupq_n_f32(inv_scale);

    for (; i + 15 < count; i += 16) {
        float32x4_t f0 = vmulq_f32(vld1q_f32(src + i),      v_inv);
        float32x4_t f1 = vmulq_f32(vld1q_f32(src + i + 4),  v_inv);
        float32x4_t f2 = vmulq_f32(vld1q_f32(src + i + 8),  v_inv);
        float32x4_t f3 = vmulq_f32(vld1q_f32(src + i + 12), v_inv);

        int32x4_t i0 = vcvtnq_s32_f32(f0);
        int32x4_t i1 = vcvtnq_s32_f32(f1);
        int32x4_t i2 = vcvtnq_s32_f32(f2);
        int32x4_t i3 = vcvtnq_s32_f32(f3);

        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int16x4_t s2 = vqmovn_s32(i2);
        int16x4_t s3 = vqmovn_s32(i3);

        int16x8_t h01 = vcombine_s16(s0, s1);
        int16x8_t h23 = vcombine_s16(s2, s3);

        int8x8_t b01 = vqmovn_s16(h01);
        int8x8_t b23 = vqmovn_s16(h23);

        vst1_s8(dst + i, b01);
        vst1_s8(dst + i + 8, b23);
    }
    for (; i < count; i++) {
        int v = (int)roundf(src[i] * inv_scale);
        dst[i] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
    }
}

static void im2col_int8(const int8_t *im, int C, int H, int W,
                        int kH, int kW, int sH, int sW, int pH, int pW,
                        int dH, int dW, int outH, int outW, int8_t *col) {
    int ohw = outH * outW;
    int row = 0;
    for (int c = 0; c < C; c++) {
        const int8_t *xc = im + (size_t)c * H * W;
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int8_t *dst = col + (size_t)row * ohw;
                for (int oh = 0; oh < outH; oh++) {
                    int ih = oh * sH - pH + kh * dH;
                    if (ih < 0 || ih >= H) {
                        memset(dst + oh * outW, 0, (size_t)outW);
                    } else {
                        const int8_t *xr = xc + ih * W;
                        for (int ow = 0; ow < outW; ow++) {
                            int iw = ow * sW - pW + kw * dW;
                            dst[oh * outW + ow] = (iw >= 0 && iw < W) ? xr[iw] : 0;
                        }
                    }
                }
                row++;
            }
        }
    }
}

static void dequantize_bias_act(const int32_t *acc, float *y,
                                int C_out, int spatial,
                                const float *w_scales, float act_scale,
                                const float *bias, const float *residual,
                                int fused_activation) {
    for (int c = 0; c < C_out; c++) {
        float combined_scale = w_scales[c] * act_scale;
        float bv = bias ? bias[c] : 0.0f;
        float *y_c = y + (size_t)c * spatial;
        const int32_t *acc_c = acc + (size_t)c * spatial;
        const float *res_c = residual ? residual + (size_t)c * spatial : NULL;

        int j = 0;
        float32x4_t v_scale = vdupq_n_f32(combined_scale);
        float32x4_t v_bias = vdupq_n_f32(bv);

        for (; j + 3 < spatial; j += 4) {
            int32x4_t vi = vld1q_s32(acc_c + j);
            float32x4_t vf = vcvtq_f32_s32(vi);
            vf = vfmaq_f32(v_bias, vf, v_scale);
            if (res_c) {
                vf = vaddq_f32(vf, vld1q_f32(res_c + j));
            }
            vf = apply_activation_neon(vf, fused_activation);
            vst1q_f32(y_c + j, vf);
        }
        for (; j < spatial; j++) {
            float val = (float)acc_c[j] * combined_scale + bv;
            if (res_c) val += res_c[j];
            val = apply_activation_scalar_simd(val, fused_activation);
            y_c[j] = val;
        }
    }
}

int kernel_conv_int8(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    Spkv2AttrRecord attr;
    int rc = simd_get_attr(ctx, node, &attr);
    if (rc != 0) return rc;
    if (!scratch && node->scratch_bytes > 0) return -13;

    const Spkv2TensorRecord *x_rec = ctx->tensors[node->inputs[0]].record;
    const Spkv2TensorRecord *w_rec = ctx->tensors[node->inputs[1]].record;
    const Spkv2TensorRecord *y_rec = ctx->tensors[node->outputs[0]].record;

    const float *x_fp32 = (const float *)ctx->tensors[node->inputs[0]].data;
    const int8_t *w_int8 = (const int8_t *)ctx->tensors[node->inputs[1]].data;
    const float *bias = (node->input_count > 2)
        ? (const float *)ctx->tensors[node->inputs[2]].data : NULL;
    const float *residual = (node->input_count >= 4)
        ? (const float *)ctx->tensors[node->inputs[3]].data : NULL;
    float *y_fp32 = (float *)ctx->tensors[node->outputs[0]].data;

    int N_batch = (int)x_rec->shape[0];
    int C_in  = (int)x_rec->shape[1];
    int H     = (int)x_rec->shape[2];
    int W     = (int)x_rec->shape[3];
    int C_out = (int)w_rec->shape[0];
    int kH    = (int)w_rec->shape[2];
    int kW    = (int)w_rec->shape[3];
    int outH  = (int)y_rec->shape[2];
    int outW  = (int)y_rec->shape[3];
    int spatial = outH * outW;
    int K = C_in * kH * kW;

    const Spkv2QuantParamRecord *w_qp = spkv2_find_quant_param(ctx, node->inputs[1]);
    const Spkv2QuantParamRecord *x_qp = spkv2_find_quant_param(ctx, node->inputs[0]);
    if (!w_qp || !x_qp) return -99;

    const float *w_scales = spkv2_quant_scales(ctx, w_qp);
    const float *x_scale_ptr = spkv2_quant_scales(ctx, x_qp);
    if (!w_scales || !x_scale_ptr) return -99;
    float x_scale = x_scale_ptr[0];

    /* Scratch layout: [int8 act | int8 col | int32 acc] */
    size_t act_i8_size = (size_t)C_in * H * W;
    size_t col_i8_size = (size_t)K * spatial;
    size_t acc_i32_bytes = (size_t)C_out * spatial * sizeof(int32_t);

    int8_t *x_int8 = (int8_t *)scratch;
    int8_t *col_int8 = x_int8 + act_i8_size;
    int32_t *acc_int32 = (int32_t *)(col_int8 + col_i8_size);

    /* Lazy weight packing via node_cache */
    const int8_t *packed_w = NULL;
    if (ctx->node_cache && node->id < (uint32_t)ctx->node_cache_count) {
        if (!ctx->node_cache[node->id]) {
            ctx->node_cache[node->id] = i8gemm_pack_a(C_out, K, w_int8, K);
        }
        packed_w = (const int8_t *)ctx->node_cache[node->id];
    }

    for (int n = 0; n < N_batch; n++) {
        const float *xn = x_fp32 + (size_t)n * C_in * H * W;
        float *yn = y_fp32 + (size_t)n * C_out * spatial;
        const float *res_n = residual ? residual + (size_t)n * C_out * spatial : NULL;

        quantize_f32_to_i8(xn, x_int8, act_i8_size, x_scale);

        im2col_int8(x_int8, C_in, H, W, kH, kW,
                    attr.strides[0], attr.strides[1],
                    attr.pads[0], attr.pads[1],
                    attr.dilations[0], attr.dilations[1],
                    outH, outW, col_int8);

        memset(acc_int32, 0, acc_i32_bytes);

        if (packed_w) {
            i8gemm_nn_packed_a(C_out, spatial, K,
                               packed_w, col_int8, spatial,
                               acc_int32, spatial);
        } else {
            for (int m = 0; m < C_out; m++)
                for (int j = 0; j < spatial; j++)
                    for (int k = 0; k < K; k++)
                        acc_int32[m * spatial + j] +=
                            (int32_t)w_int8[m * K + k] * (int32_t)col_int8[k * spatial + j];
        }

        dequantize_bias_act(acc_int32, yn, C_out, spatial,
                            w_scales, x_scale, bias, res_n,
                            attr.fused_activation);
    }
    return 0;
}

#endif /* __ARM_NEON && __ARM_FEATURE_DOTPROD */
