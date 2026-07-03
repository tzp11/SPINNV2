/*
 * bench_bnns_int8.c — BNNS INT8 conv benchmark vs NEON INT8 vs cblas FP32 (AMX)
 *
 * Built via CMake (add_executable in runtime/CMakeLists.txt).
 * Links spkv2_runtime which brings in -framework Accelerate.
 *
 *   Usage: ./spkv2_bench_bnns
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static double now_ms(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)mach_absolute_time() * info.numer / info.denom / 1e6;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

/* ─── cblas_sgemm FP32 (AMX on Apple Silicon) ─── */
static double bench_cblas_fp32(int M, int N, int K, int runs) {
#ifdef __APPLE__
    float *A = calloc((size_t)M * K, sizeof(float));
    float *B = calloc((size_t)K * N, sizeof(float));
    float *C = calloc((size_t)M * N, sizeof(float));
    for (int i = 0; i < M * K; i++) A[i] = 0.01f * (i % 200 - 100);
    for (int i = 0; i < K * N; i++) B[i] = 0.01f * (i % 200 - 100);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    double t0 = now_ms();
    for (int r = 0; r < runs; r++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    double elapsed = (now_ms() - t0) / runs;
    free(A); free(B); free(C);
    return elapsed;
#else
    (void)M; (void)N; (void)K; (void)runs;
    return -1;
#endif
}

/* ─── BNNS FP32 Convolution ─── */
#ifdef __APPLE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

static double bench_bnns_fp32_conv(int Ci, int Co, int H, int W,
                                    int kH, int kW, int sH, int sW,
                                    int pH, int pW, int runs) {
    int oH = (H + 2*pH - kH) / sH + 1;
    int oW = (W + 2*pW - kW) / sW + 1;

    float *input  = calloc((size_t)Ci * H * W, sizeof(float));
    float *output = calloc((size_t)Co * oH * oW, sizeof(float));
    float *weight = calloc((size_t)Co * Ci * kH * kW, sizeof(float));
    float *bias   = calloc((size_t)Co, sizeof(float));
    for (int i = 0; i < Ci * H * W; i++) input[i] = 0.01f * (i % 200 - 100);
    for (int i = 0; i < Co * Ci * kH * kW; i++) weight[i] = 0.01f * (i % 200 - 100);

    BNNSLayerParametersConvolution p;
    memset(&p, 0, sizeof(p));
    p.i_desc.layout = BNNSDataLayoutImageCHW;
    p.i_desc.size[0] = (size_t)W; p.i_desc.size[1] = (size_t)H; p.i_desc.size[2] = (size_t)Ci;
    p.i_desc.data_type = BNNSDataTypeFloat32;
    p.w_desc.layout = BNNSDataLayoutConvolutionWeightsOIHW;
    p.w_desc.size[0] = (size_t)kW; p.w_desc.size[1] = (size_t)kH;
    p.w_desc.size[2] = (size_t)Ci; p.w_desc.size[3] = (size_t)Co;
    p.w_desc.data_type = BNNSDataTypeFloat32;
    p.w_desc.data = weight;
    p.o_desc.layout = BNNSDataLayoutImageCHW;
    p.o_desc.size[0] = (size_t)oW; p.o_desc.size[1] = (size_t)oH; p.o_desc.size[2] = (size_t)Co;
    p.o_desc.data_type = BNNSDataTypeFloat32;
    p.bias.layout = BNNSDataLayoutVector;
    p.bias.size[0] = (size_t)Co;
    p.bias.data = bias; p.bias.data_type = BNNSDataTypeFloat32;
    p.x_stride = (size_t)sW; p.y_stride = (size_t)sH;
    p.x_padding = (size_t)pW; p.y_padding = (size_t)pH;
    p.activation.function = BNNSActivationFunctionIdentity;

    BNNSFilterParameters fp;
    memset(&fp, 0, sizeof(fp));
    BNNSFilter f = BNNSFilterCreateLayerConvolution(&p, &fp);
    if (!f) {
        fprintf(stderr, "BNNS FP32 filter FAIL: Ci=%d Co=%d %dx%d\n", Ci, Co, kH, kW);
        free(input); free(output); free(weight); free(bias);
        return -1;
    }

    BNNSFilterApply(f, input, output);
    double t0 = now_ms();
    for (int r = 0; r < runs; r++) BNNSFilterApply(f, input, output);
    double elapsed = (now_ms() - t0) / runs;

    BNNSFilterDestroy(f);
    free(input); free(output); free(weight); free(bias);
    return elapsed;
}

/* ─── BNNS INT8 Convolution ─── */
static double bench_bnns_int8_conv(int Ci, int Co, int H, int W,
                                    int kH, int kW, int sH, int sW,
                                    int pH, int pW, int runs) {
    int oH = (H + 2*pH - kH) / sH + 1;
    int oW = (W + 2*pW - kW) / sW + 1;

    int8_t *input  = calloc((size_t)Ci * H * W, 1);
    float  *output = calloc((size_t)Co * oH * oW, sizeof(float));
    int8_t *weight = calloc((size_t)Co * Ci * kH * kW, 1);
    float  *bias   = calloc((size_t)Co, sizeof(float));
    for (int i = 0; i < Ci * H * W; i++) input[i] = (int8_t)(i % 127 - 63);
    for (int i = 0; i < Co * Ci * kH * kW; i++) weight[i] = (int8_t)(i % 127 - 63);

    BNNSLayerParametersConvolution p;
    memset(&p, 0, sizeof(p));
    p.i_desc.layout = BNNSDataLayoutImageCHW;
    p.i_desc.size[0] = (size_t)W; p.i_desc.size[1] = (size_t)H; p.i_desc.size[2] = (size_t)Ci;
    p.i_desc.data_type = BNNSDataTypeInt8;
    p.i_desc.data_scale = 1.0f / 127.0f;
    p.w_desc.layout = BNNSDataLayoutConvolutionWeightsOIHW;
    p.w_desc.size[0] = (size_t)kW; p.w_desc.size[1] = (size_t)kH;
    p.w_desc.size[2] = (size_t)Ci; p.w_desc.size[3] = (size_t)Co;
    p.w_desc.data_type = BNNSDataTypeInt8;
    p.w_desc.data_scale = 1.0f / 127.0f;
    p.w_desc.data = weight;
    p.o_desc.layout = BNNSDataLayoutImageCHW;
    p.o_desc.size[0] = (size_t)oW; p.o_desc.size[1] = (size_t)oH; p.o_desc.size[2] = (size_t)Co;
    p.o_desc.data_type = BNNSDataTypeFloat32;
    p.bias.layout = BNNSDataLayoutVector;
    p.bias.size[0] = (size_t)Co;
    p.bias.data = bias; p.bias.data_type = BNNSDataTypeFloat32;
    p.x_stride = (size_t)sW; p.y_stride = (size_t)sH;
    p.x_padding = (size_t)pW; p.y_padding = (size_t)pH;
    p.activation.function = BNNSActivationFunctionIdentity;

    BNNSFilterParameters fp;
    memset(&fp, 0, sizeof(fp));
    BNNSFilter f = BNNSFilterCreateLayerConvolution(&p, &fp);
    if (!f) {
        fprintf(stderr, "BNNS INT8 filter FAIL: Ci=%d Co=%d %dx%d\n", Ci, Co, kH, kW);
        free(input); free(output); free(weight); free(bias);
        return -1;
    }

    BNNSFilterApply(f, input, output);
    double t0 = now_ms();
    for (int r = 0; r < runs; r++) BNNSFilterApply(f, input, output);
    double elapsed = (now_ms() - t0) / runs;

    BNNSFilterDestroy(f);
    free(input); free(output); free(weight); free(bias);
    return elapsed;
}

#pragma clang diagnostic pop
#endif /* __APPLE__ */

/* ─── NEON vdotq INT8 GEMM (mirrors our runtime micro-kernel) ─── */
static double bench_neon_i8_gemm(int M, int N, int K, int runs) {
    int8_t  *A   = calloc((size_t)M * K, 1);
    int8_t  *B   = calloc((size_t)K * N, 1);
    int32_t *C   = calloc((size_t)M * N, sizeof(int32_t));
    for (int i = 0; i < M * K; i++) A[i] = (int8_t)(i % 127 - 63);
    for (int i = 0; i < K * N; i++) B[i] = (int8_t)(i % 127 - 63);

    /* warmup */
    memset(C, 0, (size_t)M * N * sizeof(int32_t));
#ifdef __ARM_NEON
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32x4_t acc = vdupq_n_s32(0);
            int k = 0;
            for (; k + 15 < K; k += 16) {
                acc = vdotq_s32(acc, vld1q_s8(A + m*K + k), vld1q_s8(B + n*K + k));
            }
            int32_t s = vaddvq_s32(acc);
            for (; k < K; k++) s += (int32_t)A[m*K+k] * (int32_t)B[n*K+k];
            C[m*N+n] = s;
        }
#endif

    double t0 = now_ms();
    for (int r = 0; r < runs; r++) {
        memset(C, 0, (size_t)M * N * sizeof(int32_t));
#ifdef __ARM_NEON
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                int32x4_t acc = vdupq_n_s32(0);
                int k = 0;
                for (; k + 15 < K; k += 16) {
                    acc = vdotq_s32(acc, vld1q_s8(A + m*K + k), vld1q_s8(B + n*K + k));
                }
                int32_t s = vaddvq_s32(acc);
                for (; k < K; k++) s += (int32_t)A[m*K+k] * (int32_t)B[n*K+k];
                C[m*N+n] = s;
            }
#endif
    }
    double elapsed = (now_ms() - t0) / runs;
    free(A); free(B); free(C);
    return elapsed;
}

typedef struct {
    const char *label;
    int Ci, Co, H, W, kH, kW, sH, sW, pH, pW;
} BenchConv;

int main(void) {
    int runs = 20;

    BenchConv cfgs[] = {
        {"1x1 s1 64->256  56x56",   64, 256, 56, 56, 1, 1, 1, 1, 0, 0},
        {"3x3 s1 64->64   56x56",   64,  64, 56, 56, 3, 3, 1, 1, 1, 1},
        {"3x3 s2 128->256 28x28",  128, 256, 28, 28, 3, 3, 2, 2, 1, 1},
        {"1x1 s1 512->128  7x7",   512, 128,  7,  7, 1, 1, 1, 1, 0, 0},
        {"3x3 s1 32->64   80x80",   32,  64, 80, 80, 3, 3, 1, 1, 1, 1},
        {"1x1 s1 128->64  40x40",  128,  64, 40, 40, 1, 1, 1, 1, 0, 0},
    };
    int ncfg = sizeof(cfgs) / sizeof(cfgs[0]);

    printf("BNNS INT8 Convolution Exploration\n");
    printf("==================================\n\n");
    printf("%-28s %10s %10s %10s %10s\n",
           "Config", "cblas+AMX", "BNNS FP32", "BNNS INT8", "NEON i8");
    printf("%-28s %10s %10s %10s %10s\n", "", "(ms)", "(ms)", "(ms)", "(ms)");
    printf("────────────────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < ncfg; i++) {
        BenchConv *c = &cfgs[i];
        int oH = (c->H + 2*c->pH - c->kH) / c->sH + 1;
        int oW = (c->W + 2*c->pW - c->kW) / c->sW + 1;
        int K  = c->Ci * c->kH * c->kW;
        int spatial = oH * oW;

        double t_cblas = bench_cblas_fp32(c->Co, spatial, K, runs);
        double t_neon  = bench_neon_i8_gemm(c->Co, spatial, K, runs);

#ifdef __APPLE__
        double t_bnns_fp = bench_bnns_fp32_conv(c->Ci, c->Co, c->H, c->W,
                                                 c->kH, c->kW, c->sH, c->sW,
                                                 c->pH, c->pW, runs);
        double t_bnns_i8 = bench_bnns_int8_conv(c->Ci, c->Co, c->H, c->W,
                                                  c->kH, c->kW, c->sH, c->sW,
                                                  c->pH, c->pW, runs);
#else
        double t_bnns_fp = -1, t_bnns_i8 = -1;
#endif

        printf("%-28s", c->label);
        printf(" %8.3f ms", t_cblas);
        if (t_bnns_fp >= 0) printf(" %8.3f ms", t_bnns_fp);
        else printf(" %10s", "FAIL");
        if (t_bnns_i8 >= 0) printf(" %8.3f ms", t_bnns_i8);
        else printf(" %10s", "FAIL");
        printf(" %8.3f ms", t_neon);
        printf("\n");
    }

    printf("\nLegend:\n");
    printf("  cblas+AMX  = cblas_sgemm GEMM only (AMX-accelerated)\n");
    printf("  BNNS FP32  = BNNSFilterCreateLayerConvolution FP32 (full conv)\n");
    printf("  BNNS INT8  = BNNSFilterCreateLayerConvolution INT8 in/weight, FP32 out\n");
    printf("  NEON i8    = vdotq_s32 INT8 GEMM only (our current kernel)\n");

    return 0;
}
