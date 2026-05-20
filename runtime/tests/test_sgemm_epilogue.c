#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern float *sgemm_pack_a_impl(int M, int K, const float *A, int lda);
extern void sgemm_nn_packed_a_epilogue(
    int M, int N, int K,
    const float *packed_a,
    const float *B, int ldb,
    float *C, int ldc,
    const float *bias,
    int act_type,
    int allow_parallel);
extern int sgemm_should_parallelize(int M, int N, int K, int allow_parallel);

static int close_enough(float a, float b) {
    float d = fabsf(a - b);
    return d < 1e-5f;
}

static float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static int test_edge_m_full_n_epilogue(void) {
    const int M = 5;
    const int N = 16;
    const int K = 3;
    float A[M * K];
    float B[K * N];
    float bias[M];
    float C[M * N];
    float expected[M * N];

    for (int i = 0; i < M * K; i++)
        A[i] = ((float)(i % 7) - 3.0f) * 0.25f;
    for (int i = 0; i < K * N; i++)
        B[i] = ((float)(i % 11) - 5.0f) * 0.125f;
    for (int m = 0; m < M; m++)
        bias[m] = (float)m * 0.1f - 0.2f;
    for (int i = 0; i < M * N; i++)
        C[i] = -999.0f;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = bias[m];
            for (int k = 0; k < K; k++)
                acc += A[m * K + k] * B[k * N + n];
            expected[m * N + n] = relu(acc);
        }
    }

    float *packed_a = sgemm_pack_a_impl(M, K, A, K);
    if (!packed_a) return 0;
    sgemm_nn_packed_a_epilogue(M, N, K, packed_a, B, N, C, N, bias, 1, 0);
    free(packed_a);

    for (int i = 0; i < M * N; i++) {
        if (!close_enough(C[i], expected[i])) {
            fprintf(stderr, "edge C[%d]=%.8f expected %.8f\n", i, C[i], expected[i]);
            return 0;
        }
    }
    return 1;
}

static int test_edge_m_tail_n_accumulating_epilogue(void) {
    const int M = 5;
    const int N = 9;
    const int K = 300;
    float *A = (float *)malloc((size_t)M * K * sizeof(float));
    float *B = (float *)malloc((size_t)K * N * sizeof(float));
    float *C = (float *)malloc((size_t)M * N * sizeof(float));
    float *expected = (float *)malloc((size_t)M * N * sizeof(float));
    float bias[M];
    if (!A || !B || !C || !expected) {
        free(A); free(B); free(C); free(expected);
        return 0;
    }

    for (int i = 0; i < M * K; i++)
        A[i] = ((float)(i % 13) - 6.0f) * 0.03125f;
    for (int i = 0; i < K * N; i++)
        B[i] = ((float)(i % 17) - 8.0f) * 0.015625f;
    for (int m = 0; m < M; m++)
        bias[m] = (float)m * 0.05f - 0.1f;
    for (int i = 0; i < M * N; i++)
        C[i] = -777.0f;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = bias[m];
            for (int k = 0; k < K; k++)
                acc += A[m * K + k] * B[k * N + n];
            expected[m * N + n] = relu(acc);
        }
    }

    float *packed_a = sgemm_pack_a_impl(M, K, A, K);
    if (!packed_a) {
        free(A); free(B); free(C); free(expected);
        return 0;
    }
    sgemm_nn_packed_a_epilogue(M, N, K, packed_a, B, N, C, N, bias, 1, 0);
    free(packed_a);

    int ok = 1;
    for (int i = 0; i < M * N; i++) {
        if (!close_enough(C[i], expected[i])) {
            fprintf(stderr, "tail C[%d]=%.8f expected %.8f\n", i, C[i], expected[i]);
            ok = 0;
            break;
        }
    }

    free(A); free(B); free(C); free(expected);
    return ok;
}

int main(void) {
    const int M = 2;
    const int N = 3;
    const int K = 2;
    const float A[4] = {
        1.0f, -2.0f,
        3.0f,  4.0f,
    };
    const float B[6] = {
        2.0f, -1.0f, 0.5f,
        1.0f,  2.0f, -3.0f,
    };
    const float bias[2] = {-1.0f, 0.5f};
    const float expected[6] = {
        0.0f, 0.0f, 5.5f,
        10.5f, 5.5f, 0.0f,
    };
    float C[6] = {123.0f, 123.0f, 123.0f, 123.0f, 123.0f, 123.0f};

    float *packed_a = sgemm_pack_a_impl(M, K, A, K);
    if (!packed_a) {
        fprintf(stderr, "pack failed\n");
        return 1;
    }

    sgemm_nn_packed_a_epilogue(M, N, K, packed_a, B, N, C, N, bias, 1, 0);
    free(packed_a);

    for (int i = 0; i < M * N; i++) {
        if (!close_enough(C[i], expected[i])) {
            fprintf(stderr, "C[%d]=%.8f expected %.8f\n", i, C[i], expected[i]);
            return 1;
        }
    }

    if (!test_edge_m_full_n_epilogue()) {
        fprintf(stderr, "edge M<MR,N=NR epilogue failed\n");
        return 1;
    }
    if (!test_edge_m_tail_n_accumulating_epilogue()) {
        fprintf(stderr, "edge M<MR,N<NR accumulating epilogue failed\n");
        return 1;
    }

    if (sgemm_should_parallelize(24, 16, 8, 1) != 0) {
        fprintf(stderr, "tiny GEMM should stay single-threaded\n");
        return 1;
    }
#ifdef _OPENMP
    if (sgemm_should_parallelize(64, 512, 256, 1) == 0) {
        fprintf(stderr, "large GEMM should allow parallelism\n");
        return 1;
    }
#else
    if (sgemm_should_parallelize(64, 512, 256, 1) != 0) {
        fprintf(stderr, "non-OpenMP build should stay single-threaded\n");
        return 1;
    }
#endif
    if (sgemm_should_parallelize(64, 512, 256, 0) != 0) {
        fprintf(stderr, "allow_parallel=0 should force single-threaded\n");
        return 1;
    }

    return 0;
}
