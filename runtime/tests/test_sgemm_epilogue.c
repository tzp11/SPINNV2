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
