#include "test_helpers.h"
#include "reference_kernels.h"

static int test_relu_basic(void) {
    TestContext tc;
    Spkv2AttrRecord attr;
    init_attr_record(&attr, SPKV2_OP_RELU);
    init_test_context(&tc, 2, &attr, sizeof(attr));

    uint32_t shape[] = {1, 1, 1, 8};
    init_tensor_record(&tc.tensor_records[0], 0, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, shape, 8 * sizeof(float));
    init_tensor_record(&tc.tensor_records[1], 1, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, shape, 8 * sizeof(float));

    float x[] = {-3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 100.0f};
    float y[8] = {0};
    set_tensor_data(&tc, 0, x);
    set_tensor_data(&tc, 1, y);

    Spkv2NodeRecord node;
    uint32_t inputs[] = {0};
    uint32_t outputs[] = {1};
    init_node_record(&node, 0, SPKV2_OP_RELU, 1, inputs, 1, outputs);

    ASSERT_OK(kernel_relu(&tc.ctx, &node, NULL));

    ASSERT_NEAR(y[0], 0.0f, 1e-7f);
    ASSERT_NEAR(y[1], 0.0f, 1e-7f);
    ASSERT_NEAR(y[2], 0.0f, 1e-7f);
    ASSERT_NEAR(y[3], 0.0f, 1e-7f);
    ASSERT_NEAR(y[4], 0.5f, 1e-7f);
    ASSERT_NEAR(y[5], 1.0f, 1e-7f);
    ASSERT_NEAR(y[6], 2.0f, 1e-7f);
    ASSERT_NEAR(y[7], 100.0f, 1e-7f);
    return 0;
}

static int test_relu_large(void) {
    enum { N = 1024 };
    TestContext tc;
    Spkv2AttrRecord attr;
    init_attr_record(&attr, SPKV2_OP_RELU);
    init_test_context(&tc, 2, &attr, sizeof(attr));

    uint32_t shape[] = {1, 1, 1, N};
    init_tensor_record(&tc.tensor_records[0], 0, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, shape, N * sizeof(float));
    init_tensor_record(&tc.tensor_records[1], 1, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, shape, N * sizeof(float));

    float x[N], y[N];
    for (int i = 0; i < N; i++)
        x[i] = (float)(i - N / 2) * 0.01f;
    memset(y, 0, sizeof(y));
    set_tensor_data(&tc, 0, x);
    set_tensor_data(&tc, 1, y);

    Spkv2NodeRecord node;
    uint32_t inputs[] = {0};
    uint32_t outputs[] = {1};
    init_node_record(&node, 0, SPKV2_OP_RELU, 1, inputs, 1, outputs);

    ASSERT_OK(kernel_relu(&tc.ctx, &node, NULL));

    for (int i = 0; i < N; i++) {
        float expected = x[i] > 0.0f ? x[i] : 0.0f;
        ASSERT_NEAR(y[i], expected, 1e-7f);
    }
    return 0;
}

int main(void) {
    printf("test_kernel_relu:\n");
    RUN_TEST(test_relu_basic);
    RUN_TEST(test_relu_large);
    TEST_SUMMARY();
}
