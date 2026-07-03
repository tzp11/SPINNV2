#include "test_helpers.h"
#include "reference_kernels.h"

static int test_softmax_basic(void) {
    TestContext tc;
    Spkv2AttrRecord attr;
    init_attr_record(&attr, SPKV2_OP_SOFTMAX);
    attr.axis = -1;
    init_test_context(&tc, 2, &attr, sizeof(attr));

    uint32_t shape[] = {1, 4};
    init_tensor_record(&tc.tensor_records[0], 0, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 2, shape, 4 * sizeof(float));
    init_tensor_record(&tc.tensor_records[1], 1, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 2, shape, 4 * sizeof(float));

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[4] = {0};
    set_tensor_data(&tc, 0, x);
    set_tensor_data(&tc, 1, y);

    Spkv2NodeRecord node;
    uint32_t inputs[] = {0};
    uint32_t outputs[] = {1};
    init_node_record(&node, 0, SPKV2_OP_SOFTMAX, 1, inputs, 1, outputs);
    node.attr_offset = 0;
    node.attr_size = sizeof(Spkv2AttrRecord);

    ASSERT_OK(kernel_softmax(&tc.ctx, &node, NULL));

    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(y[i] > 0.0f);
        ASSERT_TRUE(y[i] < 1.0f);
        sum += y[i];
    }
    ASSERT_NEAR(sum, 1.0f, 1e-5f);

    ASSERT_TRUE(y[3] > y[2]);
    ASSERT_TRUE(y[2] > y[1]);
    ASSERT_TRUE(y[1] > y[0]);
    return 0;
}

static int test_softmax_large_values(void) {
    TestContext tc;
    Spkv2AttrRecord attr;
    init_attr_record(&attr, SPKV2_OP_SOFTMAX);
    attr.axis = -1;
    init_test_context(&tc, 2, &attr, sizeof(attr));

    uint32_t shape[] = {1, 4};
    init_tensor_record(&tc.tensor_records[0], 0, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 2, shape, 4 * sizeof(float));
    init_tensor_record(&tc.tensor_records[1], 1, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 2, shape, 4 * sizeof(float));

    float x[] = {1000.0f, 1001.0f, 1002.0f, 1003.0f};
    float y[4] = {0};
    set_tensor_data(&tc, 0, x);
    set_tensor_data(&tc, 1, y);

    Spkv2NodeRecord node;
    uint32_t inputs[] = {0};
    uint32_t outputs[] = {1};
    init_node_record(&node, 0, SPKV2_OP_SOFTMAX, 1, inputs, 1, outputs);
    node.attr_offset = 0;
    node.attr_size = sizeof(Spkv2AttrRecord);

    ASSERT_OK(kernel_softmax(&tc.ctx, &node, NULL));

    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(isfinite(y[i]));
        sum += y[i];
    }
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
    return 0;
}

int main(void) {
    printf("test_kernel_softmax:\n");
    RUN_TEST(test_softmax_basic);
    RUN_TEST(test_softmax_large_values);
    TEST_SUMMARY();
}
