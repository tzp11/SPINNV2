#include "test_helpers.h"
#include "reference_kernels.h"

static int test_maxpool_2x2_s2(void) {
    TestContext tc;
    Spkv2AttrRecord attr;
    init_attr_record(&attr, SPKV2_OP_MAXPOOL);
    attr.kernel_shape[0] = 2;
    attr.kernel_shape[1] = 2;
    attr.strides[0] = 2;
    attr.strides[1] = 2;
    attr.pads[0] = 0; attr.pads[1] = 0; attr.pads[2] = 0; attr.pads[3] = 0;
    init_test_context(&tc, 2, &attr, sizeof(attr));

    uint32_t x_shape[] = {1, 1, 4, 4};
    uint32_t y_shape[] = {1, 1, 2, 2};
    init_tensor_record(&tc.tensor_records[0], 0, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, x_shape, 16 * sizeof(float));
    init_tensor_record(&tc.tensor_records[1], 1, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, y_shape, 4 * sizeof(float));

    float x[] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    float y[4] = {0};
    set_tensor_data(&tc, 0, x);
    set_tensor_data(&tc, 1, y);

    Spkv2NodeRecord node;
    uint32_t inputs[] = {0};
    uint32_t outputs[] = {1};
    init_node_record(&node, 0, SPKV2_OP_MAXPOOL, 1, inputs, 1, outputs);
    node.attr_offset = 0;
    node.attr_size = sizeof(Spkv2AttrRecord);

    ASSERT_OK(kernel_maxpool(&tc.ctx, &node, NULL));

    ASSERT_NEAR(y[0],  6.0f, 1e-7f);
    ASSERT_NEAR(y[1],  8.0f, 1e-7f);
    ASSERT_NEAR(y[2], 14.0f, 1e-7f);
    ASSERT_NEAR(y[3], 16.0f, 1e-7f);
    return 0;
}

static int test_maxpool_with_negatives(void) {
    TestContext tc;
    Spkv2AttrRecord attr;
    init_attr_record(&attr, SPKV2_OP_MAXPOOL);
    attr.kernel_shape[0] = 2;
    attr.kernel_shape[1] = 2;
    attr.strides[0] = 2;
    attr.strides[1] = 2;
    init_test_context(&tc, 2, &attr, sizeof(attr));

    uint32_t x_shape[] = {1, 1, 4, 4};
    uint32_t y_shape[] = {1, 1, 2, 2};
    init_tensor_record(&tc.tensor_records[0], 0, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, x_shape, 16 * sizeof(float));
    init_tensor_record(&tc.tensor_records[1], 1, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, y_shape, 4 * sizeof(float));

    float x[] = {
        -5, -3, -1, -7,
        -2, -4, -6, -8,
        -9, -1, -3, -5,
        -7, -2, -4, -6
    };
    float y[4] = {0};
    set_tensor_data(&tc, 0, x);
    set_tensor_data(&tc, 1, y);

    Spkv2NodeRecord node;
    uint32_t inputs[] = {0};
    uint32_t outputs[] = {1};
    init_node_record(&node, 0, SPKV2_OP_MAXPOOL, 1, inputs, 1, outputs);
    node.attr_offset = 0;
    node.attr_size = sizeof(Spkv2AttrRecord);

    ASSERT_OK(kernel_maxpool(&tc.ctx, &node, NULL));

    ASSERT_NEAR(y[0], -2.0f, 1e-7f);
    ASSERT_NEAR(y[1], -1.0f, 1e-7f);
    ASSERT_NEAR(y[2], -1.0f, 1e-7f);
    ASSERT_NEAR(y[3], -3.0f, 1e-7f);
    return 0;
}

int main(void) {
    printf("test_kernel_maxpool:\n");
    RUN_TEST(test_maxpool_2x2_s2);
    RUN_TEST(test_maxpool_with_negatives);
    TEST_SUMMARY();
}
