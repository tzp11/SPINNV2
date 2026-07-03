#include "test_helpers.h"
#include "kernel_common.h"
#include "spkv2_runtime.h"

int spkv2_execute_node(Spkv2Context *ctx, const Spkv2NodeRecord *node);

static int custom_relu_42(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2TensorState *out = &ctx->tensors[node->outputs[0]];
    size_t n = spkv2_kernel_elem_count(out->record);
    float *dst = (float *)out->data;
    for (size_t i = 0; i < n; i++) dst[i] = 42.0f;
    return 0;
}

static int custom_relu_99(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)scratch;
    Spkv2TensorState *out = &ctx->tensors[node->outputs[0]];
    size_t n = spkv2_kernel_elem_count(out->record);
    float *dst = (float *)out->data;
    for (size_t i = 0; i < n; i++) dst[i] = 99.0f;
    return 0;
}

static void setup_relu(TestContext *tc, Spkv2AttrRecord *attr,
                       Spkv2NodeRecord *node,
                       float *x, float *y, int n) {
    init_attr_record(attr, SPKV2_OP_RELU);
    init_test_context(tc, 2, attr, sizeof(*attr));
    uint32_t shape[] = {1, 1, 1, (uint32_t)n};
    init_tensor_record(&tc->tensor_records[0], 0, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, shape, n * sizeof(float));
    init_tensor_record(&tc->tensor_records[1], 1, SPKV2_DTYPE_FP32,
                       SPKV2_ROLE_ACTIVATION, 4, shape, n * sizeof(float));
    set_tensor_data(tc, 0, x);
    set_tensor_data(tc, 1, y);
    uint32_t inputs[] = {0};
    uint32_t outputs[] = {1};
    init_node_record(node, 0, SPKV2_OP_RELU, 1, inputs, 1, outputs);
    node->kernel_spec_id = 0xFFFFFFFFu;
}

static int test_register_and_dispatch(void) {
    spkv2_clear_user_kernels();
    ASSERT_OK(spkv2_register_kernel(
        SPKV2_OP_RELU, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE,
        custom_relu_42, 10));

    TestContext tc;
    Spkv2AttrRecord attr;
    Spkv2NodeRecord node;
    float x[4] = {-1, 0, 1, 2};
    float y[4] = {0};
    setup_relu(&tc, &attr, &node, x, y, 4);

    ASSERT_OK(spkv2_execute_node(&tc.ctx, &node));
    ASSERT_NEAR(y[0], 42.0f, 1e-7f);
    ASSERT_NEAR(y[3], 42.0f, 1e-7f);

    spkv2_clear_user_kernels();
    return 0;
}

static int test_priority_override(void) {
    spkv2_clear_user_kernels();
    ASSERT_OK(spkv2_register_kernel(
        SPKV2_OP_RELU, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE,
        custom_relu_42, 5));
    ASSERT_OK(spkv2_register_kernel(
        SPKV2_OP_RELU, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE,
        custom_relu_99, 50));

    TestContext tc;
    Spkv2AttrRecord attr;
    Spkv2NodeRecord node;
    float x[4] = {1, 2, 3, 4};
    float y[4] = {0};
    setup_relu(&tc, &attr, &node, x, y, 4);

    ASSERT_OK(spkv2_execute_node(&tc.ctx, &node));
    ASSERT_NEAR(y[0], 99.0f, 1e-7f);

    spkv2_clear_user_kernels();
    return 0;
}

static int test_user_overrides_builtin(void) {
    spkv2_clear_user_kernels();
    ASSERT_OK(spkv2_register_kernel(
        SPKV2_OP_RELU, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE,
        custom_relu_42, 100));

    TestContext tc;
    Spkv2AttrRecord attr;
    Spkv2NodeRecord node;
    float x[4] = {-1, 0, 1, 2};
    float y[4] = {0};
    setup_relu(&tc, &attr, &node, x, y, 4);

    ASSERT_OK(spkv2_execute_node(&tc.ctx, &node));
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(y[i], 42.0f, 1e-7f);

    spkv2_clear_user_kernels();
    return 0;
}

static int test_clear(void) {
    spkv2_clear_user_kernels();
    ASSERT_OK(spkv2_register_kernel(
        SPKV2_OP_RELU, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE,
        custom_relu_42, 100));
    spkv2_clear_user_kernels();

    TestContext tc;
    Spkv2AttrRecord attr;
    Spkv2NodeRecord node;
    float x[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
    float y[4] = {0};
    setup_relu(&tc, &attr, &node, x, y, 4);

    ASSERT_OK(spkv2_execute_node(&tc.ctx, &node));
    ASSERT_NEAR(y[0], 0.0f, 1e-7f);
    ASSERT_NEAR(y[2], 1.0f, 1e-7f);
    ASSERT_NEAR(y[3], 2.0f, 1e-7f);
    return 0;
}

static int dummy_kernel(Spkv2Context *ctx, const Spkv2NodeRecord *node, void *scratch) {
    (void)ctx; (void)node; (void)scratch;
    return 0;
}

static int test_full_capacity(void) {
    spkv2_clear_user_kernels();
    for (int i = 0; i < 64; i++) {
        int rc = spkv2_register_kernel(
            (uint16_t)(200 + i), SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE,
            dummy_kernel, 1);
        ASSERT_EQ(rc, 0);
    }
    int rc = spkv2_register_kernel(
        999, SPKV2_BACKEND_REF, SPKV2_KERNEL_REFERENCE, dummy_kernel, 1);
    ASSERT_EQ(rc, -2);

    spkv2_clear_user_kernels();
    return 0;
}

int main(void) {
    printf("test_register_kernel:\n");
    RUN_TEST(test_register_and_dispatch);
    RUN_TEST(test_priority_override);
    RUN_TEST(test_user_overrides_builtin);
    RUN_TEST(test_clear);
    RUN_TEST(test_full_capacity);
    TEST_SUMMARY();
}
