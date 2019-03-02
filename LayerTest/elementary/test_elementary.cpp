namespace Test::Layer
{
    void test_abs_layer();
    void test_add_layer();
    void test_bias_layer();
    void test_multiply_layer();
    void test_interpolate_layer();
    void test_relu_layer();
    void test_sigmoid_layer();
    void test_softmax_layer();
    void test_tanh_layer();
    void test_transpose_layer();
    void test_weight_layer();
    
    void test_elementary()
    {
        test_abs_layer();
        test_add_layer();
        test_bias_layer();
        test_multiply_layer();
        test_interpolate_layer();
        test_relu_layer();
        test_sigmoid_layer();
        test_softmax_layer();
        test_tanh_layer();
        test_transpose_layer();
        test_weight_layer();
    }
}