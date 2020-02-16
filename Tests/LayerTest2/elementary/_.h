namespace Test::Layer::Elementary
{
    void test_abs_layer();
    void test_relu_layer();
    void test_sigmoid_layer();
    void test_tanh_layer();

    void Test()
    {
        test_abs_layer();
        test_relu_layer();
        test_sigmoid_layer();
        test_tanh_layer();
    }
}