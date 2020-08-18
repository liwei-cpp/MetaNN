namespace Test::Layer::Principal
{
    void test_abs_layer();
    void test_acos_layer();
    void test_add_layer();
    void test_asin_layer();
    void test_multiply_layer();
    void test_sigmoid_layer();
    void test_substract_layer();
    void test_tanh_layer();

    void test_math()
    {
        test_abs_layer();
        test_acos_layer();
        test_add_layer();
        test_asin_layer();
        test_multiply_layer();
        test_sigmoid_layer();
        test_substract_layer();
        test_tanh_layer();
    }
}