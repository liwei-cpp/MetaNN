namespace Test::Layer::Compose
{
    void test_compose_kenrel();
    void test_bias_layer();
    void test_weight_layer();
    void test_linear_layer();
    void test_single_layer_perceptron();
    
    void Test()
    {
        test_compose_kenrel();
        
        test_bias_layer();
        test_weight_layer();
        test_linear_layer();
        test_single_layer_perceptron();
    }
}