namespace Test::Layer
{
    namespace Composite
    {
        void test_compose_kenrel();

        void test_bias_layer();
        void test_linear_layer();
        void test_single_layer_perceptron();
        void test_weight_layer();
    }

    void test_composite()
    {
        Composite::test_compose_kenrel();

        Composite::test_bias_layer();
        Composite::test_linear_layer();
        Composite::test_single_layer_perceptron();
        Composite::test_weight_layer();
    }
}