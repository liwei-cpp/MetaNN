namespace Test::Layer::Principal
{
    void test_nll_loss_layer();
    void test_relu_layer();
    void test_softmax_layer();

    void test_nn()
    {
        test_nll_loss_layer();
        test_relu_layer();
        test_softmax_layer();
    }
}