#pragma once

namespace Test::Operation::NN
{
    void test_nll_loss();   void test_nll_loss_grad();
    void test_relu();       void test_relu_grad();
    void test_softmax();    void test_softmax_grad();

    void test()
    {
        test_relu();        test_relu_grad();
        test_softmax();     test_softmax_grad();
    }
}