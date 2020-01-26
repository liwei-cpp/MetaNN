#pragma once

namespace Test::Operators::Activation
{
    void test_relu();       void test_relu_grad();
    void test_sigmoid();    void test_sigmoid_grad();
    void test_tanh();       void test_tanh_grad();

    void test()
    {
        test_relu();        test_relu_grad();
        test_sigmoid();     test_sigmoid_grad();
        test_tanh();        test_tanh_grad();
    }
}