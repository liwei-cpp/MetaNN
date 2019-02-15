#pragma once

namespace Test::Operators
{
    void test_relu();
    void test_sigmoid();    void test_sigmoid_grad();
    void test_softmax();    void test_softmax_grad();
    void test_tanh();       void test_tanh_grad();

    void test_activation_operators()
    {
        test_relu();
        test_sigmoid();     test_sigmoid_grad();
        test_softmax();     test_softmax_grad();
        test_tanh();        test_tanh_grad();
    }
}