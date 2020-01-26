#pragma once

namespace Test::Operators::Activation
{
    void test_relu();       void test_relu_grad();

    void test()
    {
        test_relu();        test_relu_grad();
    }
}