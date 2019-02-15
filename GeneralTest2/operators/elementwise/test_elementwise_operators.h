#pragma once

namespace Test::Operators
{
    void test_abs();
    void test_acos();   void test_acos_grad();

    void test_elementwise_operators()
    {
        test_abs();
        test_acos();    test_acos_grad();
    }
}