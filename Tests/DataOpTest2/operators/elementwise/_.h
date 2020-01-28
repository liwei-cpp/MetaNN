#pragma once

namespace Test::Operators::Elwentwise
{
    void test_abs();
    void test_acos();   void test_acos_grad();
    void test_sign();

    void test()
    {
        test_abs();
        test_acos();    test_acos_grad();
        test_sign();
    }
}