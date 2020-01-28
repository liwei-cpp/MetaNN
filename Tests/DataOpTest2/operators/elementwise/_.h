#pragma once

namespace Test::Operators::Elwentwise
{
    void test_abs();
    void test_acos();   void test_acos_grad();
    void test_asin();   void test_asin_grad();
    void test_negative();
    void test_sign();

    void test()
    {
        test_abs();
        test_acos();    test_acos_grad();
        test_asin();    test_asin_grad();
        test_negative();
        test_sign();
    }
}