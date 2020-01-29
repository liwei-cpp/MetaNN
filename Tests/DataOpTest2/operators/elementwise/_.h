#pragma once

namespace Test::Operators::Elwentwise
{
    void test_abs();
    void test_acos();   void test_acos_grad();
    void test_add();
    void test_asin();   void test_asin_grad();
    void test_multiply();
    void test_negative();
    void test_sign();
    void test_substract();

    void test()
    {
        test_abs();
        test_acos();    test_acos_grad();
        test_add();
        test_asin();    test_asin_grad();
        test_multiply();
        test_negative();
        test_sign();
        test_substract();
    }
}