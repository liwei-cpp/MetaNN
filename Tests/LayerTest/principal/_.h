#pragma once
namespace Test::Layer
{
    namespace Principal
    {
        void test_math();
        void test_nn();
        void test_others();
        void test_tensor();
    }

    void test_principal()
    {
        Principal::test_math();
        Principal::test_nn();
        Principal::test_others();
        Principal::test_tensor();
    }
}