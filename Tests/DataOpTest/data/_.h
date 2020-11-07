#pragma once

#include <data/facilities/_.h>

namespace Test::Data
{
    void test_scalar();     void test_tensor();

    void test_bias_vector();
    void test_trivial_tensor();
    void test_zero_tensor();

    void test_dynamic();
    void test_scalable_tensor();

    inline void test()
    {
        Facilities::test();

        test_scalar();
        test_tensor();

        test_bias_vector();
        test_trivial_tensor();
        test_zero_tensor();

        test_dynamic();
        test_scalable_tensor();
    }
}