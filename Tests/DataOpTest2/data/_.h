#pragma once

#include <data/facilities/_.h>

namespace Test::Data
{
    void test_scalar();     void test_tensor();

    void test_trival_tensor();
    void test_zero_tensor();
    inline void test()
    {
        Facilities::test();
        test_scalar();
        test_tensor();
        
        test_trival_tensor();
        test_zero_tensor();
    }
}