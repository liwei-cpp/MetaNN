#pragma once

#include <data/facilities/_.h>

namespace Test::Data
{
    void test_scalar();     void test_tensor();

    inline void test()
    {
        Facilities::test();
        test_scalar();
        test_tensor();
    }
}