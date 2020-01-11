#pragma once

#include <data/facilities/_.h>

namespace Test::Data
{
    void test_scalar();

    inline void test()
    {
        Facilities::test();
        test_scalar();
    }
}