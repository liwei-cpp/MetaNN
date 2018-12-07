#pragma once
#include <operators/test_duplicate.h>
#include <operators/test_collapse.h>

namespace Test::Operators
{
    void test_operators()
    {
        test_duplicate();
        test_collapse();
    }
}