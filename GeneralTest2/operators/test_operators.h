#pragma once
#include <operators/test_duplicate.h>
#include <operators/test_collapse.h>

#include <operators/test_abs.h>
#include <operators/test_sign.h>

namespace Test::Operators
{
    void test_operators()
    {
        test_duplicate();
        test_collapse();
        
        test_abs();
        test_sign();
    }
}