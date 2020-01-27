#pragma once
#include <operators/activation/_.h>
#include <operators/mutating/_.h>

namespace Test::Operators
{
    void test()
    {
        Test::Operators::Activation::test();
        Test::Operators::Mutating::test();
    }
}