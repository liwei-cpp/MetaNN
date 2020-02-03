#pragma once
#include <operators/activation/_.h>
#include <operators/elementwise/_.h>
#include <operators/math/_.h>
#include <operators/mutating/_.h>

namespace Test::Operators
{
    void test()
    {
        Test::Operators::Activation::test();
        Test::Operators::Elwentwise::test();
        Test::Operators::Math::test();
        Test::Operators::Mutating::test();
    }
}