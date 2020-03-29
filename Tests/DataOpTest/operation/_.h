#pragma once
#include <operation/activation/_.h>
#include <operation/blas/_.h>
#include <operation/elementwise/_.h>
#include <operation/loss/_.h>
#include <operation/math/_.h>
#include <operation/mutating/_.h>
#include <operation/tensor/_.h>

namespace Test::Operation
{
    void test()
    {
        Test::Operation::Activation::test();
        Test::Operation::Blas::test();
        Test::Operation::Elwentwise::test();
        Test::Operation::Loss::test();
        Test::Operation::Math::test();
        Test::Operation::Mutating::test();
        Test::Operation::Tensor::test();
    }
}