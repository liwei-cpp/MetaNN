#pragma once
#include <operation/math/_.h>
#include <operation/nn/_.h>
#include <operation/others/_.h>
#include <operation/tensor/_.h>

namespace Test::Operation
{
    void test()
    {
        Test::Operation::Math::test();
        Test::Operation::NN::test();
        Test::Operation::Others::test();
        Test::Operation::Tensor::test();
    }
}