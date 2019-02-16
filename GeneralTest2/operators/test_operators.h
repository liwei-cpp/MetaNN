#pragma once
#include <operators/activation/test_activation_operators.h>
#include <operators/blas/test_blas_operators.h>
#include <operators/cate_trans/test_cate_trans_operators.h>
#include <operators/elementwise/test_elementwise_operators.h>
#include <operators/mutating/test_mutating_operators.h>
#include <operators/loss/test_loss_operators.h>

namespace Test::Operators
{
    void test_operators()
    {
        test_activation_operators();
        test_blas_operators();
        test_cate_trans_operators();
        test_elementwise_operators();
        test_mutating_operators();
        test_loss_operators();
    }
}