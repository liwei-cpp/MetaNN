#pragma once
#include <operators/test_duplicate.h>
#include <operators/test_collapse.h>

// mutation operators
#include <operators/test_transpose.h>

// blas operators
#include <operators/test_dot.h>

#include <operators/activation/test_activation_operators.h>
#include <operators/elementwise/test_elementwise_operators.h>
#include <operators/loss/test_loss_operators.h>

namespace Test::Operators
{
    void test_operators()
    {
        test_duplicate();
        test_collapse();
        
        // mutation operators
        test_transpose();
        
        // blas operators
        test_dot();
        
        test_activation_operators();
        test_elementwise_operators();
        test_loss_operators();
    }
}