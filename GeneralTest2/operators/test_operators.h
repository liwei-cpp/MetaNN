#pragma once
#include <operators/test_duplicate.h>
#include <operators/test_collapse.h>

// elementwise operators
#include <operators/test_add.h>
#include <operators/test_asin.h>
#include <operators/test_asin_grad.h>
#include <operators/test_divide.h>
#include <operators/test_interpolate.h>
#include <operators/test_multiply.h>
#include <operators/test_sign.h>
#include <operators/test_substract.h>

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
        
        // elementwise operators
        test_abs();
        test_acos();        test_acos_grad();
        test_add();
        test_asin();        test_asin_grad();
        test_divide();
        test_interpolate();
        test_multiply();
        test_sign();
        test_substract();
        
        // mutation operators
        test_transpose();
        
        // blas operators
        test_dot();
        
        test_activation_operators();
        test_elementwise_operators();
        test_loss_operators();
    }
}