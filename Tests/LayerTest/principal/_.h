#pragma once
#include <principal/math/_.h>
#include <principal/nn/_.h>
#include <principal/others/_.h>
#include <principal/tensor/_.h>
namespace Test::Layer
{
    void test_principal()
    {
        Principal::test_math();
        Principal::test_nn();
        Principal::test_others();
        Principal::test_tensor();
    }
}