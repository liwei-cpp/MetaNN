#pragma once
#include <elementary/test_abs_layer.h>
#include <elementary/test_add_layer.h>
#include <elementary/test_bias_layer.h>
#include <elementary/test_element_mul_layer.h>
#include <elementary/test_sigmoid_layer.h>
#include <elementary/test_tanh_layer.h>
#include <elementary/test_weight_layer.h>

namespace Test::Layer
{
    inline void test_elementary()
    {
        test_abs_layer();
        test_add_layer();
        test_bias_layer();
        test_element_mul_layer();
        test_sigmoid_layer();
        test_tanh_layer();
        test_weight_layer();
    }
}