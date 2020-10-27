#pragma once

namespace Test::Operation::Tensor
{
    void test_dot();
    void test_permute();
    void test_reshape();
    void test_slice();
    void test_tile();

    void test()
    {
        test_dot();
        test_permute();
        test_reshape();
        test_slice();
        test_tile();
    }
}