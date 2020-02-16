#pragma once

namespace Test::Operators::Tensor
{
    void test_slice();
    void test_tile();

    void test()
    {
        test_slice();
        test_tile();
    }
}