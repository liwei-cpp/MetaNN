#pragma once

namespace Test::Operators
{
    void test_collapse();
    void test_duplicate();

    void test_cate_trans_operators()
    {
        test_collapse();
        test_duplicate();
    }
}