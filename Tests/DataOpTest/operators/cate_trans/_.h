#pragma once

namespace Test::Operators::CateTrans
{
    void test_collapse();
    void test_duplicate();
    void test_slice();
    void test()
    {
        test_collapse();
        test_duplicate();
        test_slice();
    }
}