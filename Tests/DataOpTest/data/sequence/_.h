#pragma once

namespace Test::Data::Sequence
{
    void test_static_sequence();
    void test_dynamic_sequence();

    inline void test()
    {
        test_static_sequence();
        test_dynamic_sequence();
    }
}
