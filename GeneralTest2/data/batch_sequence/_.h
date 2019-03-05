#pragma once

namespace Test::Data::BatchSequence
{
    void test_static_batch_sequence();
    void test_dynamic_batch_sequence();
    
    inline void test()
    {
        test_static_batch_sequence();
        test_dynamic_batch_sequence();
    }
}
