#pragma once

namespace Test::Data::Batch
{
    void test_static_batch();
    void test_dynamic_batch();
    
    inline void test()
    {
        test_static_batch();
        test_dynamic_batch();
    }
}
