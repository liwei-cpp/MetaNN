#pragma once

#include <data/batch/test_static_batch.h>
#include <data/batch/test_dynamic_batch.h>

namespace Test::Data::Batch
{
    inline void test_batch_pack()
    {
        test_static_batch();
        test_dynamic_batch();
    }
}
