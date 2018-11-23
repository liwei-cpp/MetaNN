#pragma once

#include <data/batch_sequence/test_static_batch_sequence.h>
#include <data/batch_sequence/test_dynamic_batch_sequence.h>

namespace Test::Data::BatchSequence
{
    inline void test_batch_sequence_pack()
    {
        test_static_batch_sequence();
        test_dynamic_batch_sequence();
    }
}
