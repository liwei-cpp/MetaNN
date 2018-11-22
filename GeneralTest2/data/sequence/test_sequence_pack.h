#pragma once

#include <data/sequence/test_static_sequence.h>
#include <data/sequence/test_dynamic_sequence.h>

namespace Test::Data::Sequence
{
    inline void test_sequence_pack()
    {
        test_static_sequence();
        test_dynamic_sequence();
    }
}
