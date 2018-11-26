#pragma once

#include <data/cardinal/test_cardinal_pack.h>
#include <data/batch/test_batch_pack.h>
#include <data/sequence/test_sequence_pack.h>
#include <data/batch_sequence/test_batch_sequence_pack.h>
#include <data/test_dynamic.h>

namespace Test::Data
{
    inline void test_data()
    {
        Cardinal::test_cardinal_pack();
        Batch::test_batch_pack();
        Sequence::test_sequence_pack();
        BatchSequence::test_batch_sequence_pack();
        test_dynamic();
    }
}