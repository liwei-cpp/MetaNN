#pragma once

#include <data/cardinal/test_cardinal_pack.h>
#include <data/batch/test_batch_pack.h>
#include <data/general/_.h>
#include <data/sequence/test_sequence_pack.h>
#include <data/batch_sequence/test_batch_sequence_pack.h>

namespace Test::Data
{
    inline void test_data()
    {
        Cardinal::test_cardinal_pack();
        Batch::test_batch_pack();
        General::test();
        Sequence::test_sequence_pack();
        BatchSequence::test_batch_sequence_pack();
    }
}