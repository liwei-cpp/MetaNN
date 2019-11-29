#pragma once

#include <data/cardinal/_.h>
#include <data/batch/_.h>
#include <data/general/_.h>
#include <data/sequence/_.h>
#include <data/batch_sequence/_.h>

namespace Test::Data
{
    inline void test()
    {
        Cardinal::test();
        Batch::test();
        General::test();
        Sequence::test();
        BatchSequence::test();
    }
}