#pragma once

#include <data/cardinal/matrix/test_matrix.h>
#include <data/cardinal/matrix/test_zero_matrix.h>
#include <data/cardinal/matrix/test_trival_matrix.h>

namespace Test::Data::Cardinal::Matrix
{
    inline void test_matrix_pack()
    {
        test_matrix();
        test_zero_matrix();
        test_trival_matrix();
    }
}
