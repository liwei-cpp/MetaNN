#pragma once

#include <data/cardinal/matrix/test_matrix.h>
#include <data/cardinal/matrix/test_trival_matrix.h>

#include <data/cardinal/matrix/test_vector.h>
#include <data/cardinal/matrix/test_one_hot_vector.h>

namespace Test::Data::Cardinal::Matrix
{
    inline void test_matrix_pack()
    {
        test_matrix();
        test_trival_matrix();
        test_vector();
        test_one_hot_vector();
    }
}
