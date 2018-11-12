#pragma once

#include <data/cardinal/scalar/test_scalar_pack.h>
#include <data/cardinal/matrix/test_matrix_pack.h>

namespace Test::Data::Cardinal
{
    inline void test_cardinal_pack()
    {
        Scalar::test_scalar_pack();
        Matrix::test_matrix_pack();
    }
}
