#pragma once

#include <data/cardinal/scalar/_.h>
#include <data/cardinal/matrix/_.h>
#include <data/cardinal/3d_array/_.h>

namespace Test::Data::Cardinal
{
    inline void test()
    {
        Scalar::test();
        Matrix::test();
        ThreeDArray::test();
    }
}
