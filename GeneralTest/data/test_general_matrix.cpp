#include "test_general_matrix.h"
#include "../facilities/calculate_tags.h"
#include <iostream>
#include <cassert>
#include <set>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
void TestMatrix1()
{
    cout << "Test general matrix case 1...\t";
    static_assert(IsMatrix<Matrix<CheckElement, CheckDevice>>, "Test Error");
    static_assert(IsMatrix<Matrix<CheckElement, CheckDevice>&>, "Test Error");
    static_assert(IsMatrix<Matrix<CheckElement, CheckDevice>&&>, "Test Error");
    static_assert(IsMatrix<const Matrix<CheckElement, CheckDevice>&>, "Test Error");
    static_assert(IsMatrix<const Matrix<CheckElement, CheckDevice>&&>, "Test Error");

    Matrix<CheckElement, CheckDevice> rm;
    assert(rm.RowNum() == 0);
    assert(rm.ColNum() == 0);

    rm = Matrix<CheckElement, CheckDevice>(10, 20);
    assert(rm.RowNum() == 10);
    assert(rm.ColNum() == 20);

    int c = 0;
    for (size_t i=0; i<10; ++i)
    {
        for (size_t j=0; j<20; ++j)
        {
            rm.SetValue(i, j, (float)(c++));
        }
    }

    const Matrix<CheckElement, CheckDevice> rm2 = rm;
    c = 0;
    for (size_t i=0; i<10; ++i)
    {
        for (size_t j=0; j<20; ++j)
            assert(rm2(i, j) == c++);
    }
    cout << "done" << endl;
}

void TestMatrix2()
{
    cout << "Test general matrix case 2...\t";
    auto rm1 = Matrix<CheckElement, CheckDevice>(10, 20);
    int c = 0;
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            rm1.SetValue(i, j, (float)(c++));
        }
    }

    auto rm2 = Matrix<CheckElement, CheckDevice>(3, 7);
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 7; ++j)
        {
            rm2.SetValue(i, j, (float)(c++));
        }
    }
    cout << "done" << endl;
}
}

void test_general_matrix()
{
    TestMatrix1();
    TestMatrix2();
}