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
void TestVector1()
{
    cout << "Test general vector case 1...\t";
    static_assert(IsMatrix<Vector<CheckElement, CheckDevice>>, "Test Error");
    static_assert(IsMatrix<Vector<CheckElement, CheckDevice>&>, "Test Error");
    static_assert(IsMatrix<Vector<CheckElement, CheckDevice>&&>, "Test Error");
    static_assert(IsMatrix<const Vector<CheckElement, CheckDevice>&>, "Test Error");
    static_assert(IsMatrix<const Vector<CheckElement, CheckDevice>&&>, "Test Error");

    Vector<CheckElement, CheckDevice> rm;
    assert(rm.RowNum() == 1);
    assert(rm.ColNum() == 0);

    rm = Vector<CheckElement, CheckDevice>(20);
    assert(rm.RowNum() == 1);
    assert(rm.ColNum() == 20);

    int c = 0;
    for (size_t j=0; j<20; ++j)
    {
        rm.SetValue(j, (float)(c++));
    }

    const Vector<CheckElement, CheckDevice> rm2 = rm;
    c = 0;
    for (size_t j=0; j<20; ++j)
        assert(rm2(j) == c++);
    cout << "done" << endl;
}

void TestVector2()
{
    cout << "Test general matrix case 2...\t";
    auto rm1 = Vector<CheckElement, CheckDevice>(20);
    int c = 0;
    for (size_t j = 0; j < 20; ++j)
    {
        rm1.SetValue(j, (float)(c++));
    }
    
    Matrix<CheckElement, CheckDevice> res = Evaluate(rm1);
    assert(res.RowNum() == 1);
    assert(res.ColNum() == 20);
    c = 0;
    for (size_t j = 0; j < 20; ++j)
    {
        assert(res(0, j) == c++);
    }
    cout << "done" << endl;
}
}

void test_general_vector()
{
    TestVector1();
    TestVector2();
}