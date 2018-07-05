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
void Test3DArray1()
{
    cout << "Test 3d array case 1...\t";
    static_assert(IsThreeDArray<ThreeDArray<CheckElement, CheckDevice>>, "Test Error");
    static_assert(IsThreeDArray<ThreeDArray<CheckElement, CheckDevice>&>, "Test Error");
    static_assert(IsThreeDArray<ThreeDArray<CheckElement, CheckDevice>&&>, "Test Error");
    static_assert(IsThreeDArray<const ThreeDArray<CheckElement, CheckDevice>&>, "Test Error");
    static_assert(IsThreeDArray<const ThreeDArray<CheckElement, CheckDevice>&&>, "Test Error");

    ThreeDArray<CheckElement, CheckDevice> rm;
    assert(rm.PageNum() == 0);
    assert(rm.RowNum() == 0);
    assert(rm.ColNum() == 0);

    rm = ThreeDArray<CheckElement, CheckDevice>(5, 10, 20);
    assert(rm.PageNum() == 5);
    assert(rm.RowNum() == 10);
    assert(rm.ColNum() == 20);

    int c = 0;
    for (size_t p = 0; p < 5; ++p)
    {
        for (size_t i=0; i<10; ++i)
        {
            for (size_t j=0; j<20; ++j)
            {
                rm.SetValue(p, i, j, (float)(c++));
            }
        }
    }

    const ThreeDArray<CheckElement, CheckDevice> rm2 = rm;
    c = 0;
    for (size_t p = 0; p < 5; ++p)
    {
        for (size_t i=0; i<10; ++i)
        {
            for (size_t j=0; j<20; ++j)
                assert(rm2(p, i, j) == c++);
        }
    }

    auto evalHandle = rm.EvalRegister();
    auto cm = evalHandle.Data();

    for (size_t p = 0; p < cm.PageNum(); ++p)
    {
        for (size_t i=0; i < cm.RowNum(); ++i)
        {
            for (size_t j = 0; j < cm.ColNum(); ++j)
            {
                assert(cm(p, i, j) == rm(p, i, j));
            }
        }
    }
    cout << "done" << endl;
}
}

void test_3d_array()
{
    Test3DArray1();
}